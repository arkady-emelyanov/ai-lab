// fakebmc emulates the Redfish service of a BMC, modelled on NVIDIA's GB200
// BMCs (NVIDIA/bmcweb), in one of two roles:
//   - tray: a GPU compute tray: system power, GPUs and their NVLink ports.
//     Port changes reach the tray's fake GPUs through a sideband directory;
//     power actions go through the Incus API.
//   - nvswitch: an NVLink switch tray: the NVLink fabric, NVSwitch chips and
//     their ports, cabled to the GPU trays' NVLinks.
package main

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"flag"
	"log"
	"math/big"
	"net/http"
	"os"
	"path/filepath"
	"time"
)

const version = "1.0.0"

func main() {
	configPath := flag.String("config", "/etc/fakebmc/config.json", "configuration file")
	flag.Parse()

	cfg, err := loadConfig(*configPath)
	if err != nil {
		log.Fatal(err)
	}
	var (
		state *State
		power *Power
		sw    *switchState
		name  = cfg.Tray
	)
	if cfg.Role == "nvswitch" {
		name = "nvswitch"
		if sw, err = loadSwitchState(cfg.StateDir); err != nil {
			log.Fatal(err)
		}
	} else {
		if state, err = loadState(cfg.StateDir, cfg.SidebandDir); err != nil {
			log.Fatal(err)
		}
		if power, err = newPower(cfg.Incus, cfg.Tray); err != nil {
			log.Fatal(err)
		}
	}
	cert, err := serverCertificate(cfg.StateDir, name)
	if err != nil {
		log.Fatal(err)
	}
	server := newServer(cfg, state, power, sw)
	if sw != nil {
		if err := server.publishSwitchLinks(); err != nil { // trays see current port state
			log.Fatal(err)
		}
	}

	srv := &http.Server{
		Addr:              cfg.Listen,
		Handler:           server.routes(),
		TLSConfig:         &tls.Config{Certificates: []tls.Certificate{cert}, MinVersion: tls.VersionTLS12},
		ReadHeaderTimeout: 10 * time.Second,
	}
	log.Printf("fakebmc %s: Redfish (%s) for %s on https://%s", version, cfg.Role, name, cfg.Listen)
	log.Fatal(srv.ListenAndServeTLS("", ""))
}

// serverCertificate loads the BMC's self-signed TLS certificate, creating it
// on first start, as a factory-fresh BMC does.
func serverCertificate(dir, host string) (tls.Certificate, error) {
	certPath, keyPath := filepath.Join(dir, "https.crt"), filepath.Join(dir, "https.key")
	if c, err := tls.LoadX509KeyPair(certPath, keyPath); err == nil {
		return c, nil
	}
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		return tls.Certificate{}, err
	}
	serial, _ := rand.Int(rand.Reader, new(big.Int).Lsh(big.NewInt(1), 62))
	tmpl := &x509.Certificate{
		SerialNumber: serial,
		Subject:      pkix.Name{CommonName: host + "-bmc", Organization: []string{"fakebmc"}},
		DNSNames:     []string{host + "-bmc", "localhost"},
		NotBefore:    time.Now().Add(-time.Hour),
		NotAfter:     time.Now().AddDate(10, 0, 0),
		KeyUsage:     x509.KeyUsageDigitalSignature,
		ExtKeyUsage:  []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
	}
	der, err := x509.CreateCertificate(rand.Reader, tmpl, tmpl, &key.PublicKey, key)
	if err != nil {
		return tls.Certificate{}, err
	}
	keyDER, err := x509.MarshalECPrivateKey(key)
	if err != nil {
		return tls.Certificate{}, err
	}
	if err := os.WriteFile(certPath, pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der}), 0o644); err != nil {
		return tls.Certificate{}, err
	}
	if err := os.WriteFile(keyPath, pem.EncodeToMemory(&pem.Block{Type: "EC PRIVATE KEY", Bytes: keyDER}), 0o600); err != nil {
		return tls.Certificate{}, err
	}
	return tls.LoadX509KeyPair(certPath, keyPath)
}
