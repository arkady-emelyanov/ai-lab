package main

import (
	"bytes"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"encoding/pem"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"time"
)

// Power controls the tray through the Incus API, the emulation's equivalent
// of the BMC's power rails. The client certificate is restricted to the
// project holding the GPU trays.
type Power struct {
	cfg    Incus
	tray   string
	client *http.Client
}

func newPower(cfg Incus, tray string) (*Power, error) {
	if cfg.URL == "" {
		return &Power{tray: tray}, nil // power control disabled
	}
	cert, err := tls.LoadX509KeyPair(cfg.Cert, cfg.Key)
	if err != nil {
		return nil, fmt.Errorf("incus client certificate: %w", err)
	}
	serverPEM, err := os.ReadFile(cfg.ServerCert)
	if err != nil {
		return nil, fmt.Errorf("incus server certificate: %w", err)
	}
	block, _ := pem.Decode(serverPEM)
	if block == nil {
		return nil, fmt.Errorf("incus server certificate: no PEM data")
	}
	pinned := block.Bytes
	// Incus' self-signed certificate is pinned rather than verified by name.
	tlsCfg := &tls.Config{
		Certificates:       []tls.Certificate{cert},
		InsecureSkipVerify: true,
		VerifyPeerCertificate: func(raw [][]byte, _ [][]*x509.Certificate) error {
			if len(raw) == 0 || !bytes.Equal(raw[0], pinned) {
				return fmt.Errorf("incus: server certificate does not match the pinned one")
			}
			return nil
		},
	}
	return &Power{cfg: cfg, tray: tray, client: &http.Client{
		Timeout:   90 * time.Second,
		Transport: &http.Transport{TLSClientConfig: tlsCfg},
	}}, nil
}

func (p *Power) enabled() bool { return p.client != nil }

func (p *Power) do(method, path string, body any) (map[string]any, error) {
	var rd io.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			return nil, err
		}
		rd = bytes.NewReader(b)
	}
	u := p.cfg.URL + path
	if p.cfg.Project != "" {
		sep := "?"
		if bytes.ContainsRune([]byte(path), '?') {
			sep = "&"
		}
		u += sep + "project=" + url.QueryEscape(p.cfg.Project)
	}
	req, err := http.NewRequest(method, u, rd)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := p.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var out map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, fmt.Errorf("incus %s %s: %s", method, path, resp.Status)
	}
	if resp.StatusCode >= 400 {
		return nil, fmt.Errorf("incus %s %s: %v", method, path, out["error"])
	}
	return out, nil
}

// PowerState returns "On" or "Off" from the instance status.
func (p *Power) PowerState() (string, error) {
	if !p.enabled() {
		return "On", nil
	}
	out, err := p.do("GET", "/1.0/instances/"+p.tray+"/state", nil)
	if err != nil {
		return "", err
	}
	md, _ := out["metadata"].(map[string]any)
	if md["status"] == "Running" {
		return "On", nil
	}
	return "Off", nil
}

// Action changes the tray's power state and waits for completion.
// action is one of Incus' start, stop, restart.
func (p *Power) Action(action string, force bool, timeout int) error {
	if !p.enabled() {
		return fmt.Errorf("power control is not configured")
	}
	out, err := p.do("PUT", "/1.0/instances/"+p.tray+"/state",
		map[string]any{"action": action, "force": force, "timeout": timeout})
	if err != nil {
		return err
	}
	op, _ := out["operation"].(string)
	if op == "" {
		return nil
	}
	res, err := p.do("GET", op+"/wait?timeout=120", nil)
	if err != nil {
		return err
	}
	md, _ := res["metadata"].(map[string]any)
	if msg, _ := md["err"].(string); msg != "" {
		return fmt.Errorf("incus %s: %s", action, msg)
	}
	return nil
}
