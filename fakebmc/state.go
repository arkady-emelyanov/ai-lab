package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
)

// LinkKey identifies one NVLink port: GPU index and link index.
type LinkKey struct{ GPU, Link int }

func (k LinkKey) String() string { return fmt.Sprintf("%d:%d", k.GPU, k.Link) }

// State is what the BMC remembers across restarts: which NVLink ports are
// disabled now, which changes wait for the next tray reset, and which
// disables are sticky.
type State struct {
	mu       sync.Mutex
	path     string
	sideband string

	Disabled map[string]bool `json:"disabled"` // applied, visible on the tray
	Pending  map[string]bool `json:"pending"`  // requested LinkState: true = disabled
	Sticky   map[string]bool `json:"sticky"`
}

func loadState(dir, sideband string) (*State, error) {
	s := &State{
		path:     filepath.Join(dir, "state.json"),
		sideband: filepath.Join(sideband, "nvlink-disabled"),
		Disabled: map[string]bool{},
		Pending:  map[string]bool{},
		Sticky:   map[string]bool{},
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	data, err := os.ReadFile(s.path)
	if err == nil {
		if err := json.Unmarshal(data, s); err != nil {
			return nil, fmt.Errorf("%s: %w", s.path, err)
		}
	} else if !os.IsNotExist(err) {
		return nil, err
	}
	return s, s.writeSideband()
}

// LinkDisabled reports the applied state and the pending one, if any.
func (s *State) LinkDisabled(k LinkKey) (current bool, pending *bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	current = s.Disabled[k.String()]
	if p, ok := s.Pending[k.String()]; ok {
		pending = &p
	}
	return
}

func (s *State) IsSticky(k LinkKey) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.Sticky[k.String()]
}

func (s *State) SetPending(k LinkKey, disabled bool) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.Disabled[k.String()] == disabled {
		delete(s.Pending, k.String())
	} else {
		s.Pending[k.String()] = disabled
	}
	return s.save()
}

func (s *State) SetSticky(k LinkKey, sticky bool) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if sticky {
		s.Sticky[k.String()] = true
	} else {
		delete(s.Sticky, k.String())
	}
	return s.save()
}

// ApplyPending runs at tray reset: pending link changes take effect, and
// non-sticky disables that were not re-requested are cleared, as on real
// hardware where a plain disable lasts until the next reset.
func (s *State) ApplyPending() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	next := map[string]bool{}
	for k := range s.Disabled {
		if s.Sticky[k] {
			next[k] = true
		}
	}
	for k, disabled := range s.Pending {
		if disabled {
			next[k] = true
		} else {
			delete(next, k)
		}
	}
	s.Disabled = next
	s.Pending = map[string]bool{}
	if err := s.save(); err != nil {
		return err
	}
	return s.writeSideband()
}

func (s *State) save() error {
	data, err := json.MarshalIndent(s, "", "  ")
	if err != nil {
		return err
	}
	tmp := s.path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, s.path)
}

// writeSideband publishes the applied link state to the tray, one
// "<gpu> <link>" pair per line; the tray's fake NVML reports those links as
// inactive.
func (s *State) writeSideband() error {
	if err := os.MkdirAll(filepath.Dir(s.sideband), 0o755); err != nil {
		return err
	}
	var lines []string
	for k := range s.Disabled {
		var g, l int
		if _, err := fmt.Sscanf(k, "%d:%d", &g, &l); err == nil {
			lines = append(lines, fmt.Sprintf("%d %d", g, l))
		}
	}
	sort.Strings(lines)
	body := "# NVLink ports disabled by the BMC: <gpu> <link>\n" + strings.Join(lines, "\n")
	if len(lines) > 0 {
		body += "\n"
	}
	tmp := s.sideband + ".tmp"
	if err := os.WriteFile(tmp, []byte(body), 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, s.sideband)
}
