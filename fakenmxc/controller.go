package main

import (
	"bufio"
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strings"
	"sync"

	pb "fakenmxc/gen/nmxlabv1"
)

const (
	defaultPartition = 32766 // as in NMX-C
	maxPartitionID   = 32765
	apiMajor         = 1
	apiMinor         = 0
)

type gpu struct {
	uid  uint64
	uuid string
	tray int // index into cfg.Trays
	idx  int // GPU index within the tray
}

type partition struct {
	Name string   `json:"name"`
	GPUs []uint64 `json:"gpus"`
}

type controller struct {
	pb.UnimplementedNMXControllerServer
	cfg  *Config
	gpus []gpu // global order: tray by tray

	mu         sync.Mutex
	path       string
	Partitions map[uint32]*partition `json:"partitions"`
	gateways   map[string]bool
}

// GPU identities exactly as the fake NVML computes them (fakegpu/fakegpu.h).
func uuidBytes(host string, idx int) [16]byte {
	h := uint64(1469598103934665603) // FNV-1a
	for i := 0; i < len(host); i++ {
		h = (h ^ uint64(host[i])) * 1099511628211
	}
	var out [16]byte
	for i := 0; i < 16; i++ {
		h = (h ^ uint64(idx*16+i)) * 1099511628211
		out[i] = byte(h >> 56)
	}
	return out
}

func newController(cfg *Config) (*controller, error) {
	c := &controller{cfg: cfg, path: filepath.Join(cfg.StateDir, "partitions.json"),
		Partitions: map[uint32]*partition{}, gateways: map[string]bool{}}
	for t, tray := range cfg.Trays {
		for i := 0; i < cfg.GPUCount; i++ {
			u := uuidBytes(tray.Name, i)
			c.gpus = append(c.gpus, gpu{
				uid:  binary.BigEndian.Uint64(u[:8]),
				uuid: fmt.Sprintf("GPU-%x-%x-%x-%x-%x", u[0:4], u[4:6], u[6:8], u[8:10], u[10:16]),
				tray: t, idx: i,
			})
		}
	}
	if err := os.MkdirAll(cfg.StateDir, 0o755); err != nil {
		return nil, err
	}
	if data, err := os.ReadFile(c.path); err == nil {
		if err := json.Unmarshal(data, c); err != nil {
			return nil, fmt.Errorf("%s: %w", c.path, err)
		}
	} else {
		// Factory state: every GPU in the default partition.
		all := make([]uint64, len(c.gpus))
		for i, g := range c.gpus {
			all[i] = g.uid
		}
		c.Partitions[defaultPartition] = &partition{Name: "default", GPUs: all}
	}
	return c, nil
}

func (c *controller) save() error {
	data, err := json.MarshalIndent(c, "", "  ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(c.path+".tmp", data, 0o644); err != nil {
		return err
	}
	return os.Rename(c.path+".tmp", c.path)
}

// cliqueOf is what NVML reports: the partition id, the configured default
// clique for the default partition, 0 for GPUs in no partition.
func (c *controller) cliqueOf(uid uint64) (partitionID, clique uint32) {
	for id, p := range c.Partitions {
		for _, g := range p.GPUs {
			if g == uid {
				if id == defaultPartition {
					return id, c.cfg.DefaultClique
				}
				return id, id
			}
		}
	}
	return 0, 0
}

// publish writes every GPU's clique to its tray's sideband (fabric-clique).
func (c *controller) publish() error {
	lines := make([][]string, len(c.cfg.Trays))
	for _, g := range c.gpus {
		_, clique := c.cliqueOf(g.uid)
		lines[g.tray] = append(lines[g.tray], fmt.Sprintf("%d %d", g.idx, clique))
	}
	for t, tray := range c.cfg.Trays {
		body := "# NVLink partition (clique) per GPU, from the partition controller: <gpu> <clique>\n" +
			strings.Join(lines[t], "\n") + "\n"
		path := filepath.Join(tray.SidebandDir, "fabric-clique")
		if err := os.WriteFile(path+".tmp", []byte(body), 0o644); err != nil {
			return err
		}
		if err := os.Rename(path+".tmp", path); err != nil {
			return err
		}
	}
	return nil
}

// disabledLinks counts a GPU's NVLinks that are down, from the tray and
// switch BMCs' sideband files.
func (c *controller) disabledLinks(g gpu) map[int]bool {
	down := map[int]bool{}
	for _, name := range []string{"nvlink-disabled", "nvlink-disabled-switch"} {
		f, err := os.Open(filepath.Join(c.cfg.Trays[g.tray].SidebandDir, name))
		if err != nil {
			continue
		}
		sc := bufio.NewScanner(f)
		for sc.Scan() {
			var gi, l int
			if _, err := fmt.Sscanf(sc.Text(), "%d %d", &gi, &l); err == nil && gi == g.idx {
				down[l] = true
			}
		}
		f.Close()
	}
	return down
}

func (c *controller) location(g gpu) *pb.Location {
	return &pb.Location{ChassisId: 1, SlotId: c.cfg.Trays[g.tray].Slot, HostId: 1, GpuId: uint32(g.idx)}
}

func (c *controller) gpuInfo(g gpu) *pb.GpuInfo {
	pid, clique := c.cliqueOf(g.uid)
	active := c.cfg.NVLinks - len(c.disabledLinks(g))
	health := pb.GpuHealth_NMX_GPU_HEALTH_HEALTHY
	if active < c.cfg.NVLinks {
		health = pb.GpuHealth_NMX_GPU_HEALTH_DEGRADED_BANDWIDTH
	}
	return &pb.GpuInfo{GpuUid: g.uid, Uuid: g.uuid, Location: c.location(g), Hostname: c.cfg.Trays[g.tray].Name,
		PartitionId: pid, CliqueId: clique, ActiveNvlinks: uint32(active), Health: health}
}

func (c *controller) partitionInfo(id uint32) *pb.PartitionInfo {
	p := c.Partitions[id]
	health := pb.GpuHealth_NMX_GPU_HEALTH_HEALTHY
	for _, uid := range p.GPUs {
		if g, ok := c.byUID(uid); ok && c.gpuInfo(g).Health != pb.GpuHealth_NMX_GPU_HEALTH_HEALTHY {
			health = pb.GpuHealth_NMX_GPU_HEALTH_DEGRADED_BANDWIDTH
		}
	}
	return &pb.PartitionInfo{PartitionId: id, PartitionName: p.Name, GpuUids: append([]uint64(nil), p.GPUs...),
		IsDefault: id == defaultPartition, State: "ACTIVE", Health: health}
}

func (c *controller) byUID(uid uint64) (gpu, bool) {
	for _, g := range c.gpus {
		if g.uid == uid {
			return g, true
		}
	}
	return gpu{}, false
}

// resolve turns UIDs or locations (not both) into GPU UIDs.
func (c *controller) resolve(uids []uint64, locs []*pb.Location) ([]uint64, error) {
	if len(uids) > 0 && len(locs) > 0 {
		return nil, fmt.Errorf("give GPUs by UID or by location, not both")
	}
	if len(uids) == 0 && len(locs) == 0 {
		return nil, fmt.Errorf("no GPUs given")
	}
	var out []uint64
	for _, u := range uids {
		if _, ok := c.byUID(u); !ok {
			return nil, fmt.Errorf("unknown GPU UID %d", u)
		}
		out = append(out, u)
	}
	for _, l := range locs {
		found := false
		for _, g := range c.gpus {
			gl := c.location(g)
			if gl.SlotId == l.SlotId && gl.GpuId == l.GpuId {
				out = append(out, g.uid)
				found = true
			}
		}
		if !found {
			return nil, fmt.Errorf("no GPU at slot %d gpu %d", l.SlotId, l.GpuId)
		}
	}
	return out, nil
}

func (c *controller) checkGateway(id string) bool { return id != "" && c.gateways[id] }

// commit persists and publishes a change, returning the partition result.
func (c *controller) commit(id uint32, msg string) *pb.PartitionResult {
	if err := c.save(); err != nil {
		return &pb.PartitionResult{ReturnCode: pb.ReturnCode_NMX_ST_INVALID_ARGUMENT, Message: err.Error()}
	}
	if err := c.publish(); err != nil {
		return &pb.PartitionResult{ReturnCode: pb.ReturnCode_NMX_ST_INVALID_ARGUMENT, Message: err.Error()}
	}
	res := &pb.PartitionResult{ReturnCode: pb.ReturnCode_NMX_ST_SUCCESS, Message: msg}
	if _, ok := c.Partitions[id]; ok {
		res.Partition = c.partitionInfo(id)
	}
	return res
}

func fail(code pb.ReturnCode, format string, args ...any) *pb.PartitionResult {
	return &pb.PartitionResult{ReturnCode: code, Message: fmt.Sprintf(format, args...)}
}

// ---- RPCs -------------------------------------------------------------------------

func (c *controller) Hello(_ context.Context, r *pb.ClientHello) (*pb.ServerHello, error) {
	if r.GatewayId == "" {
		return &pb.ServerHello{ReturnCode: pb.ReturnCode_NMX_ST_INVALID_ARGUMENT}, nil
	}
	c.mu.Lock()
	c.gateways[r.GatewayId] = true
	c.mu.Unlock()
	return &pb.ServerHello{ReturnCode: pb.ReturnCode_NMX_ST_SUCCESS, MajorVersion: apiMajor, MinorVersion: apiMinor,
		DomainUuid: c.cfg.DomainUUID}, nil
}

func (c *controller) GetDomainProperties(_ context.Context, r *pb.DomainRequest) (*pb.DomainProperties, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.checkGateway(r.GatewayId) {
		return &pb.DomainProperties{ReturnCode: pb.ReturnCode_NMX_ST_NOT_HELLO}, nil
	}
	return &pb.DomainProperties{ReturnCode: pb.ReturnCode_NMX_ST_SUCCESS, DomainUuid: c.cfg.DomainUUID,
		DomainName: c.cfg.DomainName, ComputeNodeCount: uint32(len(c.cfg.Trays)), SwitchNodeCount: 1,
		GpuCount: uint32(len(c.gpus)), NvlinksPerGpu: uint32(c.cfg.NVLinks), DefaultPartitionId: defaultPartition,
		MaxPartitions: maxPartitionID}, nil
}

func (c *controller) GetDomainStateInfo(_ context.Context, r *pb.DomainRequest) (*pb.DomainStateInfo, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.checkGateway(r.GatewayId) {
		return &pb.DomainStateInfo{ReturnCode: pb.ReturnCode_NMX_ST_NOT_HELLO}, nil
	}
	return &pb.DomainStateInfo{ReturnCode: pb.ReturnCode_NMX_ST_SUCCESS,
		ControlPlaneState:       pb.ControlPlaneState_NMX_CONTROL_PLANE_STATE_CONFIGURED,
		ConfigStatusDescription: "fabric discovered, partitioning service available"}, nil
}

func (c *controller) GetComputeNodeInfoList(_ context.Context, r *pb.DomainRequest) (*pb.ComputeNodeInfoList, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.checkGateway(r.GatewayId) {
		return &pb.ComputeNodeInfoList{ReturnCode: pb.ReturnCode_NMX_ST_NOT_HELLO}, nil
	}
	out := &pb.ComputeNodeInfoList{ReturnCode: pb.ReturnCode_NMX_ST_SUCCESS}
	for _, t := range c.cfg.Trays {
		out.Nodes = append(out.Nodes, &pb.ComputeNodeInfo{Hostname: t.Name, GpuCount: uint32(c.cfg.GPUCount),
			Location: &pb.Location{ChassisId: 1, SlotId: t.Slot, HostId: 1}})
	}
	return out, nil
}

func (c *controller) GetSwitchNodeInfoList(_ context.Context, r *pb.DomainRequest) (*pb.SwitchNodeInfoList, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.checkGateway(r.GatewayId) {
		return &pb.SwitchNodeInfoList{ReturnCode: pb.ReturnCode_NMX_ST_NOT_HELLO}, nil
	}
	down := 0
	for _, g := range c.gpus {
		for range c.disabledLinks(g) {
			down++
		}
	}
	return &pb.SwitchNodeInfoList{ReturnCode: pb.ReturnCode_NMX_ST_SUCCESS, Nodes: []*pb.SwitchNodeInfo{{
		Hostname: c.cfg.SwitchHost, SwitchCount: uint32(c.cfg.Switches), PortsPerSwitch: uint32(c.cfg.SwitchPorts),
		DisabledPorts: uint32(down)}}}, nil
}

func (c *controller) GetGpuInfoList(_ context.Context, r *pb.GpuInfoListRequest) (*pb.GpuInfoList, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.checkGateway(r.GatewayId) {
		return &pb.GpuInfoList{ReturnCode: pb.ReturnCode_NMX_ST_NOT_HELLO}, nil
	}
	out := &pb.GpuInfoList{ReturnCode: pb.ReturnCode_NMX_ST_SUCCESS}
	for _, g := range c.gpus {
		info := c.gpuInfo(g)
		if r.PartitionId != nil && info.PartitionId != *r.PartitionId {
			continue
		}
		if len(r.SlotIds) > 0 && !slices.Contains(r.SlotIds, info.Location.SlotId) {
			continue
		}
		out.Gpus = append(out.Gpus, info)
	}
	return out, nil
}

func (c *controller) GetTopologyInfo(_ context.Context, r *pb.DomainRequest) (*pb.TopologyInfo, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.checkGateway(r.GatewayId) {
		return &pb.TopologyInfo{ReturnCode: pb.ReturnCode_NMX_ST_NOT_HELLO}, nil
	}
	out := &pb.TopologyInfo{ReturnCode: pb.ReturnCode_NMX_ST_SUCCESS}
	per := c.cfg.NVLinks / c.cfg.Switches
	for gi, g := range c.gpus {
		down := c.disabledLinks(g)
		for l := 0; l < c.cfg.NVLinks; l++ {
			// Same cabling as the switch BMC: link l -> switch l%S, port g*(L/S) + l/S.
			out.Connections = append(out.Connections, &pb.NvlinkConnection{GpuUid: g.uid, GpuLink: uint32(l),
				SwitchId: uint32(l % c.cfg.Switches), SwitchPort: uint32(gi*per + l/c.cfg.Switches), Active: !down[l]})
		}
	}
	return out, nil
}

func (c *controller) sortedIDs() []uint32 {
	ids := make([]uint32, 0, len(c.Partitions))
	for id := range c.Partitions {
		ids = append(ids, id)
	}
	sort.Slice(ids, func(i, j int) bool { return ids[i] < ids[j] })
	return ids
}

func (c *controller) GetPartitionCount(_ context.Context, r *pb.DomainRequest) (*pb.PartitionCount, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.checkGateway(r.GatewayId) {
		return &pb.PartitionCount{ReturnCode: pb.ReturnCode_NMX_ST_NOT_HELLO}, nil
	}
	return &pb.PartitionCount{ReturnCode: pb.ReturnCode_NMX_ST_SUCCESS, Count: uint32(len(c.Partitions))}, nil
}

func (c *controller) GetPartitionIdList(_ context.Context, r *pb.DomainRequest) (*pb.PartitionIdList, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.checkGateway(r.GatewayId) {
		return &pb.PartitionIdList{ReturnCode: pb.ReturnCode_NMX_ST_NOT_HELLO}, nil
	}
	return &pb.PartitionIdList{ReturnCode: pb.ReturnCode_NMX_ST_SUCCESS, PartitionIds: c.sortedIDs()}, nil
}

func (c *controller) GetPartitionInfoList(_ context.Context, r *pb.PartitionInfoListRequest) (*pb.PartitionInfoList, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.checkGateway(r.GatewayId) {
		return &pb.PartitionInfoList{ReturnCode: pb.ReturnCode_NMX_ST_NOT_HELLO}, nil
	}
	out := &pb.PartitionInfoList{ReturnCode: pb.ReturnCode_NMX_ST_SUCCESS}
	ids := r.PartitionIds
	if len(ids) == 0 {
		ids = c.sortedIDs()
	}
	for _, id := range ids {
		if _, ok := c.Partitions[id]; !ok {
			return &pb.PartitionInfoList{ReturnCode: pb.ReturnCode_NMX_ST_NOT_FOUND}, nil
		}
		out.Partitions = append(out.Partitions, c.partitionInfo(id))
	}
	return out, nil
}

// owner returns the partition a GPU belongs to, or 0.
func (c *controller) owner(uid uint64) uint32 {
	id, _ := c.cliqueOf(uid)
	return id
}

func (c *controller) CreatePartition(_ context.Context, r *pb.CreatePartitionRequest) (*pb.PartitionResult, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.checkGateway(r.GatewayId) {
		return fail(pb.ReturnCode_NMX_ST_NOT_HELLO, "call Hello first"), nil
	}
	uids, err := c.resolve(r.GpuUids, r.Locations)
	if err != nil {
		return fail(pb.ReturnCode_NMX_ST_INVALID_ARGUMENT, "%v", err), nil
	}
	for _, p := range c.Partitions {
		if r.PartitionName != "" && p.Name == r.PartitionName {
			return fail(pb.ReturnCode_NMX_ST_ALREADY_EXISTS, "partition name %q is in use", r.PartitionName), nil
		}
	}
	for _, u := range uids {
		if id := c.owner(u); id != 0 {
			return fail(pb.ReturnCode_NMX_ST_GPU_IN_USE, "GPU %d is in partition %d; remove it there first", u, id), nil
		}
	}
	id := r.PartitionId
	if id == 0 {
		for id = 1; id <= maxPartitionID; id++ {
			if _, used := c.Partitions[id]; !used {
				break
			}
		}
	}
	if id > maxPartitionID {
		return fail(pb.ReturnCode_NMX_ST_INVALID_ARGUMENT, "partition ids are 1..%d", maxPartitionID), nil
	}
	if _, used := c.Partitions[id]; used {
		return fail(pb.ReturnCode_NMX_ST_ALREADY_EXISTS, "partition %d exists", id), nil
	}
	name := r.PartitionName
	if name == "" {
		name = fmt.Sprintf("partition-%d", id)
	}
	c.Partitions[id] = &partition{Name: name, GPUs: uids}
	return c.commit(id, "partition created"), nil
}

func (c *controller) DeletePartition(_ context.Context, r *pb.DeletePartitionRequest) (*pb.PartitionResult, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.checkGateway(r.GatewayId) {
		return fail(pb.ReturnCode_NMX_ST_NOT_HELLO, "call Hello first"), nil
	}
	if _, ok := c.Partitions[r.PartitionId]; !ok {
		return fail(pb.ReturnCode_NMX_ST_NOT_FOUND, "no partition %d", r.PartitionId), nil
	}
	delete(c.Partitions, r.PartitionId)
	return c.commit(r.PartitionId, "partition deleted; its GPUs are in no partition"), nil
}

func (c *controller) AddGpusToPartition(_ context.Context, r *pb.GpuMembershipRequest) (*pb.PartitionResult, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.checkGateway(r.GatewayId) {
		return fail(pb.ReturnCode_NMX_ST_NOT_HELLO, "call Hello first"), nil
	}
	p, ok := c.Partitions[r.PartitionId]
	if !ok {
		return fail(pb.ReturnCode_NMX_ST_NOT_FOUND, "no partition %d", r.PartitionId), nil
	}
	uids, err := c.resolve(r.GpuUids, r.Locations)
	if err != nil {
		return fail(pb.ReturnCode_NMX_ST_INVALID_ARGUMENT, "%v", err), nil
	}
	for _, u := range uids {
		if id := c.owner(u); id != 0 && id != r.PartitionId {
			return fail(pb.ReturnCode_NMX_ST_GPU_IN_USE, "GPU %d is in partition %d; remove it there first", u, id), nil
		}
	}
	for _, u := range uids {
		if c.owner(u) == 0 {
			p.GPUs = append(p.GPUs, u)
		}
	}
	return c.commit(r.PartitionId, "GPUs added"), nil
}

func (c *controller) RemoveGpusFromPartition(_ context.Context, r *pb.GpuMembershipRequest) (*pb.PartitionResult, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.checkGateway(r.GatewayId) {
		return fail(pb.ReturnCode_NMX_ST_NOT_HELLO, "call Hello first"), nil
	}
	p, ok := c.Partitions[r.PartitionId]
	if !ok {
		return fail(pb.ReturnCode_NMX_ST_NOT_FOUND, "no partition %d", r.PartitionId), nil
	}
	uids, err := c.resolve(r.GpuUids, r.Locations)
	if err != nil {
		return fail(pb.ReturnCode_NMX_ST_INVALID_ARGUMENT, "%v", err), nil
	}
	remove := map[uint64]bool{}
	for _, u := range uids {
		remove[u] = true
	}
	kept := p.GPUs[:0]
	for _, u := range p.GPUs {
		if !remove[u] {
			kept = append(kept, u)
		}
	}
	p.GPUs = kept
	return c.commit(r.PartitionId, "GPUs removed"), nil
}
