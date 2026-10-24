"""MIG through NVML, as a GB200 shows it and as its consumers use it: go-nvlib
(device plugin, GPU Feature Discovery, mig-parted, NVIDIA's DRA driver),
mig-parted's apply flow, the DRA driver's dynamic MIG, Slurm's NVML plugin
and dcgm-exporter. Values come from real GB200 output (mig-parted#227,
dra-driver-nvidia-gpu#907 and #1454) and nvml.h; GPU instance IDs follow the
hardware's slot numbering (a 1g at slot 6 is GPU instance 13, as on GB200).

Each test runs NVML in fresh processes on its own emulated tray (mig_tray),
as separate programs on a tray do; MIG state is the tray's, shared by all."""
import pytest

from mig_tray import (BY_NAME, CI_PROFILE_COUNT, CI_PROFILES, CI_SLICES, FULL_CI, PLACEMENTS, PROFILE_COUNT,
                      PROFILES, UNSUPPORTED, Tray, go_nvlib_name)

SUCCESS, INVALID_ARGUMENT, NOT_SUPPORTED, NO_PERMISSION, NOT_FOUND = 0, 2, 3, 4, 6
IN_USE, INSUFFICIENT_RESOURCES = 19, 23


@pytest.fixture
def tray(tmp_path):
    return Tray(tmp_path)


# ---- MIG mode ------------------------------------------------------------------


def test_capable_and_off_by_default(tray):
    """go-nvlib IsMigCapable (GetMigMode succeeds), IsMigEnabled (current 0);
    VisitMigDevices finds none: max count 0, no handle at index 0."""
    out = tray.nvml("d = m.dev(0); out = [m.mig_mode(d), m.max_mig(d), m.mig_handle(d, 0)[0], m.count()]")
    assert out == [[SUCCESS, 0, 0], [SUCCESS, 0], NOT_FOUND, 1]


def test_enable_takes_effect_at_once_on_an_idle_gpu(tray):
    """Hopper and later: no reset needed (nvidia-smi -mig 1: 'Enabled MIG
    Mode', no pending warning). activationStatus is written: the DRA driver
    checks it (nvlib.go:1005)."""
    out = tray.nvml("d = m.dev(0); out = [m.set_mig(d, 1), m.mig_mode(d), m.max_mig(d)]")
    assert out == [[SUCCESS, SUCCESS], [SUCCESS, 1, 1], [SUCCESS, 7]]


def test_mode_is_the_trays_and_the_gpus(tmp_path):
    """MIG mode is driver state: another process on the tray sees it, another
    GPU and another tray do not."""
    tray = Tray(tmp_path, count=2)
    tray.enable_mig(0)
    assert tray.nvml("out = [m.mig_mode(m.dev(0)), m.mig_mode(m.dev(1))]") == [[0, 1, 1], [0, 0, 0]]
    assert Tray(tmp_path, name="tray1").nvml("out = m.mig_mode(m.dev(0))") == [0, 0, 0]


def test_enable_while_in_use_is_pending_until_a_reset(tray):
    """A GPU a process holds cannot switch: SetMigMode succeeds with
    activationStatus IN_USE and the mode pending (mig-parted's
    IsMigModeChangePending: cur != pend); it applies at the GPU reset
    (mig-parted: nvidia-smi -r -i <bus ids>)."""
    import subprocess
    import sys
    holder = subprocess.Popen(
        [sys.executable, "-c", "import ctypes, sys, time\n"
         f"cu = ctypes.CDLL('libcuda.so.1'); cu.cuInit(0); c = ctypes.c_void_p()\n"
         "cu.cuDevicePrimaryCtxRetain(ctypes.byref(c), 0); cu.cuCtxSetCurrent(c); p = ctypes.c_ulonglong()\n"
         "cu.cuMemAlloc_v2(ctypes.byref(p), ctypes.c_size_t(1 << 20)); print('ready', flush=True); sys.stdin.read()"],
        env=tray.env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "ready"
        out = tray.nvml("d = m.dev(0); out = [m.set_mig(d, 1), m.mig_mode(d)]")
        assert out == [[SUCCESS, IN_USE], [SUCCESS, 0, 1]]
    finally:
        holder.stdin.close()
        holder.wait(timeout=30)
    tray.smi("-r", "-i", "00000008:01:00.0", root=True)
    assert tray.nvml("out = m.mig_mode(m.dev(0))") == [SUCCESS, 1, 1]


def test_changes_need_root(tray):
    """As with the real driver, MIG changes need root: NO_PERMISSION for a
    user (mode, GPU and compute instances, destroying them); reading works."""
    user = "d = m.dev(0); out = [m.set_mig(d, 1)[0], m.mig_mode(d)]"
    assert tray.nvml(user, root=False) == [NO_PERMISSION, [SUCCESS, 0, 0]]
    tray.enable_mig()
    tray.nvml("d = m.dev(0); r, gi = m.create_gi(d, 19); r, ci = m.create_ci(gi, 0); out = r")
    out = tray.nvml("""
d = m.dev(0); r, gi = m.gi_by_id(d, 13); r, ci = m.ci_by_id(gi, 0)
out = [m.create_gi(d, 19)[0], m.create_ci(gi, 0)[0], m.destroy_ci(ci), m.destroy_gi(gi), m.set_mig(d, 0)[0],
       len(m.mig_devices(d))]
""", root=False)
    assert out == [NO_PERMISSION] * 5 + [1]


def test_disable_is_refused_while_gpu_instances_exist(tray):
    """nvidia-smi -mig 0 with instances: 'In use by another client'."""
    tray.enable_mig()
    out = tray.nvml("d = m.dev(0); r, gi = m.create_gi(d, 19); out = [r, m.set_mig(d, 0)[0], m.mig_mode(d)]")
    assert out == [SUCCESS, IN_USE, [SUCCESS, 1, 1]]
    out = tray.nvml("d = m.dev(0); r, gi = m.gi_by_id(d, 13); out = [m.destroy_gi(gi), m.set_mig(d, 0), "
                    "m.mig_mode(d), m.max_mig(d)]")
    assert out == [SUCCESS, [SUCCESS, SUCCESS], [SUCCESS, 0, 0], [SUCCESS, 0]]


# ---- GPU instance profiles and placements ---------------------------------------


def test_profile_table(tray):
    """nvmlDeviceGetGpuInstanceProfileInfo (v1, go-nvml) for every profile
    index: GB200's profiles, with MIG off too (go-nvlib VisitMigProfiles, the
    device plugin's mixed strategy and DRA run it before enabling MIG)."""
    out = tray.nvml(f"d = m.dev(0); out = {{i: m.profile(d, i) for i in range({PROFILE_COUNT})}}")
    for index, (name, pid, slices, inst, sms, mib, ce, dec, enc, jpeg, ofa) in PROFILES.items():
        ret, info = out[str(index)]
        assert ret == SUCCESS, name
        assert info == {"id": pid, "isP2pSupported": 0, "sliceCount": slices, "instanceCount": inst,
                        "multiprocessorCount": sms, "copyEngineCount": ce, "decoderCount": dec, "encoderCount": enc,
                        "jpegCount": jpeg, "ofaCount": ofa, "memorySizeMB": mib}, name


def test_unsupported_and_invalid_profiles(tray):
    """Profiles GB200 lacks are NOT_SUPPORTED (index 17, 3_SLICE_GFX, must not
    succeed: go-nvlib's NewMigProfile rejects it and the DRA driver exits);
    beyond the enum, INVALID_ARGUMENT (enumerate until it)."""
    out = tray.nvml(f"d = m.dev(0); out = [m.profile(d, i)[0] for i in {UNSUPPORTED + [18, 99]}]")
    assert out == [NOT_SUPPORTED] * len(UNSUPPORTED) + [INVALID_ARGUMENT] * 2


def test_profile_names_v2_v3_and_by_id(tray):
    """Versioned infos carry the name ('MIG 1g.23gb'); v3 drops
    isP2pSupported for capabilities (no P2P on GB200); dcgm-exporter looks
    profiles up by id (GetGpuInstanceProfileInfoByIdV)."""
    out = tray.nvml("d = m.dev(0); out = [[m.profile_v(d, i, 2), m.profile_v(d, i, 3), m.profile_by_id(d, m.profile(d, i)[1]['id'])] "
                    f"for i in {list(PROFILES)}]")
    for (v2, v3, by_id), (name, *_rest) in zip(out, PROFILES.values()):
        assert v2[0] == v3[0] == by_id[0] == SUCCESS
        assert v2[1]["name"] == v3[1]["name"] == by_id[1]["name"] == f"MIG {name}"
        assert v3[1]["capabilities"] == 0 and v3[1]["memorySizeMB"] == v2[1]["memorySizeMB"]


def test_go_nvlib_names_are_mig_parteds_gb200_profiles(tray):
    """go-nvlib names a profile from slices and memory relative to the GPU's
    (GetMemoryInfo): the names mig-parted's GB200 configurations use."""
    out = tray.nvml(f"d = m.dev(0); out = [m.memory(d)[1], {{i: m.profile(d, i)[1] for i in {list(PROFILES)}}}]")
    total, infos = out
    assert total == 189471 << 20
    names = {i: go_nvlib_name(info["sliceCount"], info["sliceCount"], info["memorySizeMB"],
                              "+me" if int(i) == 7 else "") for i, info in infos.items()}
    assert sorted(names.values()) == sorted(v[0] for v in PROFILES.values())


def test_placements(tray):
    """GetGpuInstancePossiblePlacements_v2, go-nvml's two calls (count, then
    list), with MIG off too (the DRA driver's discovery)."""
    out = tray.nvml(f"d = m.dev(0); out = {{p: m.placements(d, p) for p in {list(PLACEMENTS)}}}")
    for pid, expected in PLACEMENTS.items():
        assert out[str(pid)] == [SUCCESS, [list(p) for p in expected]], pid


def test_placements_of_an_unknown_profile_id(tray):
    assert tray.nvml("out = m.placements(m.dev(0), 7)[0]") in (INVALID_ARGUMENT, NOT_SUPPORTED)


# ---- GPU instances ------------------------------------------------------------------


def test_no_gpu_instances_while_mig_is_off(tray):
    """With MIG off, instance calls are NOT_SUPPORTED (the DRA driver's
    deleteMigDevice treats it as nothing to delete)."""
    out = tray.nvml("d = m.dev(0); out = [m.create_gi(d, 19)[0], m.gis(d, 19)[0], m.gi_by_id(d, 13)[0]]")
    assert out == [NOT_SUPPORTED] * 3


@pytest.mark.parametrize("name", list(BY_NAME))
def test_fill_a_gpu_with_one_profile(tray, name):
    """CreateGpuInstance (no placement, mig-parted) until the GPU is full:
    instanceCount instances, then INSUFFICIENT_RESOURCES; the remaining
    capacity counts down; GetGpuInstances lists them (go-nvml sizes its
    buffer from instanceCount)."""
    index, pid, _slices, inst, *_ = BY_NAME[name]
    tray.enable_mig()
    out = tray.nvml(f"""
d = m.dev(0); info = m.profile(d, {index})[1]
caps, rets = [m.remaining(d, {pid})], []
while True:
    r, gi = m.create_gi(d, {pid}); rets.append(r); caps.append(m.remaining(d, {pid}))
    if r: break
out = [rets, caps, m.gis(d, {pid}, info['instanceCount'])[0], len(m.gis(d, {pid}, info['instanceCount'])[1])]
""")
    rets, caps, ret_list, listed = out
    assert rets == [SUCCESS] * inst + [INSUFFICIENT_RESOURCES]
    assert [c[1] for c in caps] == list(range(inst, -1, -1)) + [0]
    assert (ret_list, listed) == (SUCCESS, inst)


def test_gpu_instance_ids_and_info(tray):
    """GPU instance IDs are slots: 7g is 0; a 1g at start s is 7 + s (the
    first 1g nvidia-smi creates on GB200 is GPU instance 13, start 6).
    GetInfo reports the profile id, the placement and the parent device."""
    tray.enable_mig()
    out = tray.nvml("""
d = m.dev(0)
r, gi = m.create_gi(d, 19)
first = m.gi_info(gi)[1]
placed = [m.gi_info(m.create_gi(d, 19, (s, 1))[1])[1] for s in (0, 3)]
out = [first, placed, m.gi_info(m.gi_by_id(d, 7)[1])[1]['placement'], m.gi_by_id(d, 1)[0], first['device'] == d.value]
""")
    first, placed, by_id, missing, parent = out
    assert first["id"] == 13 and first["profileId"] == 19 and first["placement"] == [6, 1]
    assert [(p["id"], p["placement"]) for p in placed] == [(7, [0, 1]), (10, [3, 1])]
    assert by_id == [0, 1] and missing == NOT_FOUND and parent


def test_whole_gpu_instance_is_0(tray):
    tray.enable_mig()
    out = tray.nvml("d = m.dev(0); r, gi = m.create_gi(d, 0); out = [r, m.gi_info(gi)[1]]")
    assert out[0] == SUCCESS and out[1]["id"] == 0 and out[1]["placement"] == [0, 8]


def test_placements_do_not_overlap(tray):
    """An occupied or overlapping placement is INSUFFICIENT_RESOURCES (the DRA
    driver's createMigDevice, nvlib.go:1028); a free one still works."""
    tray.enable_mig()
    out = tray.nvml("""
d = m.dev(0)
out = [m.create_gi(d, 14, (2, 2))[0], m.create_gi(d, 14, (2, 2))[0], m.create_gi(d, 19, (3, 1))[0],
       m.create_gi(d, 0, (0, 8))[0], m.create_gi(d, 9, (0, 4))[0], m.create_gi(d, 9, (4, 4))[0]]
""")
    assert out == [SUCCESS, INSUFFICIENT_RESOURCES, INSUFFICIENT_RESOURCES, INSUFFICIENT_RESOURCES,
                   INSUFFICIENT_RESOURCES, SUCCESS]


def test_mixed_layout_all_balanced(tray):
    """mig-parted's GB200 all-balanced: 1g.23gb x2, 2g.47gb x1, 3g.93gb x1
    fit one GPU together (all 8 memory slices) when created in that order
    without placements: an instance takes the highest free placement."""
    tray.enable_mig()
    out = tray.nvml("d = m.dev(0); out = [m.create_gi(d, p)[0] for p in (9, 14, 19, 19, 19)] + "
                    "[sorted(tuple(m.gi_info(g)[1]['placement']) for p in (9, 14, 19) for g in m.gis(d, p)[1])]")
    assert out[:5] == [SUCCESS] * 4 + [INSUFFICIENT_RESOURCES]
    assert out[5] == [[0, 1], [1, 1], [2, 2], [4, 4]]


def test_instances_outlive_their_creator(tray):
    """Instances are driver state: a later process finds them by id and in
    GetGpuInstances (mig-parted's GetMigConfig, the DRA driver at startup)."""
    tray.enable_mig()
    tray.nvml("d = m.dev(0); r, gi = m.create_gi(d, 9); out = r")
    out = tray.nvml("d = m.dev(0); r, gis = m.gis(d, 9, 2); out = [r, [m.gi_info(g)[1]['id'] for g in gis]]")
    assert out[0] == SUCCESS and len(out[1]) == 1


# ---- compute instances ------------------------------------------------------------------


@pytest.mark.parametrize("name", list(CI_PROFILES))
def test_compute_instance_profiles(tray, name):
    """GpuInstanceGetComputeInstanceProfileInfo(V) for every CI index: the
    ones the GPU instance offers (id = index), the others NOT_SUPPORTED or
    INVALID_ARGUMENT (mig-parted and go-nvlib walk 0..8); the full one has
    the GPU instance's SMs, the others fewer; names as nvidia-smi -lcip."""
    index, pid, slices, _inst, sms, *_ = BY_NAME[name]
    tray.enable_mig()
    out = tray.nvml(f"""
d = m.dev(0); r, gi = m.create_gi(d, {pid})
out = {{j: [m.ci_profile(gi, j), m.ci_profile_v(gi, j)] for j in range({CI_PROFILE_COUNT + 1})}}
""")
    offered = CI_PROFILES[name]
    for j in range(CI_PROFILE_COUNT + 1):
        (ret, info), (ret_v, info_v) = out[str(j)]
        if j not in offered:
            assert ret in (NOT_SUPPORTED, INVALID_ARGUMENT) and ret_v == ret, (name, j)
            continue
        assert ret == ret_v == SUCCESS, (name, j)
        assert info["id"] == j and info["instanceCount"] == offered[j] and info["sliceCount"] == CI_SLICES[j]
        assert info_v["id"] == j and info_v["multiprocessorCount"] == info["multiprocessorCount"]
        g = slices
        ci_name = name if CI_SLICES[j] == g or j in (0, 7) and g == 1 else f"{CI_SLICES[j]}c.{name}"
        assert info_v["name"] == f"MIG {ci_name}", (name, j)
        if j == FULL_CI[name]:
            assert info["multiprocessorCount"] == sms
        else:
            assert 0 < info["multiprocessorCount"] < sms
    if name == "1g.47gb":
        assert out["0"][0][1]["multiprocessorCount"] == 18


def test_compute_instances_fill_their_gpu_instance(tray):
    """mig-parted's reuseGI: '1c.7g.186gb: 7' is one 7g with seven 1-slice
    compute instances; the eighth is INSUFFICIENT_RESOURCES; GetInfo gives
    each id, profile id and placement; GetComputeInstances lists them."""
    tray.enable_mig()
    out = tray.nvml("""
d = m.dev(0); r, gi = m.create_gi(d, 0); ciid = m.ci_profile(gi, 0)[1]['id']
rets, infos = [], []
for _ in range(8):
    r, ci = m.create_ci(gi, ciid); rets.append(r)
    if r == 0: infos.append(m.ci_info(ci)[1])
out = [rets, [(i['id'], i['profileId'], i['placement']) for i in infos], len(m.cis(gi, ciid)[1]), m.ci_by_id(gi, 9)[0]]
""")
    rets, infos, listed, missing = out
    assert rets == [SUCCESS] * 7 + [INSUFFICIENT_RESOURCES]
    assert [i[0] for i in infos] == list(range(7)) and {i[1] for i in infos} == {0}
    assert sorted(tuple(i[2]) for i in infos) == [(s, 1) for s in range(7)]
    assert listed == 7 and missing == NOT_FOUND


def test_compute_instance_sms_are_a_gpu_instances_of_their_slices(tray):
    """A c-slice compute instance has the SMs of a c-slice GPU instance (A100:
    1c.3g.20gb 14 like 1g.5gb; GB200: 1c.1g.47gb 18 like 1g.23gb), never
    more than a smaller GPU instance."""
    tray.enable_mig()
    out = tray.nvml("d = m.dev(0); r, gi = m.create_gi(d, 0); "
                    "out = {m.ci_profile(gi, j)[1]['sliceCount']: m.ci_profile(gi, j)[1]['multiprocessorCount'] "
                    "for j in range(5)}")
    gi_sms = {v[2]: v[4] for v in PROFILES.values() if v[0] != "1g.47gb"}
    assert out == {str(c): gi_sms[c] for c in (1, 2, 3, 4, 7)} == {"1": 18, "2": 38, "3": 72, "4": 76, "7": 152}


def test_compute_instance_placements(tray):
    """GetComputeInstancePossiblePlacements: starts aligned to the size
    rounded up to a power of two (a 3-slice CI of a 7g at 0 and 4, as in
    go-nvml's B200/H100 tables); CreateComputeInstanceWithPlacement takes
    one: elsewhere INVALID_ARGUMENT, taken or overlapping
    INSUFFICIENT_RESOURCES."""
    tray.enable_mig()
    out = tray.nvml("""
d = m.dev(0); r, gi = m.create_gi(d, 0)
places = {j: m.ci_placements(gi, j) for j in range(5)}
r3, ci3 = m.create_ci(gi, 2, (4, 3))
out = [places, r3, m.ci_info(ci3)[1]['placement'], m.create_ci(gi, 2, (3, 3))[0], m.create_ci(gi, 2, (4, 2))[0],
       m.create_ci(gi, 2, (4, 3))[0], m.create_ci(gi, 0, (5, 1))[0], m.create_ci(gi, 0, (3, 1))[0],
       m.create_ci(gi, 2)[0], m.create_ci(gi, 2)[0]]
""")
    places, r3, placed, misaligned, wrong_size, taken, overlapping, free, lowest, full = out
    assert places == {"0": [SUCCESS, [[s, 1] for s in range(7)]], "1": [SUCCESS, [[0, 2], [2, 2], [4, 2]]],
                      "2": [SUCCESS, [[0, 3], [4, 3]]], "3": [SUCCESS, [[0, 4]]], "4": [SUCCESS, [[0, 7]]]}
    assert (r3, placed) == (SUCCESS, [4, 3])
    assert (misaligned, wrong_size) == (INVALID_ARGUMENT, INVALID_ARGUMENT)
    assert (taken, overlapping, free) == (INSUFFICIENT_RESOURCES, INSUFFICIENT_RESOURCES, SUCCESS)
    assert (lowest, full) == (SUCCESS, INSUFFICIENT_RESOURCES)  # the other 3-slice placement, at 0


def test_instances_in_use_cannot_be_destroyed(tray):
    """A compute instance a process uses is IN_USE (as nvidia-smi mig -dci
    with a CUDA process on it), its GPU instance too; the GPU's processes
    list it with its GI and CI ids. Once the process ends both go."""
    tray.enable_mig()
    tray.nvml("d = m.dev(0)\nfor p in (19, 19):\n    r, gi = m.create_gi(d, p); m.create_ci(gi, 0)\nout = 0")
    uuids = tray.nvml("d = m.dev(0); out = [m.uuid(h)[1] for h in m.mig_devices(d)]")
    destroy = "d = m.dev(0); r, gi = m.gi_by_id(d, {g}); r, ci = m.ci_by_id(gi, 0); out = [m.destroy_ci(ci), m.destroy_gi(gi)]"
    with tray.holding(env={"CUDA_VISIBLE_DEVICES": uuids[1]}):  # GI 13 (slot 6); uuids[0] is GI 12
        assert tray.nvml(destroy.format(g=13)) == [IN_USE, IN_USE]
        procs = tray.py("import ctypes, sys\nsys.path.insert(0, '.')\nimport mig_nvml as m\n"
                        "class P(ctypes.Structure): _fields_ = [('pid', ctypes.c_uint), ('mem', ctypes.c_ulonglong), "
                        "('gi', ctypes.c_uint), ('ci', ctypes.c_uint)]\n"
                        "n = ctypes.c_uint(8); b = (P * 8)()\n"
                        "assert m.nvml.nvmlDeviceGetComputeRunningProcesses_v3(m.dev(0), ctypes.byref(n), b) == 0\n"
                        "print([(p.gi, p.ci) for p in b[:n.value]])").stdout.strip()
        assert procs == "[(13, 0)]"
        assert tray.nvml(destroy.format(g=12)) == [SUCCESS, SUCCESS]  # the other slice is free
    assert tray.nvml(destroy.format(g=13)) == [SUCCESS, SUCCESS]


def test_destroy_order(tray):
    """A GPU instance with compute instances is IN_USE (mig-parted's
    ClearMigConfig and the DRA driver destroy CIs first); then both go."""
    tray.enable_mig()
    out = tray.nvml("""
d = m.dev(0); r, gi = m.create_gi(d, 19); gid = m.gi_info(gi)[1]['id']
r, ci = m.create_ci(gi, 0)
out = [m.destroy_gi(gi), m.destroy_ci(ci), m.ci_by_id(gi, 0)[0], m.destroy_gi(gi), m.gi_by_id(d, gid)[0]]
""")
    assert out == [IN_USE, SUCCESS, NOT_FOUND, SUCCESS, NOT_FOUND]


# ---- MIG devices -------------------------------------------------------------------------


def make_layout(tray):
    """A 1g at slot 0 and a 3g at slot 4, a full compute instance in each."""
    tray.enable_mig()
    tray.nvml("""
d = m.dev(0)
for pid, pl, ci in ((19, (0, 1), 0), (9, (4, 4), 2)):
    r, gi = m.create_gi(d, pid, pl); assert r == 0
    r, c = m.create_ci(gi, m.ci_profile(gi, ci)[1]['id']); assert r == 0
out = 0
""")


def test_mig_devices_are_dense_and_ordered(tray):
    """GetMigDeviceHandleByIndex: dense from 0 (Slurm stops at the first
    failure), ordered by GPU instance id then compute instance id, NOT_FOUND
    beyond; each IsMigDeviceHandle, its parent the GPU; GetIndex its index."""
    make_layout(tray)
    out = tray.nvml("""
d = m.dev(0)
hs = [m.mig_handle(d, i) for i in range(7)]
out = [[r for r, _ in hs], [[m.is_mig(h)[1], m.parent(h)[1].value == d.value, m.gi_id(h)[1], m.ci_id(h)[1], m.index(h)[1]]
       for r, h in hs if r == 0], m.is_mig(d)[1], m.count()]
""")
    rets, devices, parent_is_mig, count = out
    assert rets == [SUCCESS, SUCCESS] + [NOT_FOUND] * 5
    assert devices == [[1, True, 2, 0, 0], [1, True, 7, 0, 1]]  # 3g@4 is slot 2, 1g@0 slot 7
    assert parent_is_mig == 0 and count == 1


def test_mig_device_identity(tray):
    """A MIG device's UUID (MIG-, a hash with the RFC 4122 version-5 bits
    real ones have) is found by GetHandleByUUID (device plugin health
    checks, dcgm-exporter); its name is
    the GPU's plus the profile ('NVIDIA GB200 MIG 1g.23gb': Slurm's gres
    type); memory and attributes are its GPU instance's (go-nvlib
    GetProfile: attributes.memorySizeMB equals the profile's), SMs its
    compute instance's; no minor number of its own."""
    make_layout(tray)
    out = tray.nvml("""
d = m.dev(0)
out = []
for i in range(2):
    h = m.mig_handle(d, i)[1]
    u = m.uuid(h)[1]
    out.append([u, m.by_uuid(u)[1].value == h.value, m.name(h)[1], m.memory(h)[1], m.attributes(h), m.minor(h)[0]])
""")
    import re
    (u3, found3, name3, mem3, attrs3, minor3), (u1, found1, name1, mem1, attrs1, minor1) = out
    for u in (u3, u1):
        assert re.fullmatch(r"MIG-[0-9a-f]{8}-[0-9a-f]{4}-5[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}", u), u
    assert u3 != u1 and found3 and found1
    assert (name3, name1) == ("NVIDIA GB200 MIG 3g.93gb", "NVIDIA GB200 MIG 1g.23gb")
    assert (mem3, mem1) == (93696 << 20, 22528 << 20)
    assert attrs1 == [SUCCESS, {"multiprocessorCount": 18, "sharedCopyEngineCount": 2, "sharedDecoderCount": 1,
                                "sharedEncoderCount": 0, "sharedJpegCount": 1, "sharedOfaCount": 0,
                                "gpuInstanceSliceCount": 1, "computeInstanceSliceCount": 1, "memorySizeMB": 22528}]
    assert attrs3[1]["multiprocessorCount"] == 72 and attrs3[1]["gpuInstanceSliceCount"] == 3
    assert minor3 == minor1 == NOT_SUPPORTED


def test_mig_uuids_are_stable_across_processes(tray):
    make_layout(tray)
    body = "d = m.dev(0); out = [m.uuid(m.mig_handle(d, i)[1])[1] for i in range(2)]"
    assert tray.nvml(body) == tray.nvml(body)


def test_go_nvlib_get_profile_walk(tray):
    """go-nvlib MigDevice.GetProfile: from the MIG handle to its GPU and
    compute instances and back to profile indices whose ids match."""
    make_layout(tray)
    out = tray.nvml(f"""
d = m.dev(0)
out = []
for i in range(2):
    h = m.mig_handle(d, i)[1]
    gi = m.gi_by_id(d, m.gi_id(h)[1])[1]; ci = m.ci_by_id(gi, m.ci_id(h)[1])[1]
    gpid, cpid = m.gi_info(gi)[1]['profileId'], m.ci_info(ci)[1]['profileId']
    gidx = [j for j in range({PROFILE_COUNT}) if m.profile(d, j)[0] == 0 and m.profile(d, j)[1]['id'] == gpid]
    cidx = [j for j in range({CI_PROFILE_COUNT}) if m.ci_profile(gi, j)[0] == 0 and m.ci_profile(gi, j)[1]['id'] == cpid]
    out.append([gidx, cidx, m.attributes(h)[1]['memorySizeMB'] == m.profile(d, gidx[0])[1]['memorySizeMB']])
""")
    assert out == [[[2], [2], True], [[0], [0], True]]


def test_mig_devices_go_with_their_instances(tray):
    make_layout(tray)
    out = tray.nvml("""
d = m.dev(0)
gi = m.gi_by_id(d, 2)[1]; ci = m.ci_by_id(gi, 0)[1]
out = [m.destroy_ci(ci), m.destroy_gi(gi), [m.mig_handle(d, i)[0] for i in range(2)]]
""")
    assert out == [SUCCESS, SUCCESS, [SUCCESS, NOT_FOUND]]


def test_error_strings(tray):
    out = tray.nvml("import ctypes\nm.nvml.nvmlErrorString.restype = ctypes.c_char_p\n"
                    "out = [m.nvml.nvmlErrorString(r).decode() for r in (19, 23)]")
    assert out == ["In use by another client", "Insufficient Resources"]
