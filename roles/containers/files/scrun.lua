-- scrun's staging (Slurm 23.11), for podman on the login node: the bundle
-- podman made is in the user's rootless storage on /shared, which every
-- tray reaches, so the image is not copied. The container runs rootless on
-- the tray: its user namespace maps the container's user to the job's, it
-- runs in the job step's cgroup, and the files podman keeps in its runtime
-- directory on this node (resolv.conf, hosts, hostname, .containerenv) go
-- with the bundle.
local json = require("dkjson")

local function read_config(path)
	local f = assert(io.open(path, "r"))
	local config = assert(json.decode(f:read("*a")))
	f:close()
	return config
end

local function write_config(path, config)
	local f = assert(io.open(path, "w"))
	f:write(json.encode(config))
	f:close()
end

-- Copies a file from this node's /run into the shared bundle.
local function stage_file(source, staged)
	local src = io.open(source, "rb")
	if not src then return nil end
	local copy = staged .. "/" .. source:gsub("[^%w%.%-_]", "_")
	local dst = assert(io.open(copy, "wb"))
	dst:write(src:read("*a"))
	dst:close()
	src:close()
	return copy
end

function slurm_scrun_stage_in(id, bundle, spool_dir, config_file, job_id, user_id, group_id, job_env)
	local config = read_config(config_file)
	local staged = bundle .. "/scrun-staged"
	os.execute("mkdir -p '" .. staged .. "'")

	local mounts = {}
	for _, m in ipairs(config.mounts or {}) do
		local keep = true
		if m.type == "bind" and m.source and m.source:match("^/run/") then
			m.source = stage_file(m.source, staged)
			keep = m.source ~= nil
		end
		-- Only the job's user exists in the namespace: no uid= or gid=.
		local options = {}
		for _, o in ipairs(m.options or {}) do
			if not (o:match("^gid=") or o:match("^uid=")) then table.insert(options, o) end
		end
		m.options = options
		if keep then table.insert(mounts, m) end
	end
	config.mounts = mounts

	config.linux = config.linux or {}
	config.linux.cgroupsPath = nil
	local namespaces, has_user = config.linux.namespaces or {}, false
	for _, ns in ipairs(namespaces) do
		if ns.type == "user" then has_user = true end
	end
	if not has_user then table.insert(namespaces, {type = "user"}) end
	config.linux.namespaces = namespaces
	local user = config.process.user
	config.linux.uidMappings = {{containerID = user.uid, hostID = math.floor(user_id), size = 1}}
	config.linux.gidMappings = {{containerID = user.gid, hostID = math.floor(group_id), size = 1}}

	write_config(config_file, config)
	return slurm.SUCCESS
end

function slurm_scrun_stage_out(id, bundle, orig_bundle, root_path, orig_root_path, spool_dir, config_file, jobid, user_id, group_id)
	return slurm.SUCCESS
end

return slurm.SUCCESS
