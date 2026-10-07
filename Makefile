# AI lab: a GPU cluster modelled on GB200 NVL72, scaled down to two compute
# trays (NVL8), on Incus, with Slurm or Kubernetes.
#
#   make init        prepare the host: venv, Ansible collections, secrets, checks;
#                    creates local.yml (local settings, not in git)
#   make up          init + create instances + configure the cluster
#   make frameworks  shared PyTorch + Ray venv on /shared (several GB)
#   make test        end-to-end checks (Slurm, GPUs, NCCL, Ray, BMC, metrics)
#   make test-fakegpu  fake GPU library tests on this machine, no lab needed (NCCL
#                    across NVLink partitions, InfiniBand counters)
#   make test-bmc    BMC integration tests (pytest); -disruptive power-cycles a tray,
#                    -conformance checks the known gaps to real GB200 behaviour
#   make shell       login node as joe          make shell-root   login node as root
#   make shell-NODE  root shell on any instance (e.g. make shell-sched-worker1)
#   bin/ssh login    ssh as joe to the login node (root@login, sched-control, sched-worker1, ...)
#   make down        delete instances (volumes kept)
#   make purge       delete instances and volumes
#
# Commands needing the Incus socket run under `sg incus-admin` so a fresh
# group membership works without re-login.

VENV     := .venv
ANSIBLE  := $(VENV)/bin/ansible-playbook
INCUS_SG := sg incus-admin -c
SECRETS  := .secrets
, := ,
LOCAL    := local.yml
# Ansible refuses non-blocking stdio (some terminals/IDEs); detach stdin.
# The checkout's own settings (local.yml) override everything else.
RUN      = $(INCUS_SG) '$(ANSIBLE) $$(test -f $(LOCAL) && echo --extra-vars=@$(LOCAL)) $(1) </dev/null'

.PHONY: init proto check up provision configure frameworks test test-fakegpu test-bmc test-bmc-disruptive test-bmc-conformance down purge shell shell-root shell-%

init: $(LOCAL) $(VENV)/.done fakebmc/fakebmc fakenmxc/fakenmxc fakedp/fakedp .cache/topograph/topograph $(SECRETS)/munge.key $(SECRETS)/slurmdbd.pass $(SECRETS)/ldap-admin.pass $(SECRETS)/redis.pass $(SECRETS)/grafana.pass $(SECRETS)/rustfs.access $(SECRETS)/rustfs.secret $(SECRETS)/k3s.token $(SECRETS)/ssh/id_ed25519 $(SECRETS)/ssh/controller_ed25519 check
	@chmod -R go-rwx $(SECRETS)

# Local settings, not in git: created once, never overwritten.
$(LOCAL):
	cp local.example.yml $@

$(VENV)/.done: requirements.yml
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install -q ansible-core passlib
	$(VENV)/bin/ansible-galaxy collection install -r requirements.yml -p .collections </dev/null
	touch $@

# Tray BMC (Redfish) service: static binary copied into the BMC containers.
fakebmc/fakebmc: $(wildcard fakebmc/*.go) fakebmc/go.mod
	@command -v go >/dev/null || { echo "ERROR: Go is required to build fakebmc"; exit 1; }
	cd fakebmc && CGO_ENABLED=0 go build -trimpath -ldflags='-s -w' -o fakebmc .

# NVLink partition controller (gRPC); generated code is committed, `make proto`
# regenerates it (needs protoc and the Go plugins in .cache/tools).
fakenmxc/fakenmxc: $(wildcard fakenmxc/*.go) $(wildcard fakenmxc/gen/nmxlabv1/*.go) fakenmxc/go.mod
	@command -v go >/dev/null || { echo "ERROR: Go is required to build fakenmxc"; exit 1; }
	cd fakenmxc && CGO_ENABLED=0 go build -trimpath -ldflags='-s -w' -o fakenmxc .

# GPU device plugin for the k3s scheduler (kubelet API, CDI devices).
fakedp/fakedp: $(wildcard fakedp/*.go) fakedp/go.mod
	@command -v go >/dev/null || { echo "ERROR: Go is required to build fakedp"; exit 1; }
	cd fakedp && CGO_ENABLED=0 go build -trimpath -ldflags='-s -w' -o fakedp .

proto:
	PATH=$(CURDIR)/.cache/tools/bin:$$PATH $(CURDIR)/.cache/tools/protoc/bin/protoc -I fakenmxc/proto \
		--go_out=fakenmxc/gen/nmxlabv1 --go_opt=paths=source_relative \
		--go-grpc_out=fakenmxc/gen/nmxlabv1 --go-grpc_opt=paths=source_relative fakenmxc/proto/nmxc.proto

$(SECRETS)/munge.key:
	mkdir -p -m 0700 $(SECRETS)
	dd if=/dev/urandom of=$@ bs=1024 count=1 status=none
	chmod 0600 $@

$(SECRETS)/slurmdbd.pass $(SECRETS)/ldap-admin.pass $(SECRETS)/redis.pass $(SECRETS)/grafana.pass $(SECRETS)/rustfs.secret $(SECRETS)/k3s.token:
	mkdir -p -m 0700 $(SECRETS)
	head -c 24 /dev/urandom | base64 | tr -d '/+=' > $@
	chmod 0600 $@

$(SECRETS)/rustfs.access:
	mkdir -p -m 0700 $(SECRETS)
	echo "lab$$(head -c 8 /dev/urandom | od -An -tx1 | tr -d ' \n')" > $@
	chmod 0600 $@

$(SECRETS)/ssh/id_ed25519:
	mkdir -p -m 0700 $(SECRETS)/ssh
	ssh-keygen -q -t ed25519 -N '' -C nvl8-cluster -f $@

# The controller's own key (pdsh to the trays for topograph).
$(SECRETS)/ssh/controller_ed25519:
	mkdir -p -m 0700 $(SECRETS)/ssh
	ssh-keygen -q -t ed25519 -N '' -C sched-control -f $@

# topograph needs a newer Go than may be installed: the go command fetches
# the required toolchain (verified against the checksum database).
TOPOGRAPH_REF := $(shell cat inventory/group_vars/all.yml $(wildcard $(LOCAL)) | sed -n 's/^topograph_ref: *//p' | tail -1)
.cache/topograph/topograph:
	@command -v go >/dev/null || { echo "ERROR: Go is required to build topograph"; exit 1; }
	test -d .cache/topograph/.git || git clone -q https://github.com/dsx-ai-factory/topograph.git .cache/topograph
	cd .cache/topograph && git fetch -q origin && git checkout -q $(TOPOGRAPH_REF)
	cd .cache/topograph && GOSUMDB=sum.golang.org GOTOOLCHAIN=auto CGO_ENABLED=0 go build -trimpath -ldflags='-s -w' -o topograph ./cmd/topograph

# Host prerequisites need root, so they are reported rather than fixed.
check:
	@getent group incus-admin | grep -qw $(USER) \
		|| { echo "ERROR: not in incus-admin. Run: sudo usermod -aG incus-admin $(USER)"; exit 1; }
	@$(INCUS_SG) 'incus info >/dev/null' \
		|| { echo "ERROR: Incus daemon not reachable"; exit 1; }
	@if readlink /etc/resolv.conf | grep -q NetworkManager/no-stub \
	   && ! test -f /etc/apparmor.d/abstractions/nameservice.d/networkmanager-no-stub; then \
		echo "ERROR: Incus dnsmasq cannot read NetworkManager's resolv.conf; containers get no DNS. Run:"; \
		echo "  sudo mkdir -p /etc/apparmor.d/abstractions/nameservice.d"; \
		echo "  echo '@{run}/NetworkManager/no-stub-resolv.conf r,' | sudo tee /etc/apparmor.d/abstractions/nameservice.d/networkmanager-no-stub"; \
		echo "  sudo systemctl restart incus"; \
		exit 1; \
	fi
	@test "$$(sysctl -n fs.inotify.max_user_instances)" -ge 1024 \
		|| { echo "ERROR: fs.inotify.max_user_instances is $$(sysctl -n fs.inotify.max_user_instances); containers' systemd runs out"; \
		     echo "  of inotify instances (\"Too many open files\"). Incus ships the fix; apply it: sudo sysctl --system"; exit 1; }
	@test "$$(sysctl -n kernel.keys.maxkeys)" -ge 2000 -a "$$(sysctl -n kernel.keys.maxbytes)" -ge 2000000 \
		|| { echo "ERROR: kernel.keys.maxkeys/maxbytes are $$(sysctl -n kernel.keys.maxkeys)/$$(sysctl -n kernel.keys.maxbytes); every unprivileged"; \
		     echo "  container shares one keyring quota, so container runtimes fail with \"disk quota exceeded\"."; \
		     echo "  Incus recommends 2000/2000000. Run:"; \
		     echo "  echo 'kernel.keys.maxkeys = 2000' | sudo tee /etc/sysctl.d/60-incus-keys.conf"; \
		     echo "  echo 'kernel.keys.maxbytes = 2000000' | sudo tee -a /etc/sysctl.d/60-incus-keys.conf"; \
		     echo "  sudo sysctl --system"; exit 1; }
	@echo "host ready"

up: init provision configure

provision: init
	$(call RUN,playbooks/provision.yml)

configure: init
	$(call RUN,playbooks/site.yml)

# Shared PyTorch/Ray venv on /shared (several GB).
frameworks: init
	$(call RUN,playbooks/frameworks.yml)

test: init
	$(call RUN,playbooks/test.yml)

# BMC integration tests from the host against the running lab (tests/bmc).
# Extra pytest arguments: make test-bmc PYTEST_ARGS='-k nvswitch -x'
PYTEST := $(VENV)/bin/pytest
PYTEST_ARGS ?=

$(PYTEST): $(VENV)/.done
	$(VENV)/bin/pip install -q pytest requests
	touch $@

# Builds the fake GPU libraries here and runs them with emulated trays.
test-fakegpu: $(PYTEST)
	$(PYTEST) tests/fakegpu $(PYTEST_ARGS)

test-bmc: init $(PYTEST)
	$(INCUS_SG) '$(PYTEST) tests/bmc $(PYTEST_ARGS) </dev/null'

test-bmc-disruptive: init $(PYTEST)
	$(INCUS_SG) '$(PYTEST) tests/bmc --disruptive $(PYTEST_ARGS) </dev/null'

test-bmc-conformance: init $(PYTEST)
	$(INCUS_SG) '$(PYTEST) tests/bmc --conformance -m conformance -rfxX --tb=line $(PYTEST_ARGS) </dev/null'

down: init
	$(call RUN,playbooks/destroy.yml)

purge: init
	$(call RUN,playbooks/destroy.yml --tags all$(,)purge)

shell:
	$(INCUS_SG) 'incus exec sched-login -- su - joe'

shell-root:
	$(INCUS_SG) 'incus exec sched-login -- bash -l'

shell-%:
	$(INCUS_SG) 'incus exec $* -- bash -l'
