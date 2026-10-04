# Local NVL8 Slurm cluster on Incus.
#
#   make init        prepare the host: venv, Ansible collections, secrets, checks
#   make up          init + create instances + configure the cluster
#   make frameworks  shared PyTorch + Ray venv on /shared (several GB)
#   make test        end-to-end checks (Slurm, GPUs, NCCL, Ray, BMC, metrics)
#   make shell       login node as joe          make shell-root   login node as root
#   make shell-NODE  root shell on any instance (e.g. make shell-slurm-worker1)
#   bin/ssh slurm    ssh as joe to the login node (root@slurm, slurm-control, slurm-worker1, ...)
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
# Ansible refuses non-blocking stdio (some terminals/IDEs); detach stdin.
RUN      = $(INCUS_SG) '$(ANSIBLE) $(1) </dev/null'

.PHONY: init proto check up provision configure frameworks test down purge shell shell-root shell-%

init: $(VENV)/.done fakebmc/fakebmc fakenmxc/fakenmxc .cache/topograph/topograph $(SECRETS)/munge.key $(SECRETS)/slurmdbd.pass $(SECRETS)/ldap-admin.pass $(SECRETS)/redis.pass $(SECRETS)/grafana.pass $(SECRETS)/rustfs.access $(SECRETS)/rustfs.secret $(SECRETS)/ssh/id_ed25519 $(SECRETS)/ssh/controller_ed25519 check
	@chmod -R go-rwx $(SECRETS)

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

proto:
	PATH=$(CURDIR)/.cache/tools/bin:$$PATH $(CURDIR)/.cache/tools/protoc/bin/protoc -I fakenmxc/proto \
		--go_out=fakenmxc/gen/nmxlabv1 --go_opt=paths=source_relative \
		--go-grpc_out=fakenmxc/gen/nmxlabv1 --go-grpc_opt=paths=source_relative fakenmxc/proto/nmxc.proto

$(SECRETS)/munge.key:
	mkdir -p -m 0700 $(SECRETS)
	dd if=/dev/urandom of=$@ bs=1024 count=1 status=none
	chmod 0600 $@

$(SECRETS)/slurmdbd.pass $(SECRETS)/ldap-admin.pass $(SECRETS)/redis.pass $(SECRETS)/grafana.pass $(SECRETS)/rustfs.secret:
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
	ssh-keygen -q -t ed25519 -N '' -C slurm-control -f $@

# topograph needs a newer Go than may be installed: the go command fetches
# the required toolchain (verified against the checksum database).
TOPOGRAPH_REF := $(shell sed -n 's/^topograph_ref: *//p' inventory/group_vars/all.yml)
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

down: init
	$(call RUN,playbooks/destroy.yml)

purge: init
	$(call RUN,playbooks/destroy.yml --tags all$(,)purge)

shell:
	$(INCUS_SG) 'incus exec slurm-login -- su - joe'

shell-root:
	$(INCUS_SG) 'incus exec slurm-login -- bash -l'

shell-%:
	$(INCUS_SG) 'incus exec $* -- bash -l'
