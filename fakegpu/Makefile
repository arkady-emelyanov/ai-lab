# Builds the fake NVIDIA userspace:
#   driver:     libcuda.so.1, libnvidia-ml.so.1
#   CUDA libs:  libcublas, libcublasLt (CUDA 13 and 12 SONAMEs), libnccl.so.2,
#               libcudnn.so.9
#   tools:      nvidia-smi; nvidia-container-cli and nvidia-container-runtime
#               (containers on the tray); nvidia-cuda-mps-control; for the DRA driver's
#               driver root (roles/fakegpu) nvidia-imex and nvidia-imex-ctl
#
# symbols/*.syms list every function the real libraries export. Each gets
# a weak "return 0" (= success) definition placed in section fg_weak; the
# hand-written ones in *_stub.c override them at link time.

CC      ?= cc
CFLAGS  ?= -O2 -g
CFLAGS  += -fPIC -fvisibility=hidden -Wall -Wextra -Wno-unused-function -include glibc_compat.h
LDFLAGS += -shared -Wl,--no-undefined
PREFIX  ?= /usr/local
LIBDIR  ?= $(PREFIX)/lib/fakegpu

DRIVER_LIBS = libcuda.so.1 libnvidia-ml.so.1
# Framework wheels bundle these and load them via RPATH, so they are
# installed outside the linker path and forced in with /etc/ld.so.preload.
PRELOAD_LIBS = libcublas.so.13 libcublasLt.so.13 libcublas.so.12 libcublasLt.so.12 \
               libnccl.so.2 libcudnn.so.9
LIBS = $(DRIVER_LIBS) $(PRELOAD_LIBS)

CU12 = -DCUBLAS_VER_MAJOR=12 -DCUBLAS_VER_MINOR=9 -DCUBLAS_VER_PATCH=1 -DCUDART_VERSION=12090

TOOLS = nvidia-smi nvidia-container-cli nvidia-container-runtime nvidia-cuda-mps-control nvidia-imex nvidia-imex-ctl

all: $(LIBS) $(TOOLS)

$(LIBS) nvidia-smi nvidia-container-cli nvidia-container-runtime nvidia-cuda-mps-control: glibc_compat.h

build/%_weak.c: symbols/%.syms
	@mkdir -p build
	awk '{ printf "__attribute__((weak, visibility(\"default\"), section(\"fg_weak\"))) int %s(void) { return 0; }\n", $$1 }' $< > $@

libcuda.so.1: cuda_stub.c dark_api.c timing.c vmm.c build/cuda_weak.c fakegpu.h occupancy.h mig.h
	$(CC) $(CFLAGS) $(LDFLAGS) -Wl,-soname,$@ -o $@ cuda_stub.c dark_api.c timing.c vmm.c build/cuda_weak.c -ldl -lpthread

libnvidia-ml.so.1: nvml_stub.c build/nvml_weak.c fakegpu.h occupancy.h mig.h
	$(CC) $(CFLAGS) $(LDFLAGS) -Wl,-soname,$@ -o $@ nvml_stub.c build/nvml_weak.c -lpthread -lm

libcublas.so.13: cublas_stub.c build/cublas_weak.c stub_common.h fakegpu.h
	$(CC) $(CFLAGS) $(LDFLAGS) -Wl,-soname,$@ -o $@ cublas_stub.c build/cublas_weak.c

libcublasLt.so.13: cublas_stub.c build/cublasLt_weak.c stub_common.h fakegpu.h
	$(CC) $(CFLAGS) -DSTUB_CUBLASLT $(LDFLAGS) -Wl,-soname,$@ -o $@ cublas_stub.c build/cublasLt_weak.c

libcublas.so.12: cublas_stub.c build/cublas_weak.c stub_common.h fakegpu.h
	$(CC) $(CFLAGS) $(CU12) $(LDFLAGS) -Wl,-soname,$@ -o $@ cublas_stub.c build/cublas_weak.c

libcublasLt.so.12: cublas_stub.c build/cublasLt_weak.c stub_common.h fakegpu.h
	$(CC) $(CFLAGS) $(CU12) -DSTUB_CUBLASLT $(LDFLAGS) -Wl,-soname,$@ -o $@ cublas_stub.c build/cublasLt_weak.c

libnccl.so.2: nccl_stub.c build/nccl_weak.c stub_common.h fakegpu.h
	$(CC) $(CFLAGS) $(LDFLAGS) -Wl,-soname,$@ -o $@ nccl_stub.c build/nccl_weak.c -ldl -lpthread

# MPS control daemon for NVIDIA's GPU fractioning (mps_control.c).
nvidia-cuda-mps-control: mps_control.c
	$(CC) $(CFLAGS) -o $@ $<

# IMEX daemon and its control tool (imex.c). Static: they run in NVIDIA's
# DRA driver image (Debian 12's glibc), mounted from the driver root.
nvidia-imex: imex.c
	$(CC) -O2 -Wall -Wextra -static -o $@ $<

nvidia-imex-ctl: imex.c
	$(CC) -O2 -Wall -Wextra -static -DIMEX_CTL -o $@ $<

# nvidia-smi (smi.c): loads libnvidia-ml.so.1 at run time, as NVIDIA's does;
# glibc_compat.h keeps it running on older glibc, so in containers too.
nvidia-smi: smi.c
	$(CC) $(CFLAGS) -o $@ $< -ldl -lm

# The NVIDIA container toolkit's entry points for container runtimes on the
# tray (container_cli.c, container_runtime.c, json.h): the fake driver from
# the tray's CDI specification.
nvidia-container-cli: container_cli.c json.h
	$(CC) $(CFLAGS) -o $@ $<

nvidia-container-runtime: container_runtime.c json.h
	$(CC) $(CFLAGS) -o $@ $<

libcudnn.so.9: cudnn_stub.c build/cudnn_weak.c stub_common.h fakegpu.h
	$(CC) $(CFLAGS) $(LDFLAGS) -Wl,-soname,$@ -o $@ cudnn_stub.c build/cudnn_weak.c

# Driver libs go on the normal linker path (via an ld.so.conf.d entry);
# CUDA libs go to $(LIBDIR)/preload and are listed in preload.list.
install: $(LIBS) $(TOOLS)
	install -d $(DESTDIR)$(LIBDIR) $(DESTDIR)$(LIBDIR)/preload $(DESTDIR)$(LIBDIR)/driver-tools
	install -m 0755 $(DRIVER_LIBS) $(DESTDIR)$(LIBDIR)/
	ln -sf libcuda.so.1 $(DESTDIR)$(LIBDIR)/libcuda.so
	ln -sf libnvidia-ml.so.1 $(DESTDIR)$(LIBDIR)/libnvidia-ml.so
	install -m 0755 $(PRELOAD_LIBS) $(DESTDIR)$(LIBDIR)/preload/
	for l in $(PRELOAD_LIBS); do echo $(LIBDIR)/preload/$$l; done > $(DESTDIR)$(LIBDIR)/preload.list
	install -D -m 0755 nvidia-smi $(DESTDIR)$(PREFIX)/bin/nvidia-smi
	install -D -m 0755 numactl $(DESTDIR)$(PREFIX)/bin/numactl
	install -D -m 0755 nvidia-cuda-mps-control $(DESTDIR)$(PREFIX)/bin/nvidia-cuda-mps-control
	install -D -m 0755 nvidia-container-cli $(DESTDIR)$(PREFIX)/bin/nvidia-container-cli
	install -D -m 0755 nvidia-container-runtime $(DESTDIR)$(PREFIX)/bin/nvidia-container-runtime
	install -m 0755 nvidia-imex nvidia-imex-ctl $(DESTDIR)$(LIBDIR)/driver-tools/
	install -m 0755 nvidia-smi $(DESTDIR)$(LIBDIR)/driver-tools/nvidia-smi

clean:
	rm -rf build $(LIBS) $(TOOLS)

.PHONY: all install clean
