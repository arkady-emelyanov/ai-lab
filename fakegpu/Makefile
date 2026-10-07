# Builds the fake NVIDIA userspace:
#   driver:     libcuda.so.1, libnvidia-ml.so.1
#   CUDA libs:  libcublas, libcublasLt (CUDA 13 and 12 SONAMEs), libnccl.so.2,
#               libcudnn.so.9
#
# symbols/*.syms list every function the real libraries export. Each gets
# a weak "return 0" (= success) definition placed in section fg_weak; the
# hand-written ones in *_stub.c override them at link time.

CC      ?= cc
CFLAGS  ?= -O2 -g
CFLAGS  += -fPIC -fvisibility=hidden -Wall -Wextra -Wno-unused-function
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

all: $(LIBS)

build/%_weak.c: symbols/%.syms
	@mkdir -p build
	awk '{ printf "__attribute__((weak, visibility(\"default\"), section(\"fg_weak\"))) int %s(void) { return 0; }\n", $$1 }' $< > $@

libcuda.so.1: cuda_stub.c dark_api.c timing.c build/cuda_weak.c fakegpu.h occupancy.h
	$(CC) $(CFLAGS) $(LDFLAGS) -Wl,-soname,$@ -o $@ cuda_stub.c dark_api.c timing.c build/cuda_weak.c -ldl -lpthread

libnvidia-ml.so.1: nvml_stub.c build/nvml_weak.c fakegpu.h occupancy.h
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

libcudnn.so.9: cudnn_stub.c build/cudnn_weak.c stub_common.h fakegpu.h
	$(CC) $(CFLAGS) $(LDFLAGS) -Wl,-soname,$@ -o $@ cudnn_stub.c build/cudnn_weak.c

# Driver libs go on the normal linker path (via an ld.so.conf.d entry);
# CUDA libs go to $(LIBDIR)/preload and are listed in preload.list.
install: $(LIBS)
	install -d $(DESTDIR)$(LIBDIR) $(DESTDIR)$(LIBDIR)/preload
	install -m 0755 $(DRIVER_LIBS) $(DESTDIR)$(LIBDIR)/
	ln -sf libcuda.so.1 $(DESTDIR)$(LIBDIR)/libcuda.so
	ln -sf libnvidia-ml.so.1 $(DESTDIR)$(LIBDIR)/libnvidia-ml.so
	install -m 0755 $(PRELOAD_LIBS) $(DESTDIR)$(LIBDIR)/preload/
	for l in $(PRELOAD_LIBS); do echo $(LIBDIR)/preload/$$l; done > $(DESTDIR)$(LIBDIR)/preload.list
	install -D -m 0755 nvidia-smi $(DESTDIR)$(PREFIX)/bin/nvidia-smi
	install -D -m 0755 numactl $(DESTDIR)$(PREFIX)/bin/numactl

clean:
	rm -rf build $(LIBS)

.PHONY: all install clean
