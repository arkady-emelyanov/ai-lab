/*
 * Included before every source (-include): keeps the libraries loadable by
 * older glibc (down to 2.33, for stat), as NVIDIA's are, so they work in
 * containers built on older distributions (CUDA and NVIDIA images on
 * Ubuntu 22.04: glibc 2.35).
 *
 * - glibc 2.38 redirects strtol, sscanf and friends to C23 variants
 *   (__isoc23_*, GLIBC_2.38) under _GNU_SOURCE; its own switch turns that off.
 * - glibc 2.34 moved dlopen, dlsym, dladdr and pthread_once into libc under
 *   a new version; libc still exports the original ones, which every glibc has.
 */
#ifndef FAKEGPU_GLIBC_COMPAT_H
#define FAKEGPU_GLIBC_COMPAT_H

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <features.h>
#undef __GLIBC_USE_C2X_STRTOL
#define __GLIBC_USE_C2X_STRTOL 0

#if defined(__x86_64__)
#define FAKEGPU_GLIBC_BASE "GLIBC_2.2.5"
#elif defined(__aarch64__)
#define FAKEGPU_GLIBC_BASE "GLIBC_2.17"
#endif
#ifdef FAKEGPU_GLIBC_BASE
__asm__(".symver dlopen,dlopen@" FAKEGPU_GLIBC_BASE);
__asm__(".symver dlsym,dlsym@" FAKEGPU_GLIBC_BASE);
__asm__(".symver dladdr,dladdr@" FAKEGPU_GLIBC_BASE);
__asm__(".symver pthread_once,pthread_once@" FAKEGPU_GLIBC_BASE);
#endif

#endif
