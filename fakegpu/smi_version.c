/*
 * nvidia-smi for the fake driver root (see roles/fakegpu): NVIDIA's DRA
 * driver checks a driver root by running `nvidia-smi --version` there, in
 * its own image (no Python, so not the lab's nvidia-smi). This one prints
 * the versions from the fake NVML and hands any other use to the lab's
 * nvidia-smi when it is there.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#define FULL_NVIDIA_SMI "/usr/local/bin/nvidia-smi"

int main(int argc, char **argv)
{
    if (argc < 2 || strcmp(argv[1], "--version")) {
        execv(FULL_NVIDIA_SMI, argv);
        fprintf(stderr, "nvidia-smi: only --version is available here\n");
        return 1;
    }
    char driver[80] = "", nvml[80] = "";
    int cuda = 0;
    void *h = dlopen("libnvidia-ml.so.1", RTLD_NOW);
    if (h) {
        int (*init)(void) = (int (*)(void))dlsym(h, "nvmlInit_v2");
        int (*drv)(char *, unsigned) = (int (*)(char *, unsigned))dlsym(h, "nvmlSystemGetDriverVersion");
        int (*ver)(char *, unsigned) = (int (*)(char *, unsigned))dlsym(h, "nvmlSystemGetNVMLVersion");
        int (*cu)(int *) = (int (*)(int *))dlsym(h, "nvmlSystemGetCudaDriverVersion_v2");
        if (init && init() == 0) {
            if (drv)
                drv(driver, sizeof(driver));
            if (ver)
                ver(nvml, sizeof(nvml));
            if (cu)
                cu(&cuda);
        }
    }
    if (!driver[0]) {
        fprintf(stderr, "nvidia-smi: NVML is not available\n");
        return 1;
    }
    printf("NVIDIA-SMI version  : %s\n", driver);
    printf("NVML version        : %s\n", nvml[0] ? nvml : driver);
    printf("DRIVER version      : %s\n", driver);
    printf("CUDA Version        : %d.%d\n", cuda / 1000, cuda % 1000 / 10);
    return 0;
}
