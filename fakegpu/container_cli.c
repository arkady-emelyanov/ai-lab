/*
 * nvidia-container-cli look-alike for the fake driver (libnvidia-container's CLI).
 *
 * Container runtimes on a GPU node call nvidia-container-cli to bring the
 * driver into a container: enroot's NVIDIA hook (98-nvidia.sh, so Pyxis
 * jobs), Apptainer's --nv with `use nvidia-container-cli = yes`, NVIDIA's
 * legacy OCI hook. This one brings the fake driver, as the tray's CDI
 * specification (/etc/cdi/fakegpu.json, FAKEGPU_CDI_SPEC) describes it for
 * Kubernetes pods: the GPUs' device nodes, the libraries, nvidia-smi, the
 * tray's configuration, sideband, telemetry and CUDA IPC directories, plus
 * the NCCL and fabric directories of the NVLink domain. The spec's
 * LD_PRELOAD, an environment variable this tool cannot set, becomes the
 * container's /etc/ld.so.preload (the host's, which lists the same stubs).
 *
 *   nvidia-container-cli [--user] [--debug=FILE] [--load-kmods] ... COMMAND
 *   COMMAND: info [--csv], list [--device=...] [--binaries] [--libraries],
 *            configure [--device=all|0,1|GPU-UUID] [--compute] [--utility] ...
 *                      [--ldconfig=@PATH] [--no-cgroups] [--pid=PID] ROOTFS
 */
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdarg.h>
#include <sys/mount.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/wait.h>
#include <unistd.h>

#include "json.h"

#define VERSION "1.17.8"
#define MAX_ITEMS 256

static const char *debug_path;

static void die(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    fputs("nvidia-container-cli: ", stderr);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
    exit(1);
}

static void debug(const char *fmt, ...)
{
    if (!debug_path)
        return;
    FILE *f = fopen(debug_path, "a");
    if (!f)
        return;
    va_list ap;
    va_start(ap, fmt);
    fputs("fakegpu: ", f);
    vfprintf(f, fmt, ap);
    va_end(ap);
    fputc('\n', f);
    fclose(f);
}

static const char *spec_path(void)
{
    const char *p = getenv("FAKEGPU_CDI_SPEC");
    return p && *p ? p : "/etc/cdi/fakegpu.json";
}

static json_t *load_spec(void)
{
    json_t *spec = json_load(spec_path());
    if (!spec)
        die("initialization error: driver error: no CDI specification of the GPUs (%s: %s)", spec_path(),
            errno ? strerror(errno) : "not JSON");
    return spec;
}

static int is_index(const char *s)
{
    if (!*s)
        return 0;
    for (; *s; s++)
        if (*s < '0' || *s > '9')
            return 0;
    return 1;
}

/* The tray's GPUs from the spec's device names: an index and a UUID per node. */
typedef struct { int index; const char *uuid, *node; } gpu_t;

static int gpus(json_t *spec, gpu_t *out)
{
    int n = 0;
    json_t *devices = json_get(spec, "devices");
    for (size_t i = 0; devices && i < devices->n; i++) {
        json_t *d = devices->items[i];
        const char *name = json_str(json_get(d, "name"));
        json_t *nodes = json_get(json_get(d, "containerEdits"), "deviceNodes");
        const char *node = nodes && nodes->n ? json_str(json_get(nodes->items[0], "path")) : NULL;
        if (!name || !node)
            continue;
        int k;
        for (k = 0; k < n && strcmp(out[k].node, node); k++)
            ;
        if (k == n) {
            if (n == MAX_ITEMS)
                break;
            out[n++] = (gpu_t){-1, "", node};
        }
        if (is_index(name))
            out[k].index = atoi(name);
        else
            out[k].uuid = name;
    }
    /* by index, as NVML numbers them */
    for (int i = 1; i < n; i++)
        for (int j = i; j > 0 && out[j - 1].index > out[j].index; j--) {
            gpu_t t = out[j];
            out[j] = out[j - 1];
            out[j - 1] = t;
        }
    int m = 0;
    for (int i = 0; i < n; i++)
        if (out[i].index >= 0)
            out[m++] = out[i];
    return m;
}

/* Device nodes for --device (all, none, indexes or UUIDs, comma-separated). */
static int select_nodes(json_t *spec, const char *devices, const char **nodes)
{
    gpu_t tray[MAX_ITEMS];
    int ng = gpus(spec, tray), n = 0;
    if (!devices || !*devices || !strcmp(devices, "none") || !strcmp(devices, "void"))
        return 0;
    if (!strcmp(devices, "all")) {
        for (int i = 0; i < ng; i++)
            nodes[n++] = tray[i].node;
        return n;
    }
    char *list = strdup(devices), *save, *want;
    for (want = strtok_r(list, ",", &save); want; want = strtok_r(NULL, ",", &save)) {
        int hit = 0;
        for (int i = 0; i < ng && n < MAX_ITEMS; i++) {
            char idx[16];
            snprintf(idx, sizeof idx, "%d", tray[i].index);
            if (!strcmp(want, idx) || !strcasecmp(want, tray[i].uuid)) {
                nodes[n++] = tray[i].node;
                hit = 1;
            }
        }
        if (!hit)
            die("device error: %s: unknown device", want);
    }
    free(list);
    return n;
}

/* A key of /etc/fakegpu.conf, or NULL. */
static char *fakegpu_conf(const char *key)
{
    static char value[PATH_MAX];
    FILE *f = fopen("/etc/fakegpu.conf", "r");
    if (!f)
        return NULL;
    char line[PATH_MAX + 64];
    char *found = NULL;
    while (!found && fgets(line, sizeof line, f)) {
        char *p = line;
        while (*p == ' ' || *p == '\t')
            p++;
        char *eq = strchr(p, '=');
        if (*p == '#' || !eq)
            continue;
        char *k = p, *ke = eq;
        while (ke > k && (ke[-1] == ' ' || ke[-1] == '\t'))
            ke--;
        if ((size_t)(ke - k) != strlen(key) || strncmp(k, key, strlen(key)))
            continue;
        char *v = eq + 1;
        while (*v == ' ' || *v == '\t')
            v++;
        size_t len = strlen(v);
        while (len && (v[len - 1] == '\n' || v[len - 1] == ' ' || v[len - 1] == '\t' || v[len - 1] == '\r'))
            len--;
        snprintf(value, sizeof value, "%.*s", (int)len, v);
        found = value;
    }
    fclose(f);
    return found;
}

static int exists(const char *p, int dir)
{
    struct stat st;
    return stat(p, &st) == 0 && (!dir || S_ISDIR(st.st_mode));
}

/* What the container gets besides the GPUs' nodes: the spec's mounts, the
 * NVLink domain's directories, the tray's preload list and linker path. */
typedef struct { char src[PATH_MAX], dst[PATH_MAX]; int ro; } mount_t;

static int mounts(json_t *spec, mount_t *out)
{
    int n = 0;
    json_t *list = json_get(json_get(spec, "containerEdits"), "mounts");
    for (size_t i = 0; list && i < list->n && n < MAX_ITEMS; i++) {
        json_t *m = list->items[i];
        const char *host = json_str(json_get(m, "hostPath")), *cont = json_str(json_get(m, "containerPath"));
        if (!host || !cont)
            continue;
        json_t *opts = json_get(m, "options");
        int ro = 0;
        for (size_t k = 0; opts && k < opts->n; k++)
            ro |= json_str(opts->items[k]) && !strcmp(json_str(opts->items[k]), "ro");
        snprintf(out[n].src, PATH_MAX, "%s", host);
        snprintf(out[n].dst, PATH_MAX, "%s", cont);
        out[n++].ro = ro;
    }
    static const char *domain_keys[] = {"nccl_dir", "fabric_dir"};
    for (size_t i = 0; i < 2 && n < MAX_ITEMS; i++) {
        const char *d = fakegpu_conf(domain_keys[i]);
        if (d && *d && exists(d, 1)) {
            snprintf(out[n].src, PATH_MAX, "%s", d);
            snprintf(out[n].dst, PATH_MAX, "%s", d);
            out[n++].ro = 0;
        }
    }
    static const char *files[] = {"/etc/ld.so.preload", "/etc/ld.so.conf.d/00-fakegpu.conf"};
    for (size_t i = 0; i < 2 && n < MAX_ITEMS; i++)
        if (exists(files[i], 0)) {
            snprintf(out[n].src, PATH_MAX, "%s", files[i]);
            snprintf(out[n].dst, PATH_MAX, "%s", files[i]);
            out[n++].ro = 1;
        }
    return n;
}

static void mkdirs(const char *path)
{
    char buf[PATH_MAX];
    snprintf(buf, sizeof buf, "%s", path);
    for (char *p = buf + 1; *p; p++)
        if (*p == '/') {
            *p = 0;
            mkdir(buf, 0755);
            *p = '/';
        }
    mkdir(buf, 0755);
}

/* mount(2) itself: mount(8) refuses to run in the user namespaces of
 * unprivileged runtimes (enroot, Apptainer), as setuid-aware tools do. */
static void do_mount(const char *src, const char *target, unsigned long flags)
{
    debug("mount %s %s %#lx", src, target, flags);
    if (mount(src, target, NULL, flags, NULL) != 0)
        die("mount error: %s at %s: %s", src, target, strerror(errno));
}

/* Bind src at rootfs/dst, creating the target as the source's kind; a
 * read-only remount keeps the flags a user namespace locks. */
static void bind(const char *rootfs, const char *src, const char *dst, int ro)
{
    char target[PATH_MAX * 2];
    snprintf(target, sizeof target, "%s/%s", rootfs, dst[0] == '/' ? dst + 1 : dst);
    if (exists(src, 1))
        mkdirs(target);
    else {
        char dir[sizeof target];
        snprintf(dir, sizeof dir, "%s", target);
        char *slash = strrchr(dir, '/');
        if (slash) {
            *slash = 0;
            mkdirs(dir);
        }
        struct stat st;
        if (lstat(target, &st) != 0) {
            FILE *f = fopen(target, "a");
            if (f)
                fclose(f);
        }
    }
    do_mount(src, target, MS_BIND | MS_REC);
    if (ro) {
        struct statvfs vfs;
        unsigned long keep = 0;
        if (statvfs(target, &vfs) == 0) {
            keep |= vfs.f_flag & ST_NOSUID ? MS_NOSUID : 0;
            keep |= vfs.f_flag & ST_NODEV ? MS_NODEV : 0;
            keep |= vfs.f_flag & ST_NOEXEC ? MS_NOEXEC : 0;
        }
        do_mount(src, target, MS_REMOUNT | MS_BIND | MS_RDONLY | keep);
    }
}

typedef struct { const char *device, *pid, *ldconfig; int binaries, libraries, has_device; } opts_t;

static int configure(opts_t *o, int argc, char **argv)
{
    if (!argc)
        die("configure: missing ROOTFS");
    char rootfs[PATH_MAX * 2], real[PATH_MAX];
    if (!realpath(argv[argc - 1], real))
        snprintf(real, sizeof real, "%s", argv[argc - 1]);
    if (o->pid)
        snprintf(rootfs, sizeof rootfs, "/proc/%s/root/%s", o->pid, real[0] == '/' ? real + 1 : real);
    else
        snprintf(rootfs, sizeof rootfs, "%s", real);
    json_t *spec = load_spec();
    json_t *common = json_get(spec, "containerEdits");
    json_t *dn = json_get(common, "deviceNodes");
    for (size_t i = 0; dn && i < dn->n; i++)
        if (json_str(json_get(dn->items[i], "path")))
            bind(rootfs, json_str(json_get(dn->items[i], "path")), json_str(json_get(dn->items[i], "path")), 0);
    const char *nodes[MAX_ITEMS];
    int nn = select_nodes(spec, o->device, nodes);
    for (int i = 0; i < nn; i++)
        bind(rootfs, nodes[i], nodes[i], 0);
    static mount_t m[MAX_ITEMS];
    int nm = mounts(spec, m);
    for (int i = 0; i < nm; i++)
        bind(rootfs, m[i].src, m[i].dst, m[i].ro);
    if (o->ldconfig) {
        const char *ldconfig = o->ldconfig[0] == '@' ? o->ldconfig + 1 : o->ldconfig;
        pid_t pid = fork();
        if (pid == 0) {
            int null = open("/dev/null", O_WRONLY);
            if (null >= 0)
                dup2(null, 2);
            execl(ldconfig, ldconfig, "-r", rootfs, (char *)NULL);
            _exit(127);
        }
        int status = 0;
        if (pid > 0)
            waitpid(pid, &status, 0);
        debug("%s -r %s: %d", ldconfig, rootfs, status);
    }
    return 0;
}

static int info(int argc, char **argv)
{
    int csv = 0;
    for (int i = 0; i < argc; i++)
        csv |= !strcmp(argv[i], "--csv");
    FILE *p = popen("nvidia-smi --query-gpu=driver_version,name,uuid,pci.bus_id,index --format=csv,noheader 2>&1", "r");
    char rows[MAX_ITEMS][512];
    int n = 0;
    while (p && n < MAX_ITEMS && fgets(rows[n], sizeof rows[n], p)) {
        rows[n][strcspn(rows[n], "\n")] = 0;
        n++;
    }
    int status = p ? pclose(p) : -1;
    if (status != 0)
        die("initialization error: nvml error: %s", n ? rows[0] : "driver not loaded");
    const char *cuda = fakegpu_conf("cuda_version");
    cuda = cuda ? strdup(cuda) : "";
    char f[5][512];
    if (csv)
        puts("NVRM version,CUDA version,Device Index,Device Minor,Model,Brand,GPU UUID,Bus Location,Architecture");
    for (int i = 0; i < n; i++) {
        memset(f, 0, sizeof f);
        int k = 0;
        for (char *s = rows[i], *t; k < 5 && s; s = t, k++) {
            t = strstr(s, ", ");
            if (t)
                *t = 0, t += 2;
            snprintf(f[k], sizeof f[k], "%.511s", s);
        }
        if (csv) {
            printf("%s,%s,%s,%s,%s,Nvidia,%s,%s,10.0\n", f[0], cuda, f[4], f[4], f[1], f[2], f[3]);
            continue;
        }
        if (i == 0)
            printf("NVRM version:   %s\nCUDA version:   %s\n\n", f[0], cuda);
        printf("Device Index:   %s\nDevice Minor:   %s\nModel:          %s\nBrand:          Nvidia\n"
               "GPU UUID:       %s\nBus Location:   %s\nArchitecture:   10.0\n\n", f[4], f[4], f[1], f[2], f[3]);
    }
    if (!n && !csv)
        printf("NVRM version:   \nCUDA version:   %s\n\n", cuda);
    return 0;
}

static int list(opts_t *o)
{
    json_t *spec = load_spec();
    int every = !(o->binaries || o->libraries || o->has_device);
    if (every || o->has_device) {
        json_t *dn = json_get(json_get(spec, "containerEdits"), "deviceNodes");
        for (size_t i = 0; dn && i < dn->n; i++)
            if (json_str(json_get(dn->items[i], "path")))
                puts(json_str(json_get(dn->items[i], "path")));
        const char *nodes[MAX_ITEMS];
        int nn = select_nodes(spec, o->has_device ? o->device : "all", nodes);
        for (int i = 0; i < nn; i++)
            puts(nodes[i]);
    }
    static mount_t m[MAX_ITEMS];
    int nm = mounts(spec, m);
    for (int i = 0; i < nm; i++) {
        size_t len = strlen(m[i].src);
        if (exists(m[i].src, 1) && (every || o->libraries) && len >= 8 && !strcmp(m[i].src + len - 8, "/fakegpu")) {
            struct dirent **names;
            int nd = scandir(m[i].src, &names, NULL, alphasort);
            for (int k = 0; k < nd; k++) {
                if (strstr(names[k]->d_name, ".so"))
                    printf("%s/%s\n", m[i].src, names[k]->d_name);
                free(names[k]);
            }
            if (nd >= 0)
                free(names);
        } else if (!exists(m[i].src, 1) && exists(m[i].src, 0) && access(m[i].src, X_OK) == 0 &&
                   strstr(m[i].src, "/bin/") && (every || o->binaries))
            puts(m[i].src);
    }
    return 0;
}

int main(int argc, char **argv)
{
    opts_t o = {0};
    const char *cmd = NULL;
    char **args = calloc((size_t)argc, sizeof *args);
    int nargs = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!cmd && (!strcmp(a, "-V") || !strcmp(a, "--version"))) {
            printf("cli-version: %s\nlib-version: %s\nbuild date: fakegpu\n", VERSION, VERSION);
            return 0;
        }
        const char *eq = strchr(a, '=');
        if (!strncmp(a, "--", 2) && eq) {
            size_t k = (size_t)(eq - a - 2);
            const char *v = eq + 1;
            if (k == 6 && !strncmp(a + 2, "device", 6))
                o.device = v, o.has_device = 1;
            else if (k == 3 && !strncmp(a + 2, "pid", 3))
                o.pid = v;
            else if (k == 8 && !strncmp(a + 2, "ldconfig", 8))
                o.ldconfig = v;
            else if (k == 5 && !strncmp(a + 2, "debug", 5))
                debug_path = v;
            continue;
        }
        if (a[0] == '-') {
            if (cmd && (!strcmp(a, "--device") || !strcmp(a, "--pid") || !strcmp(a, "--ldconfig")))
                die("%s needs a value (%s=...)", a, a);
            o.binaries |= !strcmp(a, "--binaries");
            o.libraries |= !strcmp(a, "--libraries");
            if (cmd)
                args[nargs++] = argv[i];
            continue;
        }
        if (!cmd)
            cmd = a;
        else
            args[nargs++] = argv[i];
    }
    if (cmd && !strcmp(cmd, "configure"))
        return configure(&o, nargs, args);
    if (cmd && !strcmp(cmd, "info"))
        return info(nargs, args);
    if (cmd && !strcmp(cmd, "list"))
        return list(&o);
    die("unknown command '%s' (info, list, configure)", cmd ? cmd : "");
    return 1;
}
