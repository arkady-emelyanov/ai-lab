/*
 * nvidia-container-runtime look-alike (CDI mode) for the fake driver.
 *
 * An OCI runtime in front of crun, as NVIDIA's runtime is in front of runc:
 * for `create` and `run` it applies the tray's CDI specification
 * (/etc/cdi/fakegpu.json, FAKEGPU_CDI_SPEC) to the bundle's config.json, for
 * the GPUs NVIDIA_VISIBLE_DEVICES in the container's environment names (all,
 * none, indexes or UUIDs), or else the Slurm job step's GPUs
 * (SLURM_STEP_GPUS): their device nodes, the fake driver's mounts and
 * environment, and the NCCL and fabric directories of the NVLink domain.
 * Then it runs crun (FAKEGPU_OCI_RUNTIME) with the same arguments. Slurm's
 * oci.conf runs it for `srun --container` and scrun.
 */
#include <errno.h>
#include <limits.h>
#include <sys/stat.h>
#include <unistd.h>

#include "json.h"

static void die(const char *what, const char *path)
{
    fprintf(stderr, "nvidia-container-runtime: %s %s: %s\n", what, path, errno ? strerror(errno) : "not JSON");
    exit(1);
}

static const char *env_or(const char *name, const char *dflt)
{
    const char *v = getenv(name);
    return v && *v ? v : dflt;
}

static const char *bundle_dir(int argc, char **argv)
{
    for (int i = 0; i < argc; i++) {
        if ((!strcmp(argv[i], "--bundle") || !strcmp(argv[i], "-b")) && i + 1 < argc)
            return argv[i + 1];
        if (!strncmp(argv[i], "--bundle=", 9))
            return argv[i] + 9;
    }
    return ".";
}

/* VAR's value in an OCI process environment, or NULL. */
static const char *env_value(json_t *env, const char *var)
{
    size_t len = strlen(var);
    for (size_t i = 0; env && i < env->n; i++) {
        const char *e = json_str(env->items[i]);
        if (e && !strncmp(e, var, len) && e[len] == '=')
            return e + len + 1;
    }
    return NULL;
}

/* A key of /etc/fakegpu.conf (NULL if missing), into buf. */
static const char *fakegpu_conf(const char *key, char *buf, size_t size)
{
    FILE *f = fopen("/etc/fakegpu.conf", "r");
    if (!f)
        return NULL;
    char line[PATH_MAX + 64];
    const char *found = NULL;
    size_t klen = strlen(key);
    while (!found && fgets(line, sizeof line, f)) {
        char *p = line;
        while (*p == ' ' || *p == '\t')
            p++;
        if (*p == '#' || strncmp(p, key, klen))
            continue;
        p += klen;
        while (*p == ' ' || *p == '\t')
            p++;
        if (*p++ != '=')
            continue;
        while (*p == ' ' || *p == '\t')
            p++;
        p[strcspn(p, " \t\r\n")] = 0;
        snprintf(buf, size, "%s", p);
        found = buf;
    }
    fclose(f);
    return found;
}

static json_t *bind_mount(const char *src, const char *dst, const char *mode)
{
    json_t *m = json_new(JSON_OBJECT), *opts = json_new(JSON_ARRAY);
    json_push(m, "destination", json_string(dst));
    json_push(m, "type", json_string("bind"));
    json_push(m, "source", json_string(src));
    json_push(opts, NULL, json_string("rbind"));
    json_push(opts, NULL, json_string(mode));
    json_push(opts, NULL, json_string("nosuid"));
    json_push(m, "options", opts);
    return m;
}

static int listed(const char *list, const char *name)
{
    size_t len = strlen(name);
    for (const char *p = list; p && *p;) {
        const char *e = strchr(p, ',');
        size_t n = e ? (size_t)(e - p) : strlen(p);
        if (n == len && !strncmp(p, name, len))
            return 1;
        p = e ? e + 1 : NULL;
    }
    return 0;
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

static void apply_cdi(json_t *config, const char *devices)
{
    const char *path = env_or("FAKEGPU_CDI_SPEC", "/etc/cdi/fakegpu.json");
    errno = 0;
    json_t *spec = json_load(path);
    if (!spec)
        die("cannot read the CDI specification", path);
    json_t *common = json_get(spec, "containerEdits");
    json_t *mounts = json_child(config, "mounts", JSON_ARRAY);
    /* Rootless crun cannot create device nodes: bind them, as it does itself. */
    const char *nodes[512];
    int nn = 0;
    json_t *dn = json_get(common, "deviceNodes");
    for (size_t i = 0; dn && i < dn->n && nn < 512; i++)
        if (json_str(json_get(dn->items[i], "path")))
            nodes[nn++] = json_str(json_get(dn->items[i], "path"));
    json_t *devs = json_get(spec, "devices");
    for (size_t i = 0; devs && i < devs->n; i++) {
        const char *name = json_str(json_get(devs->items[i], "name"));
        if (!name || !((!strcmp(devices, "all") && is_index(name)) || listed(devices, name)))
            continue;
        json_t *d = json_get(json_get(devs->items[i], "containerEdits"), "deviceNodes");
        for (size_t k = 0; d && k < d->n && nn < 512; k++) {
            const char *node = json_str(json_get(d->items[k], "path"));
            int seen = 0;
            for (int j = 0; j < nn && node; j++)
                seen |= !strcmp(nodes[j], node);
            if (node && !seen)
                nodes[nn++] = node;
        }
    }
    for (int i = 0; i < nn; i++)
        json_push(mounts, NULL, bind_mount(nodes[i], nodes[i], "rw"));
    json_t *ms = json_get(common, "mounts");
    for (size_t i = 0; ms && i < ms->n; i++) {
        const char *host = json_str(json_get(ms->items[i], "hostPath"));
        const char *cont = json_str(json_get(ms->items[i], "containerPath"));
        json_t *opts = json_get(ms->items[i], "options");
        int ro = 0;
        for (size_t k = 0; opts && k < opts->n; k++)
            ro |= json_str(opts->items[k]) && !strcmp(json_str(opts->items[k]), "ro");
        if (host && cont)
            json_push(mounts, NULL, bind_mount(host, cont, ro ? "ro" : "rw"));
    }
    static const char *domain_keys[] = {"nccl_dir", "fabric_dir"};
    for (size_t i = 0; i < 2; i++) {
        char buf[PATH_MAX];
        struct stat st;
        const char *dir = fakegpu_conf(domain_keys[i], buf, sizeof buf);
        if (dir && *dir && stat(dir, &st) == 0 && S_ISDIR(st.st_mode))
            json_push(mounts, NULL, bind_mount(dir, dir, "rw"));
    }
    /* The spec's environment replaces the container's for the same names. */
    json_t *process = json_child(config, "process", JSON_OBJECT);
    json_t *env = json_child(process, "env", JSON_ARRAY), *add = json_get(common, "env");
    json_t *merged = json_new(JSON_ARRAY);
    for (size_t i = 0; i < env->n; i++) {
        const char *e = json_str(env->items[i]);
        int replaced = 0;
        for (size_t k = 0; e && add && k < add->n; k++) {
            const char *a = json_str(add->items[k]);
            size_t len = a ? strcspn(a, "=") : 0;
            replaced |= a && !strncmp(e, a, len) && (e[len] == '=' || !e[len]);
        }
        if (!replaced)
            json_push(merged, NULL, env->items[i]);
    }
    for (size_t k = 0; add && k < add->n; k++)
        json_push(merged, NULL, add->items[k]);
    for (size_t i = 0; i < process->n; i++)
        if (!strcmp(process->keys[i], "env"))
            process->items[i] = merged;
}

int main(int argc, char **argv)
{
    const char *runtime = env_or("FAKEGPU_OCI_RUNTIME", "/usr/bin/crun");
    int creates = 0;
    for (int i = 1; i < argc; i++)
        creates |= !strcmp(argv[i], "create") || !strcmp(argv[i], "run");
    if (creates) {
        char path[PATH_MAX], tmp[PATH_MAX + 16];
        snprintf(path, sizeof path, "%s/config.json", bundle_dir(argc - 1, argv + 1));
        errno = 0;
        json_t *config = json_load(path);
        if (!config)
            die("cannot read", path);
        json_t *env = json_get(json_get(config, "process"), "env");
        const char *devices = env_value(env, "NVIDIA_VISIBLE_DEVICES");
        if (!devices)
            devices = env_value(env, "SLURM_STEP_GPUS");
        if (devices && *devices && strcmp(devices, "void") && strcmp(devices, "none")) {
            char *wanted = strdup(devices);
            apply_cdi(config, wanted);
            snprintf(tmp, sizeof tmp, "%s.fakegpu", path);
            FILE *f = fopen(tmp, "w");
            if (!f)
                die("cannot write", tmp);
            json_write(f, config);
            if (fclose(f) != 0 || rename(tmp, path) != 0)
                die("cannot write", path);
        }
    }
    argv[0] = (char *)runtime;
    execv(runtime, argv);
    fprintf(stderr, "nvidia-container-runtime: %s: %s\n", runtime, strerror(errno));
    return 127;
}
