/*
 * nvidia-imex and nvidia-imex-ctl for the fake driver: the IMEX daemon's
 * interface, for NVIDIA's DRA driver's ComputeDomains, which run one daemon
 * per node of a compute domain and probe it with nvidia-imex-ctl -q.
 *
 * The real daemon exports and imports GPU memory between the nodes of an
 * NVLink domain. The fake driver shares fabric memory through files on
 * /shared (vmm.c), so this one only does what is visible from outside:
 *
 *   nvidia-imex -c <config>         daemon (foreground, DAEMONIZE=0)
 *     - listens on SERVER_PORT for its peers and on IMEX_CMD_BIND_INTERFACE_IP
 *       :IMEX_CMD_PORT for nvidia-imex-ctl;
 *     - reads the peers from IMEX_NODE_CONFIG_FILE (one IP or name per line)
 *       and connects to each every 2 s; names resolve through /etc/hosts
 *       (where NVIDIA's compute-domain daemon puts them; 127.0.0.2 means an
 *       empty slot), so the binary is static and runs in any image;
 *     - SIGUSR1 rereads the node list, SIGTERM and SIGINT stop it.
 *   nvidia-imex-ctl -c <config> -q  READY once the local daemon serves
 *   nvidia-imex-ctl -c <config> -N  the domain's nodes and their connection
 *
 * Built twice from this file: nvidia-imex-ctl with -DIMEX_CTL.
 */
#define _GNU_SOURCE
#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#define MAX_NODES 128

struct config {
    int server_port, cmd_port;
    char cmd_ip[64], bind_ip[64], nodes_file[512];
};

static void trim(char *s)
{
    char *e = s + strlen(s);
    while (e > s && (e[-1] == '\n' || e[-1] == '\r' || e[-1] == ' ' || e[-1] == '\t'))
        *--e = 0;
}

static int read_config(const char *path, struct config *c)
{
    memset(c, 0, sizeof(*c));
    c->server_port = 50000;
    c->cmd_port = 50005;
    FILE *f = fopen(path, "r");
    if (!f)
        return -1;
    char line[1024];
    while (fgets(line, sizeof(line), f)) {
        trim(line);
        char *eq = strchr(line, '=');
        if (line[0] == '#' || !eq)
            continue;
        *eq = 0;
        const char *k = line, *v = eq + 1;
        if (!strcmp(k, "SERVER_PORT") && *v)
            c->server_port = atoi(v);
        else if (!strcmp(k, "IMEX_CMD_PORT") && *v)
            c->cmd_port = atoi(v);
        else if (!strcmp(k, "IMEX_CMD_BIND_INTERFACE_IP"))
            snprintf(c->cmd_ip, sizeof(c->cmd_ip), "%s", v);
        else if (!strcmp(k, "BIND_INTERFACE_IP"))
            snprintf(c->bind_ip, sizeof(c->bind_ip), "%s", v);
        else if (!strcmp(k, "IMEX_NODE_CONFIG_FILE"))
            snprintf(c->nodes_file, sizeof(c->nodes_file), "%s", v);
    }
    fclose(f);
    if (!c->cmd_ip[0])
        snprintf(c->cmd_ip, sizeof(c->cmd_ip), "127.0.0.1");
    return 0;
}

/* An IP as is, a name through /etc/hosts. 0 on success. */
static __attribute__((unused)) int resolve(const char *name, char *ip, size_t len)
{
    struct in_addr a;
    if (inet_pton(AF_INET, name, &a) == 1) {
        snprintf(ip, len, "%.63s", name);
        return 0;
    }
    FILE *f = fopen("/etc/hosts", "r");
    if (!f)
        return -1;
    char line[1024];
    int found = -1;
    while (found && fgets(line, sizeof(line), f)) {
        char *save, *addr = strtok_r(line, " \t\n", &save);
        if (!addr || addr[0] == '#')
            continue;
        for (char *n; (n = strtok_r(NULL, " \t\n", &save));)
            if (!strcmp(n, name)) {
                snprintf(ip, len, "%.63s", addr);
                found = 0;
                break;
            }
    }
    fclose(f);
    return found;
}

static int tcp_connect(const char *ip, int port, int timeout_ms)
{
    struct sockaddr_in sa = {.sin_family = AF_INET, .sin_port = htons(port)};
    if (inet_pton(AF_INET, ip, &sa.sin_addr) != 1)
        return -1;
    int fd = socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
    if (fd < 0)
        return -1;
    if (connect(fd, (struct sockaddr *)&sa, sizeof(sa)) < 0 && errno != EINPROGRESS) {
        close(fd);
        return -1;
    }
    struct pollfd p = {.fd = fd, .events = POLLOUT};
    int err = 0;
    socklen_t elen = sizeof(err);
    if (poll(&p, 1, timeout_ms) != 1 || getsockopt(fd, SOL_SOCKET, SO_ERROR, &err, &elen) < 0 || err) {
        close(fd);
        return -1;
    }
    fcntl(fd, F_SETFL, fcntl(fd, F_GETFL) & ~O_NONBLOCK);
    return fd;
}

#ifdef IMEX_CTL

static void usage(void)
{
    fprintf(stderr, "usage: nvidia-imex-ctl [-c <config>] -q | -N\n");
    exit(2);
}

int main(int argc, char **argv)
{
    const char *cfg = "/etc/nvidia-imex/config.cfg";
    char cmd = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-c") && i + 1 < argc)
            cfg = argv[++i];
        else if (!strcmp(argv[i], "-q"))
            cmd = 'q';
        else if (!strcmp(argv[i], "-N"))
            cmd = 'N';
        else if (!strncmp(argv[i], "--u=", 4)) {
            fprintf(stderr, "nvidia-imex-ctl: Unix domain sockets are not supported by the lab's IMEX daemon\n");
            return 1;
        } else
            usage();
    }
    if (!cmd)
        usage();
    struct config c;
    if (read_config(cfg, &c) < 0) {
        fprintf(stderr, "nvidia-imex-ctl: cannot read %s: %s\n", cfg, strerror(errno));
        return 1;
    }
    int fd = tcp_connect(c.cmd_ip, c.cmd_port, 2000);
    if (fd < 0) {
        fprintf(stderr, "nvidia-imex-ctl: cannot connect to the IMEX daemon at %s:%d\n", c.cmd_ip, c.cmd_port);
        return 1;
    }
    char req[2] = {cmd, '\n'};
    if (write(fd, req, 2) != 2) {
        close(fd);
        return 1;
    }
    char buf[8192];
    ssize_t n;
    while ((n = read(fd, buf, sizeof(buf))) > 0)
        fwrite(buf, 1, n, stdout);
    close(fd);
    return 0;
}

#else

struct node {
    char name[512], ip[64];
    int connected, ever;
};

static struct config conf;
static struct node nodes[MAX_NODES];
static int n_nodes;
static volatile sig_atomic_t reload = 1, stop;
static char my_ip[64];

static void on_signal(int s)
{
    if (s == SIGUSR1)
        reload = 1;
    else
        stop = 1;
}

static void logmsg(const char *fmt, const char *a)
{
    time_t t = time(NULL);
    char ts[32];
    strftime(ts, sizeof(ts), "%m/%d/%Y %H:%M:%S", localtime(&t));
    printf("[%s] [INFO] [tid %d] ", ts, getpid());
    printf(fmt, a);
    printf("\n");
    fflush(stdout);
}

/* The node list: the config file's entries, resolved now (names move as
 * daemons come and go). Empty slots (127.0.0.2) and unresolved names are
 * not part of the domain yet. */
static void load_nodes(void)
{
    struct node old[MAX_NODES];
    int n_old = n_nodes;
    memcpy(old, nodes, sizeof(old));
    n_nodes = 0;
    FILE *f = conf.nodes_file[0] ? fopen(conf.nodes_file, "r") : NULL;
    if (!f)
        return;
    char line[512];
    while (n_nodes < MAX_NODES && fgets(line, sizeof(line), f)) {
        trim(line);
        if (!line[0] || line[0] == '#')
            continue;
        struct node *nd = &nodes[n_nodes];
        memset(nd, 0, sizeof(*nd));
        snprintf(nd->name, sizeof(nd->name), "%s", line);
        if (resolve(line, nd->ip, sizeof(nd->ip)) < 0 || !strcmp(nd->ip, "127.0.0.2"))
            continue;
        for (int i = 0; i < n_old; i++)
            if (!strcmp(old[i].ip, nd->ip))
                nd->ever = old[i].ever, nd->connected = old[i].connected;
        n_nodes++;
    }
    fclose(f);
}

static void probe_peers(void)
{
    for (int i = 0; i < n_nodes; i++) {
        struct node *nd = &nodes[i];
        int was = nd->connected;
        if (!strcmp(nd->ip, my_ip)) {
            nd->connected = nd->ever = 1;
            continue;
        }
        int fd = tcp_connect(nd->ip, conf.server_port, 500);
        nd->connected = fd >= 0;
        if (fd >= 0) {
            nd->ever = 1;
            close(fd);
        }
        if (nd->connected != was)
            logmsg(nd->connected ? "Connection established to node %s" : "Connection lost to node %s", nd->ip);
    }
}

static int listen_on(const char *ip, int port)
{
    struct sockaddr_in sa = {.sin_family = AF_INET, .sin_port = htons(port)};
    if (!ip || !*ip || inet_pton(AF_INET, ip, &sa.sin_addr) != 1)
        sa.sin_addr.s_addr = htonl(INADDR_ANY);
    int fd = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0), one = 1;
    if (fd < 0)
        return -1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    if (bind(fd, (struct sockaddr *)&sa, sizeof(sa)) < 0 || listen(fd, 16) < 0) {
        close(fd);
        return -1;
    }
    return fd;
}

static void answer(int fd)
{
    char cmd = 0;
    struct pollfd p = {.fd = fd, .events = POLLIN};
    if (poll(&p, 1, 1000) == 1 && read(fd, &cmd, 1) != 1)
        cmd = 0;
    char out[16384];
    size_t len = 0;
#define OUT(...) (len += snprintf(out + len, sizeof(out) - len, __VA_ARGS__), len = len < sizeof(out) ? len : sizeof(out) - 1)
    if (cmd == 'q') {
        OUT("READY\n");
    } else if (cmd == 'N') {
        int up = 1;
        time_t t = time(NULL);
        char ts[32];
        strftime(ts, sizeof(ts), "%m/%d/%Y %H:%M:%S", localtime(&t));
        OUT("Connectivity Table Legend:\n"
            "N - Never Connected\n"
            "D - Disconnected - Connection was lost\n"
            "C - Connected - Ready for operation\n\n%s\nNodes:\n", ts);
        for (int i = 0; i < n_nodes; i++) {
            OUT("Node #%-3d - %-15s - %-11s - %s\n", i, nodes[i].ip, nodes[i].connected ? "READY" : "UNAVAILABLE",
                nodes[i].name);
            up &= nodes[i].connected;
        }
        OUT("\n Nodes From\\To ");
        for (int i = 0; i < n_nodes; i++)
            OUT(" %-3d", i);
        OUT("\n");
        for (int i = 0; i < n_nodes; i++) {
            if (strcmp(nodes[i].ip, my_ip))
                continue;
            OUT("       %-8d", i);
            for (int j = 0; j < n_nodes; j++)
                OUT(" %-3s", nodes[j].connected ? "C" : nodes[j].ever ? "D" : "N");
            OUT("\n");
        }
        OUT("Domain State: %s\n", up && n_nodes ? "UP" : "DEGRADED");
    } else {
        OUT("UNKNOWN COMMAND\n");
    }
    if (write(fd, out, len) < 0)
        perror("write");
}

int main(int argc, char **argv)
{
    const char *cfg = "/etc/nvidia-imex/config.cfg";
    for (int i = 1; i < argc; i++)
        if (!strcmp(argv[i], "-c") && i + 1 < argc)
            cfg = argv[++i];
    if (read_config(cfg, &conf) < 0) {
        fprintf(stderr, "nvidia-imex: cannot read %s: %s\n", cfg, strerror(errno));
        return 1;
    }
    snprintf(my_ip, sizeof(my_ip), "%s", conf.bind_ip[0] ? conf.bind_ip : conf.cmd_ip);

    struct sigaction sa = {.sa_handler = on_signal};
    sigaction(SIGUSR1, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);
    sigaction(SIGINT, &sa, NULL);
    signal(SIGPIPE, SIG_IGN);

    int srv = listen_on(conf.bind_ip, conf.server_port);
    int cmd = listen_on(conf.cmd_ip, conf.cmd_port);
    if (srv < 0 || cmd < 0) {
        fprintf(stderr, "nvidia-imex: cannot listen on ports %d/%d: %s\n", conf.server_port, conf.cmd_port,
                strerror(errno));
        return 1;
    }
    logmsg("IMEX daemon started, node %s", my_ip);

    struct timespec last = {0};
    while (!stop) {
        if (reload) {
            reload = 0;
            load_nodes();
            logmsg("Node configuration loaded from %s", conf.nodes_file);
        }
        struct timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now);
        if (now.tv_sec - last.tv_sec >= 2) {
            last = now;
            load_nodes();
            probe_peers();
        }
        struct pollfd p[2] = {{.fd = srv, .events = POLLIN}, {.fd = cmd, .events = POLLIN}};
        if (poll(p, 2, 500) <= 0)
            continue;
        if (p[0].revents & POLLIN) {
            int c = accept4(srv, NULL, NULL, SOCK_CLOEXEC);
            if (c >= 0) {
                if (write(c, "nvidia-imex\n", 12) < 0)
                    perror("write");
                close(c);
            }
        }
        if (p[1].revents & POLLIN) {
            int c = accept4(cmd, NULL, NULL, SOCK_CLOEXEC);
            if (c >= 0) {
                answer(c);
                close(c);
            }
        }
    }
    logmsg("IMEX daemon stopping%s", "");
    return 0;
}

#endif
