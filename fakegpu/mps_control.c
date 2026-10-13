/*
 * nvidia-cuda-mps-control for the fake driver: the MPS control daemon's
 * interface without MPS servers (the fake driver runs every process's work
 * itself, and enforces a fraction's memory limit in each process, see
 * fg_mem_limit). Enough for NVIDIA's GPU fractioning, whose mpsd supervises it.
 *
 *   nvidia-cuda-mps-control -f [-p N] [-m] [-a config]   daemon, foreground
 *   nvidia-cuda-mps-control -d [...]                     daemon, background
 *   echo <command> | nvidia-cuda-mps-control             client
 *
 * The daemon listens on $CUDA_MPS_PIPE_DIRECTORY/control (/tmp/nvidia-mps)
 * and logs to $CUDA_MPS_LOG_DIRECTORY/control.log (/var/log/nvidia-mps).
 * Commands come from clients and, in the foreground, from stdin: "quit"
 * stops it, "get_server_list" lists no servers, others are acknowledged.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <poll.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

static FILE *log_file;
static char sock_path[sizeof(((struct sockaddr_un *)0)->sun_path)];

static void logf_(const char *fmt, ...)
{
    if (!log_file) return;
    time_t t = time(NULL);
    char when[32];
    strftime(when, sizeof when, "%Y-%m-%d %H:%M:%S", localtime(&t));
    fprintf(log_file, "[%s] ", when);
    va_list ap;
    va_start(ap, fmt);
    vfprintf(log_file, fmt, ap);
    va_end(ap);
    fputc('\n', log_file);
    fflush(log_file);
}

static const char *env_or(const char *name, const char *dflt)
{
    const char *v = getenv(name);
    return v && *v ? v : dflt;
}

/* One command line; returns 1 for quit. Every other command gets an empty
 * answer: there are no MPS servers or clients to list (the fake driver runs
 * every client itself). */
static int command(const char *line, char *reply, size_t n)
{
    char cmd[256];
    if (n) reply[0] = 0;
    if (sscanf(line, "%255s", cmd) != 1) return 0;
    logf_("Accepting command: %s", line);
    return !strcmp(cmd, "quit");
}

static void cleanup(void)
{
    if (sock_path[0]) unlink(sock_path);
}

static void on_signal(int sig)
{
    (void)sig;
    cleanup();
    _exit(0);
}

static int daemon_main(int foreground)
{
    const char *pipe_dir = env_or("CUDA_MPS_PIPE_DIRECTORY", "/tmp/nvidia-mps");
    const char *log_dir = env_or("CUDA_MPS_LOG_DIRECTORY", "/var/log/nvidia-mps");
    mkdir(pipe_dir, 0755);
    mkdir(log_dir, 0755);
    char path[4096];
    snprintf(path, sizeof path, "%s/control.log", log_dir);
    log_file = fopen(path, "a");

    int s = socket(AF_UNIX, SOCK_STREAM, 0);
    struct sockaddr_un addr = {.sun_family = AF_UNIX};
    snprintf(addr.sun_path, sizeof addr.sun_path, "%s/control", pipe_dir);
    unlink(addr.sun_path);
    if (s < 0 || bind(s, (struct sockaddr *)&addr, sizeof addr) || listen(s, 16)) {
        fprintf(stderr, "nvidia-cuda-mps-control: %s: %s\n", addr.sun_path, strerror(errno));
        return 1;
    }
    strcpy(sock_path, addr.sun_path);
    atexit(cleanup);
    signal(SIGTERM, on_signal);
    signal(SIGINT, on_signal);
    signal(SIGPIPE, SIG_IGN);
    if (!foreground && daemon(0, 0)) return 1;
    logf_("Starting control daemon on socket %s", sock_path);

    struct pollfd fds[2] = {{.fd = s, .events = POLLIN}, {.fd = STDIN_FILENO, .events = POLLIN}};
    int nfds = foreground ? 2 : 1;
    char line[1024], reply[1024];
    for (;;) {
        if (poll(fds, nfds, -1) < 0) {
            if (errno == EINTR) continue;
            return 1;
        }
        if (nfds == 2 && fds[1].revents) {
            ssize_t r = read(STDIN_FILENO, line, sizeof line - 1);
            if (r <= 0) nfds = 1; /* stdin closed: keep serving the socket */
            else {
                line[r] = 0;
                for (char *l = strtok(line, "\n"); l; l = strtok(NULL, "\n"))
                    if (command(l, reply, sizeof reply)) {
                        logf_("Exiting");
                        return 0;
                    }
            }
        }
        if (fds[0].revents & POLLIN) {
            int c = accept(s, NULL, NULL);
            if (c < 0) continue;
            ssize_t r = read(c, line, sizeof line - 1);
            int quit = 0;
            if (r > 0) {
                line[r] = 0;
                line[strcspn(line, "\n")] = 0;
                quit = command(line, reply, sizeof reply);
                if (write(c, reply, strlen(reply)) < 0) { /* client gone */ }
            }
            close(c);
            if (quit) {
                logf_("Exiting");
                return 0;
            }
        }
    }
}

static int client_main(void)
{
    const char *pipe_dir = env_or("CUDA_MPS_PIPE_DIRECTORY", "/tmp/nvidia-mps");
    char line[1024];
    while (fgets(line, sizeof line, stdin)) {
        int s = socket(AF_UNIX, SOCK_STREAM, 0);
        struct sockaddr_un addr = {.sun_family = AF_UNIX};
        snprintf(addr.sun_path, sizeof addr.sun_path, "%s/control", pipe_dir);
        if (s < 0 || connect(s, (struct sockaddr *)&addr, sizeof addr)) {
            fprintf(stderr, "Cannot find MPS control daemon process\n");
            return 1;
        }
        if (write(s, line, strlen(line)) < 0) return 1;
        shutdown(s, SHUT_WR);
        char buf[1024];
        ssize_t r;
        while ((r = read(s, buf, sizeof buf)) > 0) fwrite(buf, 1, (size_t)r, stdout);
        close(s);
    }
    return 0;
}

int main(int argc, char **argv)
{
    int daemon_mode = 0, foreground = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-d")) daemon_mode = 1;
        else if (!strcmp(argv[i], "-f")) daemon_mode = foreground = 1;
        else if ((!strcmp(argv[i], "-p") || !strcmp(argv[i], "-a")) && i + 1 < argc) i++; /* protocol, config */
        else if (!strcmp(argv[i], "-m")) { /* multi-user: one daemon for every user anyway */ }
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            puts("Usage: nvidia-cuda-mps-control [-d | -f] [-p protocol] [-m] [-a config]");
            return 0;
        }
    }
    return daemon_mode ? daemon_main(foreground) : client_main();
}
