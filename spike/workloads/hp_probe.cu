// E6: latency seen by a high-priority app that shares the GPU.
//
// Every --period-ms the probe sends one small request: a short kernel, then
// it waits for it to finish. It records the time from "request arrives" to
// "result ready". With --pause, it first tells each low-priority app's shim
// to pause (and waits for the reply), and resumes them after the result is
// ready, which is how DynaGPU's priority feature would work.
//
// Usage: hp_probe --requests 1000 --period-ms 10 [--iters 2000]
//                 [--pause /tmp/lp1.sock,/tmp/lp2.sock] [--csv out.csv]
// Output: p50/p90/p99/max of total latency, and of the pause step alone.

#include <cuda_runtime.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>
#include <algorithm>
#include <string>
#include <vector>

#define CHECK(x)                                                                    \
  do {                                                                              \
    cudaError_t e = (x);                                                            \
    if (e != cudaSuccess) {                                                         \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, \
              __LINE__);                                                            \
      exit(1);                                                                      \
    }                                                                               \
  } while (0)

__global__ void small_work(uint32_t *state, int iters) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t x = state[i];
  for (int k = 0; k < iters; k++) {
    x = x * 1664525u + 1013904223u;
    x ^= x >> 13;
  }
  state[i] = x;
}

static double mono_us(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

struct Peer {
  struct sockaddr_un addr;
};

static int sock_fd = -1;

// Send the command to every peer first, then collect all replies, so the
// peers pause in parallel.
static bool send_all(const std::vector<Peer> &peers, const char *cmd) {
  bool ok = true;
  for (auto &p : peers)
    if (sendto(sock_fd, cmd, strlen(cmd), 0, (const struct sockaddr *)&p.addr, sizeof(p.addr)) < 0)
      ok = false;
  for (size_t i = 0; i < peers.size(); i++) {
    char buf[128] = {0};
    ssize_t n = recv(sock_fd, buf, sizeof(buf) - 1, 0);
    ok &= n > 0 && !strncmp(buf, "ok", 2);
  }
  return ok;
}

static double pct(std::vector<double> v, double q) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  size_t i = (size_t)(q * (v.size() - 1) + 0.5);
  return v[i];
}

int main(int argc, char **argv) {
  int requests = 1000, iters = 2000;
  double period_ms = 10;
  std::string pause_list, csv;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--requests")) requests = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--period-ms")) period_ms = atof(argv[++i]);
    else if (!strcmp(argv[i], "--iters")) iters = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--pause")) pause_list = argv[++i];
    else if (!strcmp(argv[i], "--csv")) csv = argv[++i];
  }

  std::vector<Peer> peers;
  if (!pause_list.empty()) {
    sock_fd = socket(AF_UNIX, SOCK_DGRAM, 0);
    struct sockaddr_un me = {};
    me.sun_family = AF_UNIX;
    snprintf(me.sun_path, sizeof(me.sun_path), "/tmp/hp_probe.%d.sock", getpid());
    unlink(me.sun_path);
    if (bind(sock_fd, (struct sockaddr *)&me, sizeof(me)) != 0) { perror("bind"); return 1; }
    struct timeval tv = {0, 200000};  // 200 ms reply timeout
    setsockopt(sock_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    size_t start = 0;
    while (start <= pause_list.size()) {
      size_t c = pause_list.find(',', start);
      std::string path = pause_list.substr(start, c == std::string::npos ? std::string::npos : c - start);
      if (!path.empty()) {
        Peer p = {};
        p.addr.sun_family = AF_UNIX;
        strncpy(p.addr.sun_path, path.c_str(), sizeof(p.addr.sun_path) - 1);
        peers.push_back(p);
      }
      if (c == std::string::npos) break;
      start = c + 1;
    }
  }

  cudaDeviceProp prop;
  CHECK(cudaGetDeviceProperties(&prop, 0));
  int blocks = prop.multiProcessorCount, threads = 256;
  uint32_t *state;
  CHECK(cudaMalloc(&state, blocks * threads * sizeof(uint32_t)));
  CHECK(cudaMemset(state, 0, blocks * threads * sizeof(uint32_t)));
  cudaStream_t s;
  CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  for (int k = 0; k < 20; k++) small_work<<<blocks, threads, 0, s>>>(state, iters);
  CHECK(cudaStreamSynchronize(s));

  std::vector<double> total, pause_t, gpu_t;
  int failures = 0;
  double next = mono_us() + 200000;
  for (int r = 0; r < requests; r++) {
    while (mono_us() < next) usleep(50);
    double t0 = mono_us();
    bool ok = true;
    if (!peers.empty()) ok &= send_all(peers, "pause");
    double t1 = mono_us();
    small_work<<<blocks, threads, 0, s>>>(state, iters);
    CHECK(cudaStreamSynchronize(s));
    double t2 = mono_us();
    if (!peers.empty()) ok &= send_all(peers, "resume");
    if (!ok) failures++;
    total.push_back(t2 - t0);
    pause_t.push_back(t1 - t0);
    gpu_t.push_back(t2 - t1);
    next += period_ms * 1000;
  }

  if (!csv.empty()) {
    FILE *f = fopen(csv.c_str(), "w");
    fprintf(f, "request,total_us,pause_us,gpu_us\n");
    for (size_t i = 0; i < total.size(); i++)
      fprintf(f, "%zu,%.1f,%.1f,%.1f\n", i, total[i], pause_t[i], gpu_t[i]);
    fclose(f);
  }
  printf("requests=%d peers=%zu failures=%d p50_us=%.1f p90_us=%.1f p99_us=%.1f max_us=%.1f "
         "pause_p50_us=%.1f pause_p99_us=%.1f\n",
         requests, peers.size(), failures, pct(total, 0.5), pct(total, 0.9), pct(total, 0.99),
         pct(total, 1.0), pct(pause_t, 0.5), pct(pause_t, 0.99));
  return 0;
}
