// W2: fixed-work GPU load for week 2 (E5, E7, E8).
//
// Each kernel runs a fixed number of dependent integer steps per thread and
// keeps its state in GPU memory. Work is counted in completed kernels, so a
// kernel that is paused or preempted part-way still counts as exactly one
// unit (unlike a clock-based spin). The final state depends only on how many
// kernels ran, so it doubles as a correctness check for E8.
//
// Modes
//   time mode (default): run for --seconds, count kernels after --warmup
//   count mode:          --kernels N  run exactly N kernels, print checksum
//   launch bench:        --launch-bench  host cost of a launch and a round trip
//
// Options
//   --iters N        steps per thread per kernel (sets kernel length)
//   --graph G        launch work as CUDA graphs of G kernels each
//   --sync-every N   launch N kernels then wait for the GPU (gaps like a
//                    framework that synchronises); default 0 = never idle
//   --start-at T     wall-clock start time shared by several apps
//   --label S        name printed in the output

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <algorithm>
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

__global__ void work(uint32_t *state, int iters) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t x = state[i];
  for (int k = 0; k < iters; k++) {
    x = x * 1664525u + 1013904223u;
    x ^= x >> 13;
  }
  state[i] = x;
}

__global__ void empty_kernel() {}

static double now_s(void) {
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
  double seconds = 20, warmup = 4, start_at = 0;
  int iters = 4000, graph = 0, sync_every = 0, launch_bench = 0;
  long long kernels = 0;
  const char *label = "w2";
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--seconds")) seconds = atof(argv[++i]);
    else if (!strcmp(argv[i], "--warmup")) warmup = atof(argv[++i]);
    else if (!strcmp(argv[i], "--start-at")) start_at = atof(argv[++i]);
    else if (!strcmp(argv[i], "--iters")) iters = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--graph")) graph = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--sync-every")) sync_every = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--kernels")) kernels = atoll(argv[++i]);
    else if (!strcmp(argv[i], "--launch-bench")) launch_bench = 1;
    else if (!strcmp(argv[i], "--label")) label = argv[++i];
  }

  cudaDeviceProp prop;
  CHECK(cudaGetDeviceProperties(&prop, 0));
  int blocks = prop.multiProcessorCount * 4, threads = 256, n = blocks * threads;
  uint32_t *state;
  CHECK(cudaMalloc(&state, n * sizeof(uint32_t)));
  std::vector<uint32_t> init(n);
  for (int i = 0; i < n; i++) init[i] = (uint32_t)i * 2654435761u;
  CHECK(cudaMemcpy(state, init.data(), n * sizeof(uint32_t), cudaMemcpyHostToDevice));
  cudaStream_t s;
  CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));

  if (launch_bench) {
    // Host cost of one async launch, and of a launch + wait round trip.
    empty_kernel<<<1, 1, 0, s>>>();
    CHECK(cudaStreamSynchronize(s));
    const int L = 20000, R = 5000;
    double t0 = now_s();
    for (int k = 0; k < L; k++) empty_kernel<<<1, 1, 0, s>>>();
    double t1 = now_s();
    CHECK(cudaStreamSynchronize(s));
    std::vector<double> rt(R);
    for (int k = 0; k < R; k++) {
      double a = now_s();
      empty_kernel<<<1, 1, 0, s>>>();
      CHECK(cudaStreamSynchronize(s));
      rt[k] = (now_s() - a) * 1e6;
    }
    std::sort(rt.begin(), rt.end());
    printf("label=%s launch_ns=%.1f roundtrip_p50_us=%.2f roundtrip_p99_us=%.2f\n", label,
           (t1 - t0) / L * 1e9, rt[R / 2], rt[(int)(R * 0.99)]);
    return 0;
  }

  cudaGraphExec_t gexec = NULL;
  if (graph > 0) {
    cudaGraph_t g;
    CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    for (int k = 0; k < graph; k++) work<<<blocks, threads, 0, s>>>(state, iters);
    CHECK(cudaStreamEndCapture(s, &g));
    CHECK(cudaGraphInstantiate(&gexec, g, 0));
  }
  // One "unit" = one launch: a kernel, or a whole graph of `graph` kernels.
  int per_unit = graph > 0 ? graph : 1;
  auto launch_unit = [&]() {
    if (gexec) CHECK(cudaGraphLaunch(gexec, s));
    else work<<<blocks, threads, 0, s>>>(state, iters);
  };

  work<<<blocks, threads, 0, s>>>(state, 1);  // context warm-up (counted below)
  CHECK(cudaStreamSynchronize(s));
  long long done = 1;
  if (start_at > 0)
    while (now_s() < start_at) usleep(500);

  enum { DEPTH = 4 };
  cudaEvent_t ev[DEPTH];
  for (int i = 0; i < DEPTH; i++) CHECK(cudaEventCreateWithFlags(&ev[i], cudaEventDisableTiming));

  double t0 = now_s();
  double tm0 = t0 + warmup, tend = t0 + seconds;
  long long counted = 0, units_left = kernels > 0 ? (kernels - 1 + per_unit - 1) / per_unit : -1;
  int inflight = 0, head = 0;

  while (1) {
    if (kernels > 0) {
      if (units_left == 0) break;
    } else if (now_s() >= tend) {
      break;
    }
    if (sync_every > 0) {
      for (int k = 0; k < sync_every && units_left != 0; k++) {
        launch_unit();
        if (units_left > 0) units_left--;
      }
      CHECK(cudaStreamSynchronize(s));
      double t = now_s();
      done += (long long)sync_every * per_unit;
      if (t >= tm0 && t <= tend) counted += (long long)sync_every * per_unit;
      continue;
    }
    if (inflight == DEPTH) {
      CHECK(cudaEventSynchronize(ev[head]));
      double t = now_s();
      done += per_unit;
      if (t >= tm0 && t <= tend) counted += per_unit;
      inflight--;
    }
    launch_unit();
    if (units_left > 0) units_left--;
    CHECK(cudaEventRecord(ev[head], s));
    head = (head + 1) % DEPTH;
    inflight++;
  }
  CHECK(cudaStreamSynchronize(s));
  double elapsed = now_s() - t0;

  if (kernels > 0) {
    std::vector<uint32_t> out(n);
    CHECK(cudaMemcpy(out.data(), state, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    uint64_t sum = 0;
    for (int i = 0; i < n; i++) sum = sum * 1099511628211ull + out[i];
    printf("label=%s pid=%d kernels=%lld iters=%d elapsed_s=%.3f checksum=%016llx\n", label,
           getpid(), kernels, iters, elapsed, (unsigned long long)sum);
  } else {
    double window = seconds - warmup;
    printf("label=%s pid=%d iters=%d graph=%d sync_every=%d window_s=%.1f kernels=%lld kps=%.2f\n",
           label, getpid(), iters, graph, sync_every, window, counted, counted / window);
  }
  return 0;
}
