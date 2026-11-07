// W1: keep the GPU busy and count how much work gets done.
//
// Each kernel spins for a fixed number of cycles on every SM, so one kernel
// is a fixed unit of work. Kernels are kept queued back to back (no idle
// gaps), so the process never voluntarily gives up its turn on the GPU.
//
// Usage:
//   w1_busy --seconds 30 --warmup 5 [--start-at <unix-epoch-seconds>]
//           [--spin-cycles 200000] [--idle-ms 0] [--burst-ms 0]
//
// --idle-ms / --burst-ms make a mostly idle app for E4: run kernels for
// burst-ms, then sleep idle-ms, repeat.
//
// Output (one line, key=value): kernels counted in the measured window
// [warmup, seconds] and kernels per second.

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define CHECK(x)                                                              \
  do {                                                                        \
    cudaError_t e = (x);                                                      \
    if (e != cudaSuccess) {                                                   \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e),      \
              __FILE__, __LINE__);                                            \
      exit(1);                                                                \
    }                                                                         \
  } while (0)

__global__ void spin(long long cycles, int *sink) {
  long long start = clock64();
  while (clock64() - start < cycles) {
  }
  if (threadIdx.x == 0 && blockIdx.x == 0) atomicAdd(sink, 1);
}

static double now_s(void) {
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
  double seconds = 30, warmup = 5, start_at = 0;
  long long spin_cycles = 200000;  // ~0.1 ms at ~2 GHz
  int idle_ms = 0, burst_ms = 0;
  const char *label = "w1";
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--seconds")) seconds = atof(argv[++i]);
    else if (!strcmp(argv[i], "--warmup")) warmup = atof(argv[++i]);
    else if (!strcmp(argv[i], "--start-at")) start_at = atof(argv[++i]);
    else if (!strcmp(argv[i], "--spin-cycles")) spin_cycles = atoll(argv[++i]);
    else if (!strcmp(argv[i], "--idle-ms")) idle_ms = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--burst-ms")) burst_ms = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--label")) label = argv[++i];
  }

  cudaDeviceProp prop;
  CHECK(cudaGetDeviceProperties(&prop, 0));
  int blocks = prop.multiProcessorCount * 4;  // fill every SM
  int threads = 256;

  int *sink;
  CHECK(cudaMalloc(&sink, sizeof(int)));
  cudaStream_t s;
  CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));

  // Warm up the context, then wait for the common start time.
  spin<<<blocks, threads, 0, s>>>(1000, sink);
  CHECK(cudaStreamSynchronize(s));
  if (start_at > 0) {
    while (now_s() < start_at) usleep(1000);
  }

  // Keep up to DEPTH batches in flight; each batch ends with an event.
  enum { BATCH = 8, DEPTH = 4 };
  cudaEvent_t ev[DEPTH];
  double ev_t[DEPTH];
  for (int i = 0; i < DEPTH; i++) CHECK(cudaEventCreateWithFlags(&ev[i], cudaEventDisableTiming));

  double t0 = now_s();
  double t_measure_start = t0 + warmup, t_end = t0 + seconds;
  long long counted = 0, total = 0;
  int inflight = 0, head = 0;
  double burst_start = t0;

  while (1) {
    double t = now_s();
    if (t >= t_end) break;

    if (burst_ms > 0 && (t - burst_start) * 1000.0 >= burst_ms) {
      CHECK(cudaStreamSynchronize(s));
      inflight = 0;
      usleep(idle_ms * 1000);
      burst_start = now_s();
      continue;
    }

    if (inflight == DEPTH) {
      int tail = (head + DEPTH - inflight) % DEPTH;
      CHECK(cudaEventSynchronize(ev[tail]));
      double done_t = now_s();
      total += BATCH;
      if (done_t >= t_measure_start && done_t <= t_end) counted += BATCH;
      (void)ev_t;
      inflight--;
    }
    for (int k = 0; k < BATCH; k++) spin<<<blocks, threads, 0, s>>>(spin_cycles, sink);
    CHECK(cudaGetLastError());
    CHECK(cudaEventRecord(ev[head], s));
    head = (head + 1) % DEPTH;
    inflight++;
  }
  CHECK(cudaStreamSynchronize(s));

  double window = seconds - warmup;
  printf("label=%s pid=%d sms=%d window_s=%.1f kernels=%lld kps=%.1f\n", label,
         getpid(), prop.multiProcessorCount, window, counted, counted / window);
  return 0;
}
