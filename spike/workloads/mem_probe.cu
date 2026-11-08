// Memory probe for the HAMi-core + DynaGPU coexistence test.
// Prints the GPU memory the app is told it has (cudaMemGetInfo), then
// allocates 1 GiB blocks until an allocation fails, and prints how many fit.
#include <cuda_runtime.h>
#include <stdio.h>

int main(void) {
  size_t free_b = 0, total_b = 0;
  cudaError_t e = cudaMemGetInfo(&free_b, &total_b);
  if (e != cudaSuccess) {
    printf("meminfo_error=%s\n", cudaGetErrorName(e));
    return 1;
  }
  int n = 0;
  void *p;
  for (; n < 256; n++) {
    e = cudaMalloc(&p, 1ull << 30);
    if (e != cudaSuccess) break;
  }
  printf("reported_total_mib=%zu reported_free_mib=%zu allocated_gib=%d stop_reason=%s\n",
         total_b >> 20, free_b >> 20, n, n < 256 ? cudaGetErrorName(e) : "limit_of_probe");
  return 0;
}
