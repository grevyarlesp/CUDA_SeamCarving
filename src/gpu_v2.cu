#include "gpu_utils.h"
#include "gpu_v1.h"
#include "gpu_v2.h"
#include <algorithm>
#include <iostream>

using std::cerr;

__global__ void V2_grayscale_kernel(unsigned char *d_in, int num_pixels,
                                    int *out) {

  int pos = blockDim.x * blockIdx.x + threadIdx.x;
  if (pos >= num_pixels) return;

  int pos_ = pos * 3;
  int ans = (d_in[pos_] * 3 + d_in[pos_ + 1] * 6 + d_in[pos_ + 2]) / 10;

  out[pos] = ans;
}


// overlapping convolution
double V2_dp_seam() {
  GpuTimer timer;
  timer.Start();


  timer.Stop();
  return timer.Elapsed();
}
