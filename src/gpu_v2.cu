#include "gpu_utils.h"
#include "gpu_v1.h"
#include "gpu_v2.h"
#include <__clang_cuda_builtin_vars.h>
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


/*
   Manual merge, Parallelized  DP.
   Split into 4 parts
   */

__global__ void V2_conv_kernel(int *d_in, int height, int width, int *d_out) {

}

__device__ int bCount;
__device__ int done;

__global__ void V2_dp_kernel(int *d_in, int height, int width, int *d_out, int *d_dp) {
  __shared__ int bi;

  if (threadIdx.x == 0) {
    bi = atomicAdd(&bCount, 1);
  }

  __syncthreads();

  int tidx = threadIdx.x;
  

  int cnt = gridDim.x * blockDim.x;
  int row = bi / cnt;
  int col = bi % cnt;

  if (row >= height || col >= width) {
    return;
  }

  // first row of the block
  if (threadIdx.y == 0) {
    
    
  }

  // wait for the required number of threads to complete 

  // merge results.


}

// tracing so we don't have to copy
__global__ void trace_kernel(int *d_trace) {
  
}



// overlapping convolution
double V2_dp_seam() {
  GpuTimer timer;
  timer.Start();




  timer.Stop();
  return timer.Elapsed();
}
