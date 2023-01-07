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
  if (pos >= num_pixels)
    return;

  int pos_ = pos * 3;
  int ans = (d_in[pos_] * 3 + d_in[pos_ + 1] * 6 + d_in[pos_ + 2]) / 10;

  out[pos] = ans;
}

/*
   Manual merge, Parallelized  DP.
   Split into n_Stream parts
   */

__global__ void V2_conv_kernel(int *d_in, int height, int width, int *d_out) {}

__device__ int bCount;
__device__ int done;
__device__ int *completed;

/*
   We divide the image into "strips",

   Block size 32 required.
   */

__global__ void V2_dp_kernel(int *d_in, int height, int width, int *d_out,
                             volatile int *d_dp, int *d_trace) {
  __shared__ int bi;

  if (threadIdx.x == 0) {
    bi = atomicAdd(&bCount, 1);
  }

  __syncthreads();

  int tidx = threadIdx.x;

  int threads_per_row = gridDim.x * blockDim.x;

  // dynamic block id
  int blockId_x = bi / gridDim.x;
  // also strip Idx
  int blockId_y = bi % gridDim.x;

  int row = blockId_y * blockDim.y + threadIdx.y;
  int col = blockId_x * blockDim.x + threadIdx.x;

  if (row >= height || col >= width) {
    return;
  }

  int pos = row * width + col;

  // first row of the block
  if (threadIdx.y == 0) {
    d_dp[pos] = d_in[pos];
    __threadfence();
    atomicAdd(&completed[threadIdx.y], 1);
  } else {
    // calculate required number of threads
    // all threads of previous rows in a strip
    int required = (threadIdx.y - 1) * width;

    // wait for the required number of threads to complete
    while (atomicAdd(&completed[threadIdx.y], 0) < required) {
      ;
    }

    int ans = -1;

    int left = col - 1;
    if (left >= 0) {
      ans = d_dp[(row - 1) * width + left];
      d_trace[pos] = left;
    }

    int middle = col;
    if (ans == -1 || ans > d_dp[(row - 1) * width + middle]) {
      ans = d_dp[(row - 1) * width + middle];
      d_trace[pos] = middle;
    }

    int right = col + 1;
    if (ans == -1 || ans > d_dp[(row - 1) * width + right]) {
      ans = d_dp[(row - 1) * width + right];
      d_trace[pos] = right;
    }

    d_dp[pos] = ans + d_dp[row * width + col];
  }

  __syncthreads();

  // merge results.
  if (blockId_y > 0 && threadIdx.y == 0) {

    int required = blockDim.y * width;

    // wait for the required number of threads to complete
    // this may not be worth it.
    // prob better just to merge on host
    while (atomicAdd(&completed[threadIdx.y - 1], 0) < required) {
      ;
    }
    int ans = -1;

    int left = col - 1;
    if (left >= 0) {
      ans = d_dp[(row - 1) * width + left];
      d_trace[pos] = left;
    }

    int middle = col;
    if (ans == -1 || ans > d_dp[(row - 1) * width + middle]) {
      ans = d_dp[(row - 1) * width + middle];
      d_trace[pos] = middle;
    }

    int right = col + 1;
    if (ans == -1 || ans > d_dp[(row - 1) * width + right]) {
      ans = d_dp[(row - 1) * width + right];
      d_trace[pos] = right;
    }

    d_dp[pos] = ans + d_dp[row * width + col];
  }
}

double V2_seam(int *in, int height, int width, int *out, int blocksize) {
}


// tracing so we don't have to copy
__global__ void trace_kernel(int *d_trace) {}

// overlapping convolution
double V2_dp_seam() {
  GpuTimer timer;
  timer.Start();

  timer.Stop();
  return timer.Elapsed();
}
