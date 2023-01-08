#include "gpu_utils.h"
#include "gpu_v1.h"
#include "gpu_v2.h"
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
__device__ int completed_block;

/*
   Use N * 1 blocks
   */

__global__ void V2_dp_kernel(int *d_in, int height, int width,
                             volatile int *d_dp, int *d_trace) {
  __shared__ int bi;

  if (threadIdx.x == 0 && threadIdx.y == 0) {
    bi = atomicAdd(&bCount, 1);
#ifdef V2_DEBUG
    printf("Block %d\n", bi);
#endif
  }

  __syncthreads();

  int tidx = threadIdx.x;

  // dynamic block id
  int blockId_y = bi / gridDim.x;
  // also strip Idx
  int blockId_x = bi % gridDim.x;

#ifdef V2_DEBUG
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    printf("Block [%d, %d]\n", blockId_x, blockId_y);
  }
#endif

  // or blockId_y
  int row = blockId_y;
  int col = blockId_x * blockDim.x + threadIdx.x;

  if (row >= height || col >= width) {
    return;
  }

#ifdef V2_DEBUG
  printf("Row %d Col %d\n", row, col);
#endif

  int pos = row * width + col;

  // first row of the block
  if (row == 0) {

    d_dp[pos] = d_in[pos];
  } else {
    // calculate required number of threads
    // all threads of previous rows in a strip
    int required = gridDim.y * (row - 1);

    // wait for the above row to complete
    if (threadIdx.x == 0)
      while (atomicAdd(&completed_block, 0) < required) {
        ;
      }

    __syncthreads();

#ifdef V2_DEBUG
    printf("[%d, %d] doing DP\n", row, col);
#endif

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
    if (right < width && (ans == -1 || ans > d_dp[(row - 1) * width + right])) {
      ans = d_dp[(row - 1) * width + right];
      d_trace[pos] = right;
    }

    d_dp[pos] = ans + d_in[pos];
    __threadfence();
  }

  __syncthreads();

  if (threadIdx.x == 0)
    atomicAdd(&completed_block, 1);

}

// must : blocksize = 256 x 1
// out[height]
double V2_seam(int *in, int height, int width, int *out, int blocksize) {

  GpuTimer timer;

  timer.Start();

  int *d_in;
  CHECK(cudaMalloc(&d_in, height * width * sizeof(int)));
  CHECK(cudaMemcpy(d_in, in, height * width * sizeof(int),
                   cudaMemcpyHostToDevice));

  int *d_dp;
  CHECK(cudaMalloc(&d_dp, height * width * sizeof(int)));

  int *d_trace;
  CHECK(cudaMalloc(&d_trace, height * width * sizeof(int)));

  dim3 grid_size((width - 1) / blocksize + 1, (height - 1) / blocksize + 1);
  dim3 block_size(blocksize, blocksize);

  int val = 0; // because we need to start at 0
  CHECK(cudaMemcpyToSymbol(bCount, &val, sizeof(int), 0));

  CHECK(cudaMemcpyToSymbol(completed_block, &val, sizeof(int), size_t(0)));

  V2_dp_kernel<<<grid_size, block_size>>>(d_in, height, width, d_dp, d_trace);

  CHECK(cudaDeviceSynchronize());
  CHECK(cudaGetLastError());

  // trace back
  int *trace = new int[height * width];

  CHECK(cudaMemcpy(trace, d_trace, height * width * sizeof(int),
                   cudaMemcpyDeviceToHost));

  // TODO: replace min
  int *dp = new int[width];
  CHECK(cudaMemcpy(dp, d_dp + (height - 1) * width, width * sizeof(int),
                   cudaMemcpyDeviceToHost));

  // fix trace
  int pos = (int)(std::min_element(dp, dp + width) - dp);

  for (int i = height - 1; i >= 0; --i) {
    out[i] = pos;

    if (i > 0)
      pos = trace[i * width + pos];
  }

  timer.Stop();

  delete[] trace;
  delete[] dp;
  CHECK(cudaFree(d_in));
  CHECK(cudaFree(d_dp));
  CHECK(cudaFree(d_trace));

  return timer.Elapsed();
}

// tracing so we don't have to copy
__global__ void trace_kernel(int *d_trace) {}
