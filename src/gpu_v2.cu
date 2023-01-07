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
__device__ int *completed;

/*
   We divide the image into "strips",
   Block size 32 required.
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

  int threads_per_row = gridDim.x * blockDim.x;

  // dynamic block id
  int blockId_y = bi / gridDim.x;
  // also strip Idx
  int blockId_x = bi % gridDim.x;

#ifdef V2_DEBUG
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    printf("Block [%d, %d]\n", blockId_x, blockId_y);
  }
#endif

  int row = blockId_y * blockDim.y + threadIdx.y;
  int col = blockId_x * blockDim.x + threadIdx.x;

  if (row >= height || col >= width) {
    return;
  }

#ifdef V2_DEBUG
  printf("Row %d Col %d\n", row, col);
#endif

  int pos = row * width + col;

  // first row of the block
  if (threadIdx.y == 0) {

    d_dp[pos] = d_in[pos];
    atomicAdd(completed + blockId_y, 1);
  } else {
    // calculate required number of threads
    // all threads of previous rows in a strip
    int required = (threadIdx.y - 1) * width;

#ifdef V2_DEBUG
    printf("[%d, %d] Required = %d \n", row, col, required);
    printf("[%d, %d] B_y %d Grid %d %d \n", row, col, blockId_y);
#endif

    // wait for the required number of threads to complete
    while (atomicAdd(completed + blockId_y, 0) < required) {
      ;
    }

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

    atomicAdd(completed + blockId_y, 1);
  }

#ifdef V2_DEBUG_PRINT_DP
  printf("[%d, %d] DP = %d \n", row, col, d_dp[pos]);
#endif

  __syncthreads();

#ifdef V2_DEBUG_MERGE
  if (blockId_x == 0 && threadIdx.y == 0) {

    printf("[%d, %d]  %d Merging results. completed threads %d\n", row, col,
           blockId_y, atomicAdd(completed + blockId_y, 0));
  }
#endif

  // merge results.
  if (blockId_y > 0 && threadIdx.y == 0) {

    int required = blockDim.y * width;

    // wait for the required number of threads to complete
    // this may not be worth it.
    // prob better just to merge on host

    while (atomicAdd(completed + blockId_y - 1, 0) < required) {
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
    if (right < width && (ans == -1 || ans > d_dp[(row - 1) * width + right])) {
      ans = d_dp[(row - 1) * width + right];
      d_trace[pos] = right;
    }

    int tmp = d_dp[pos];
    d_dp[pos] = tmp + ans;
    __threadfence();
  }
}

// must : blocksize = 32
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

#ifdef V2_DEBUG
  cerr << "Grid size " << grid_size.x << ' ' << grid_size.y << '\n';
#endif

  int *host_completed;
  CHECK(cudaMalloc(&host_completed, sizeof(int) * grid_size.y));

  CHECK(cudaMemset(host_completed, 0, grid_size.y * sizeof(int)));

  CHECK(cudaMemcpyToSymbol(completed, &host_completed, sizeof(int *), size_t(0),
                           cudaMemcpyHostToDevice));

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
