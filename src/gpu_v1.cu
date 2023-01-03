#include "gpu_utils.h"
#include "gpu_v1.h"
#include <algorithm>
#include <iostream>

using std::cerr;


const int SOBEL_X[] = {
    1, 0, -1, 2, 0, -2, 1, 0, -1,
};

const int SOBEL_Y[] = {1, 2, 1, 0, 0, 0, -1, -2, -1};

__global__ void V1_conv_kernel(int *in, int n, int m, int *out) {

}

void V1_conv(int *in, int n, int m, int *out) {}

__global__ void V1_grayscale_kernel(unsigned char *d_in, int height, int width,
                                    int *out) {

  int r = blockDim.x * blockIdx.x + threadIdx.x;
  int c = blockDim.y * blockIdx.y + threadIdx.y;

  if (r >= height || c >= width)
    return;
  int pos = r * width + c;
  int ans = (d_in[pos] + d_in[pos + 1] + d_in[pos + 2]) / 3;
  out[pos] = ans;
}

/*
   Dynamic programming kernel for finding seam
   */
__global__ void V1_dp_kernel(int *d_in, int *d_dp, int *d_trace, int width,
                             int row) {

  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (col >= width)
    return;

  if (row == 0) {
    d_dp[col] = d_in[col];
    return;
  }

  int ans = -1;
  int tr = -1;

  for (int j = -1; j <= 1; ++j) {
    int col_ = col + j;
    if (col_ < 0 || col_ >= width)
      continue;

    int tmp = d_dp[(row - 1) * width + col_];
    if (ans == -1 || tmp < ans) {
      ans = tmp;
    }
  }

  d_trace[row * width + col] = tr;
  d_dp[row * width + col] = ans + d_in[row * width + col];
}

/*
Input: n * m energy map
Output: result + time
*/
double V1_seam(int *in, int height, int width, int *out, int blocksize) {

  GpuTimer timer;
  timer.Start();

  dim3 grid_size((width - 1) / blocksize + 1);
  dim3 block_size(blocksize);

  int *d_in;
  CHECK(cudaMalloc(&d_in, height * width * sizeof(int)));
  CHECK(cudaMemcpy(d_in, in, height * width * sizeof(int),
                   cudaMemcpyHostToDevice));

  int *d_dp;
  CHECK(cudaMalloc(&d_dp, height * width * sizeof(int)));

  int *d_trace;
  CHECK(cudaMalloc(&d_trace, height * width * sizeof(int)));

  for (int i = 0; i < height; ++i) {
    V1_dp_kernel<<<grid_size, block_size>>>(d_in, d_dp, d_trace, i, width);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
  }

  // trace back
  int *trace = new int[height * width];

  CHECK(cudaMemcpy(trace, d_trace, height * width * sizeof(int),
                   cudaMemcpyDeviceToHost));

  int pos = (int)(std::min_element(trace + (height - 1) * width,
                                   trace + height * width) -
                  (trace + (height - 1) * width));

#ifdef DEBUG 
  cerr << pos << '\n';
#endif

  for (int i = height - 1; i >= 0; --i) {
    out[i] = pos;

    if (i > 0)
      pos = trace[i * width + pos];
  }

  delete[] trace;
  CHECK(cudaFree(d_in));
  CHECK(cudaFree(d_dp));
  CHECK(cudaFree(d_trace));

  timer.Stop();
  return timer.Elapsed();
}

__global__ void V1_seam_removal_kernel() {}

__global__ void V1_seam_add_kernel() {}
