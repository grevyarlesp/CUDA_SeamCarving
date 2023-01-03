#include "gpu_v1.h"
#include "gpu_utils.h"
#include <algorithm>



const int SOBEL_X[] = {
    1, 0, -1, 2, 0, -2, 1, 0, -1,
};

const int SOBEL_Y[] = {1, 2, 1, 0, 0, 0, -1, -2, -1};

__global__ void V1_conv_kernel(int *in, int n, int m, int *out) {

}

void V1_conv(int *in, int n, int m, int *out) {

}

/*
   Dynamic programming kernel for finding seam
   */
__global__ void V1_dp_kernel(int *d_in, int *d_dp, int *d_trace, int row,
                             int col_size) {

  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (col >= col_size)
    return;

  if (row == 0) {
    d_dp[col] = d_in[0 + col];
    return; 
  }

  int ans = -1;
  int tr = -1;

  for (int j = -1; j <= 1; ++j) {
    int col_ = col + j;
    if (col_ < 0 || col_ >= col_size)
      continue;

    int tmp = d_dp[(row - 1) * col_size + col_];
    if (ans == -1 || tmp < ans) {
      ans = tmp;
    }
  }

  d_trace[row * col_size + col] = tr;
  d_dp[row * col_size + col] = ans + d_in[row * col_size + col];
}

/*
Input: n * m energy map
Output: result + time
*/
double V1_seam(int *in, int n, int m, int *out, int blocksize) {

  dim3 grid_size((m - 1) / blocksize + 1);
  dim3 block_size(blocksize);

  int *d_in;
  CHECK(cudaMalloc(&d_in, n * m * sizeof(int)));
  CHECK(cudaMemcpy(d_in, in, n * m * sizeof(int), cudaMemcpyHostToDevice));

  int *d_dp;
  CHECK(cudaMalloc(&d_dp, n * m * sizeof(int)));

  int *d_trace;
  CHECK(cudaMalloc(&d_trace, n * m * sizeof(int)));


  for (int i = 0; i < n; ++i) {
    V1_dp_kernel<<<grid_size, block_size>>>(d_in, d_dp, d_trace, i, m);
    CHECK(cudaDeviceSynchronize());
  }

  // trace back
  int* trace = new int[n * m];

  CHECK(cudaMemcpy(trace, d_trace, n * m * sizeof(int), cudaMemcpyDeviceToHost));

  int pos = (int) (std::min_element(trace + (n - 1) * m, trace + n * m) - (trace + (n - 1)));
  
  for (int i = n - 1; i >= 0; --i) {
    out[i] = pos;

    if (i > 0)
      pos = trace[i * m + pos];

  }

  delete[] trace;
  CHECK(cudaFree(d_in));
  CHECK(cudaFree(d_dp));
  CHECK(cudaFree(d_trace));

  return 0.0;
}

__global__ void seam_removal_kernel() {

}

__global__ void seam_add_kernel() {


}
