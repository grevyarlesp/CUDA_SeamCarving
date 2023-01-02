#include "gpu_v1.h"

const int SOBEL_X[] = {
    1, 0, -1, 2, 0, -2, 1, 0, -1,
};

const int SOBEL_Y[] = {1, 2, 1, 0, 0, 0, -1, -2, -1};

__global__ void V1_conv_kernel(int *in, int n, int m, int *out) {}

void V1_conv(int *in, int n, int m, int *out) {}

/*
   Dynamic programming kernel for finding seam
   */
__global__ void V1_dp_kernel(int *d_in, int *d_dp, int *d_trace, int row,
                             int col_size) {

  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (col >= col_size)
    return;
  if (row == 0) {
  }

  int ans = -1;
  int tr = -1;

  for (int j = -1; j <= 1; ++j) {
    int col_ = col + j;
    if (col_ < 0 || col_ >= col_size)
      continue;
  }

  d_trace[row * col_size + col] = tr;
  d_dp[row * col_size + col] = ans;
}

__global__ void seam_removal_kernel() {

}

__global__ void seam_addition_kernel() {


}



/*
Input: n * m energy map
Output: result
*/
void V1_seam(int *in, int n, int m, int *out, int block_size = 256) {

  int grid_size = (m - 1) / block_size + 1;

  for (int i = 0; i < n; ++i) {
  }
}
