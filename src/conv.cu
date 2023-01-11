#include "conv.h"
#include "gpu_utils.h"
#include <iostream>

using std::cerr;

/*
const int SOBEL_X[] = {
    1, 0, -1,
    2, 0, -2,
    1, 0, -1,
};

const int SOBEL_Y[] = {
  1, 2, 1,
  0, 0, 0,
  -1, -2, -1};
*/

// padding 1
//  pitch = width + 2

// TILE = 16 * 16
// apron = 18 * 18

#define TILE_SIZE 30
#define BLOCK_SIZE 32

__global__ void V2_conv_kernel(int *d_in, int height, int width, bool p,
                                 int *d_out) {

  // int row =

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // pos inside the original image
  int row_o = blockIdx.y * TILE_SIZE + ty;
  int col_o = blockIdx.x * TILE_SIZE + tx;
  int pos_o = row_o * width + col_o;

  // pos inside the padded image
  int row_i = row_o - 1;
  int col_i = col_o - 1;

  // (TILE_SIZE + kernel_size - 1) * (TILE_SIZE + kernel_size -1)
  // (block_size)
  extern __shared__ int N_ds[];

  int pitch = width;

  if ((row_i >= 0) && (row_i < height) && (col_i >= 0) && (col_i < width)) {
    N_ds[ty * blockDim.x + tx] = d_in[row_i * pitch + col_i];
  } else {
    int tmp_row_i = min(max(row_i, 0), height - 1);
    int tmp_col_i = min(max(col_i, 0), width - 1);
    N_ds[ty * blockDim.x + tx] = d_in[tmp_row_i * pitch + tmp_col_i];
    // N_ds[ty * blockDim.x + tx] = 0;
  }

  __syncthreads();

  int ans = 0;

  if (pos_o == 599040) {
    printf("%d %d\n", row_o, col_o);
  }

  if (ty < TILE_SIZE && tx < TILE_SIZE) {
    // +(-1, -1), -(-1, 1), -(0, -1), +(0 ,1), -(1, -1), +(1, 1)
    // +(0,  0) , -(0, 2), -(1, 0),

    if (p) {
      ans += N_ds[ty * blockDim.x + tx] - N_ds[ty * blockDim.x + (tx + 2)];
      ans += 2 * N_ds[(ty + 1) * blockDim.x + tx] -
             2 * N_ds[(ty + 1) * blockDim.x + (tx + 2)];
      ans += N_ds[(ty + 2) * blockDim.x + tx] -
             N_ds[(ty + 2) * blockDim.x + (tx + 2)];

    } else {

      ans += N_ds[ty * blockDim.x + tx] + 2 * N_ds[ty * blockDim.x + (tx + 1)] +
             N_ds[ty * blockDim.x + (tx + 2)];

      ans -= (N_ds[(ty + 2) * blockDim.x + tx] +
              2 * N_ds[(ty + 2) * blockDim.x + (tx + 1)] +
              N_ds[(ty + 2) * blockDim.x + (tx + 2)]);
    }
    if (row_o < height && col_o < width) {
      d_out[row_o * width + col_o] = ans;
    }
  }
}

__global__ void V2_sum_abs_kernel(int *d_in1, int *d_in2, int num_pixels,
                                  int *d_out) {
  int pos = blockDim.x * blockIdx.x + threadIdx.x;
  if (pos >= num_pixels)
    return;

  d_out[pos] = abs(d_in1[pos]) + abs(d_in2[pos]);
}

void V2_conv(int *in, int height, int width, int *out) {

  int *d_in;
  CHECK(cudaMalloc(&d_in, height * width * sizeof(int)));
  CHECK(cudaMemcpy(d_in, in, height * width * sizeof(int),
                   cudaMemcpyHostToDevice));

  int *d_out1, *d_out2, *d_out;
  CHECK(cudaMalloc(&d_out1, height * width * sizeof(int)));
  CHECK(cudaMalloc(&d_out2, height * width * sizeof(int)));

#ifdef V2_CONV_DEBUG
  cerr << "Launching conv kernel for image size " << height << "x" << width
       << '\n';
#endif

  dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid_size((height - 1) / TILE_SIZE + 1, (width - 1) / TILE_SIZE + 1);

  V2_conv_kernel<<<grid_size, block_size,
                     BLOCK_SIZE * BLOCK_SIZE * sizeof(int)>>>(d_in, height,
                                                              width, 0, d_out1);

  CHECK(cudaDeviceSynchronize());
  CHECK(cudaGetLastError());

  V2_conv_kernel<<<grid_size, block_size,
                     BLOCK_SIZE * BLOCK_SIZE * sizeof(int)>>>(d_in, height,
                                                              width, 1, d_out2);

  CHECK(cudaDeviceSynchronize());
  CHECK(cudaGetLastError());

  CHECK(cudaMalloc(&d_out, height * width * sizeof(int)));
  int num_pixels = height * width;

  grid_size = dim3((num_pixels - 1) / 256 + 1);
  block_size = dim3(256);

  V2_sum_abs_kernel<<<grid_size, block_size>>>(d_out1, d_out2, num_pixels,
                                               d_out);

  CHECK(cudaDeviceSynchronize());
  CHECK(cudaGetLastError());

  CHECK(cudaMemcpy(out, d_out, num_pixels * sizeof(int),
                   cudaMemcpyDeviceToHost));

  CHECK(cudaFree(d_in));
  CHECK(cudaFree(d_out));
  CHECK(cudaFree(d_out1));
  CHECK(cudaFree(d_out2));
}
