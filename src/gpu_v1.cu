#include "gpu_utils.h"
#include "gpu_v1.h"
#include <algorithm>
#include <iostream>

using std::cerr;

const int SOBEL_X[] = {
    1, 0, -1, 2, 0, -2, 1, 0, -1,
};

const int SOBEL_Y[] = {1, 2, 1, 0, 0, 0, -1, -2, -1};

__constant__ int kern[9];

__global__ void V1_conv_kernel(int *in, int w, int h, int *out) {

  const int di = 1;
  const int filterWidth = 3;
  int blkR = blockIdx.y * blockDim.y;
  int blkC = blockIdx.x * blockDim.x;
  int r = blkR + threadIdx.y;
  int c = blkC + threadIdx.x;
  extern __shared__ int s_in[];
  s_in[threadIdx.y * blockDim.x + threadIdx.x] = in[r * w + c];
  __syncthreads();
  if (threadIdx.x < blockDim.x && threadIdx.y < blockDim.y) {
    int ind = r * w + c;
    int sum = 0;
    for (int i = 0; i < filterWidth; i++)
      for (int j = 0; j < filterWidth; j++) {
        int ki = i * filterWidth + j;
        int x = threadIdx.x - di + j;
        int y = threadIdx.y - di + i;
        if (blkC + x < 0 || blkC + x >= w)
          x = (blkC + x < 0) ? 0 : w - 1 - blkC;
        if (blkR + y < 0 || blkR + y >= h)
          y = (blkR + y < 0) ? 0 : h - 1 - blkR;
        if (x < blockDim.x && y < blockDim.y) {
          int convind = y * blockDim.x + x;
          sum += kern[ki] * s_in[convind];
        } else {
          y += blkR;
          x += blkC;
          int convind = y * w + x;
          sum += kern[ki] * in[convind];
        }
      }
    out[ind] = sum;
  }
}

// __global__ void min_kern(int* in, int n, int* out)
// {
//   int numElemsBeforeBlk = blockIdx.x * blockDim.x * 2;
//     for (int stride = blockDim.x; stride >= 1; stride /= 2)
//     {
//         int i = numElemsBeforeBlk + threadIdx.x;
//         if (threadIdx.x < stride)
//             if ((i + stride < n) && (in[i] != in[i + stride]))
//                 in[i] = in[i] < in[i + stride] ? in[i] : i[i + stride];

//         __syncthreads();
//     }
//     if (threadIdx.x == 0)
//         out[blockIdx.x] = in[numElemsBeforeBlk];
// }

#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void Tpose_kern(int *d_in, int height, int width, int *out) {
  __shared__ int tile[TILE_DIM][TILE_DIM];
  int i_n = blockIdx.x * TILE_DIM + threadIdx.x;
  int i_m = blockIdx.y * TILE_DIM + threadIdx.y;

  int i;
  for (i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
    if (i_n < height && (i_m + i) < width) {
      tile[threadIdx.y + i][threadIdx.x] = d_in[(i_m + i) * width + i_n];
    }
  }
  __syncthreads();

  i_n = blockIdx.y * TILE_DIM + threadIdx.x;
  i_m = blockIdx.x * TILE_DIM + threadIdx.y;

  for (i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
    if (i_n < height && (i_m + i) < width) {
      out[(i_m + i) * width + i_n] = tile[threadIdx.x][threadIdx.y + i];
    }
  }
}

__global__ void V1_grayscale_kernel(unsigned char *d_in, int height, int width,
                                    int *out) {

  int r = blockDim.y * blockIdx.y + threadIdx.y;
  int c = blockDim.x * blockIdx.x + threadIdx.x;

  if (r >= height || c >= width)
    return;
  int pos = (r * width + c) * 3;
  int ans = (d_in[pos] + d_in[pos + 1] + d_in[pos + 2]) / 3;
  out[pos] = ans;
}

void V1_grayscale(unsigned char *in, int height, int width, int *out, int block_size) {
  unsigned char *d_in;
  int *d_out;
  cudaMalloc(&d_in, height * width * sizeof(unsigned char));
  cudaMalloc(&d_out, height * width * sizeof(int));
  cudaMemcpy(d_in, in, height * width * sizeof(unsigned char),
             cudaMemcpyHostToDevice);
  dim3 blockSize(block_size, block_size);
  dim3 gridSize((width - 1) / blockSize.x + 1, (height - 1) / blockSize.y + 1);
  V1_grayscale_kernel<<<gridSize, blockSize>>>(d_in, height, width, d_out);
  cudaMemcpy(out, d_out, height * width * sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_in);
  cudaFree(d_out);
}

void V1_conv(int *in, int height, int width, bool sobelx, int *out) {
  int *d_in, *d_out;
  size_t imgSize = width * height * sizeof(int);
  size_t kernSize = 9 * sizeof(int);
  CHECK(cudaMalloc(&d_in, imgSize));
  CHECK(cudaMalloc(&d_out, imgSize));
  CHECK(cudaMemcpy(d_in, in, imgSize, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpyToSymbol(kern, SOBEL_X, kernSize));
  CHECK(cudaMemcpyToSymbol(kern, SOBEL_Y, kernSize));

  dim3 blockSize(32, 32);
  dim3 gridSize((width - 1) / blockSize.x + 1, (height - 1) / blockSize.y + 1);
  V1_conv_kernel<<<gridSize, blockSize, width * height * sizeof(int)>>>(d_in, width, height,
                                                               d_out);
  cudaDeviceSynchronize();
  cudaGetLastError();
  cudaMemcpy(out, d_out, width * height * sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_in);
  CHECK(cudaFree(d_out));
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
      tr = col_;
    }
  }

  d_trace[row * width + col] = tr;

#ifdef V1_DEBUG
  printf("%d %d %d\n", row, col, d_in[row * width + col]);
#endif

  d_dp[row * width + col] = ans + d_in[row * width + col];
#ifdef V1_DEBUG
  printf("DP %d %d %d\n", row, col, d_dp[row * width + col]);
#endif
}

/*
Input: n * m energy map
Output: result + time
*/
double V1_seam(int *in, int height, int width, int *out, int blocksize) {

#ifdef V1_DEBUG
  cerr << "==================================\n";
  cerr << "Debug for V1_seam" << '\n';
  cerr << "==================================\n";
#endif

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
#ifdef V1_DEBUG_ROW
    cerr << i << ' ';
#endif
    V1_dp_kernel<<<grid_size, block_size>>>(d_in, d_dp, d_trace, width, i);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
  }

#ifdef V1_DEBUG_ROW
    cerr << i << '\n';
#endif

  // trace back
  int *trace = new int[height * width];

  CHECK(cudaMemcpy(trace, d_trace, height * width * sizeof(int),
                   cudaMemcpyDeviceToHost));

  int *dp = new int[width];
  CHECK(cudaMemcpy(dp, d_dp + (height - 1) * width, width * sizeof(int),
                   cudaMemcpyDeviceToHost));

  // fix trace
  int pos = (int)(std::min_element(dp, dp + width) - dp);

#ifdef V1_DEBUG
  cerr << "Pos = " << pos << '\n';
#endif

  for (int i = height - 1; i >= 0; --i) {
    out[i] = pos;

    if (i > 0)
      pos = trace[i * width + pos];
  }

  delete[] trace;
  delete[] dp;
  CHECK(cudaFree(d_in));
  CHECK(cudaFree(d_dp));
  CHECK(cudaFree(d_trace));

  timer.Stop();

#ifdef DEBUG
  cerr << "End of debug for V1_seam" << '\n';
  cerr << "==================================\n";
#endif

  return timer.Elapsed();
}

__global__ void V1_seam_removal_kernel() {}

__global__ void V1_seam_add_kernel() {}

void v1_in_to_seam(unsigned char *in, int height, int width, char *out,
                   int blocksize) {

  unsigned char *d_in;
  CHECK(cudaMalloc(&d_in, sizeof(unsigned char) * 3 * height * width));

  int *d_gray;
  CHECK(cudaMalloc(&d_gray, sizeof(char) * height * width));

  dim3 block_size(blocksize, blocksize);
  dim3 grid_size((height - 1) / blocksize + 1, (width - 1) / blocksize + 1);

  V1_grayscale_kernel<<<grid_size, block_size>>>(d_in, height, width, d_gray);
}
