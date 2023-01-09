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

#define TILE_DIM 32

__global__ void V1_sum(int *in1, int *in2, int width, int height, int *out) {
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= height || c >= width)
    return;
  int pos = r * width + c;
  out[pos] = abs(in1[pos]) + abs(in2[pos]);
}

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

// __inline__ __device__ void warpReduceMin(int& val, int& idx)
// {
//     for (int offset = warpSize / 2; offset > 0; offset /= 2) {
//         int tmpVal = __shfl_down(val, offset);
//         int tmpIdx = __shfl_down(idx, offset);
//         if (tmpVal < val) {
//             val = tmpVal;
//             idx = tmpIdx;
//         }
//     }
// }
//
// __inline__ __device__  void blockReduceMin(int& val, int& idx)
// {
//
//     __shared__ int values[32], indices[32]; // Shared mem for 32 partial mins
//     int lane = threadIdx.x % warpSize;
//     int wid = threadIdx.x / warpSize;
//
//     warpReduceMin(val, idx);     // Each warp performs partial reduction
//
//     if (lane == 0) {
//         values[wid] = val; // Write reduced value to shared memory
//         indices[wid] = idx; // Write reduced value to shared memory
//     }
//
//     __syncthreads();              // Wait for all partial reductions
//
//     //read from shared memory only if that warp existed
//     if (threadIdx.x < blockDim.x / warpSize) {
//         val = values[lane];
//         idx = indices[lane];
//     } else {
//         val = INT_MAX;
//         idx = 0;
//     }
//
//     if (wid == 0) {
//          warpReduceMin(val, idx); //Final reduce within first warp
//     }
// }
//
// __global__ void V1_min_kernel(int * in, int* ind, int n, int * out)
// {
// 	// TODO
//     int min_val = INT_MAX;
//     int min_ind = 0;
//     int numElemsBeforeBlk = blockIdx.x * blockDim.x * 2;
//     for (int i = blockIdx.x * blockDim.x + threadIdx.x;
//         i < n;
//         i += blockDim.x * gridDim.x)
//         {
//             if (in[i] < min_val)
//             {
//                 min_val = in[i];
//                 min_ind = i;
//             }
//         }
//     blockReduceMin(min_val, min_ind);
//     if (threadIdx.x == 0)
//         out[blockIdx.x] = min_ind;
// }
//
// int V1_min(int* in, int* ind, int n, int block_size){
//     int *d_in, *d_ind, *d_out;
//     dim3 blockSize(block_size);
//     dim3 gridSize((n - 1)/blockSize.x + 1);
//     int* out = (int*)malloc(gridSize.x*sizeof(int));
//     CHECK(cudaMalloc(&d_in, n*sizeof(int)));
//     CHECK(cudaMalloc(&d_ind, n*sizeof(int)));
//     CHECK(cudaMalloc(&d_out, gridSize.x*sizeof(int)));
//     CHECK(cudaMemcpy(d_in, in, n*sizeof(int), cudaMemcpyHostToDevice));
//     CHECK(cudaMemcpy(d_ind, ind, n*sizeof(int), cudaMemcpyHostToDevice));
//
//     V1_min_kernel<<<gridSize, blockSize>>>(d_in, d_ind, n, d_out);
//     CHECK(cudaMemcpy(out, d_out, gridSize.x*sizeof(int),
//     cudaMemcpyDeviceToHost)); int min_val = INT_MAX; int min_ind = 0; for(int
//     i = 0; i < gridSize.x; i++)
//     {
//         if(in[out[i]] < min_val)
//         {
//             min_val = in[out[i]];
//             min_ind = out[i];
//         }
//     }
//     free(out);
//     CHECK(cudaFree(d_in));
//     CHECK(cudaFree(d_ind));
//     CHECK(cudaFree(d_out));
//     return min_ind;
// }
//
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

  int pos = r * width + c;
  int pos_ = pos * 3;

  int ans = d_in[pos_] * 3;
  ans = ans + d_in[pos_ + 1] * 6 + d_in[pos_ + 2];
  ans /= 10;

  out[pos] = ans;
}

void V1_grayscale(unsigned char *in, int height, int width, int *out,
                  int block_size) {
  unsigned char *d_in;
  int *d_out;
  cudaMalloc(&d_in, height * width * 3 * sizeof(unsigned char));
  cudaMalloc(&d_out, height * width * sizeof(int));
  cudaMemcpy(d_in, in, height * width * 3 * sizeof(unsigned char),
             cudaMemcpyHostToDevice);
  dim3 blockSize(block_size, block_size);
  dim3 gridSize((width - 1) / blockSize.x + 1, (height - 1) / blockSize.y + 1);
  V1_grayscale_kernel<<<gridSize, blockSize>>>(d_in, height, width, d_out);
  cudaDeviceSynchronize();
  CHECK(cudaMemcpy(out, d_out, height * width * sizeof(int),
                   cudaMemcpyDeviceToHost));
  cudaFree(d_in);
  cudaFree(d_out);
}

void V1_conv(int *in, int height, int width, int *out, int block_size) {
  int *d_in, *d_out, *d_temp2, *d_temp1;
  size_t imgSize = width * height * sizeof(int);
  size_t kernSize = 9 * sizeof(int);

  CHECK(cudaMalloc(&d_in, imgSize));
  CHECK(cudaMalloc(&d_temp1, imgSize));
  CHECK(cudaMalloc(&d_temp2, imgSize));
  CHECK(cudaMalloc(&d_out, imgSize));
  CHECK(cudaMemcpy(d_in, in, imgSize, cudaMemcpyHostToDevice));

  dim3 blockSize(block_size, block_size);
  dim3 gridSize((width - 1) / blockSize.x + 1, (height - 1) / blockSize.y + 1);

  // Sobel X
  CHECK(cudaMemcpyToSymbol(kern, SOBEL_X, kernSize));
  V1_conv_kernel<<<gridSize, blockSize, TILE_DIM * TILE_DIM * sizeof(int)>>>(
      d_in, width, height, d_temp1);
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaGetLastError());

  // Sobel Y
  CHECK(cudaMemcpyToSymbol(kern, SOBEL_Y, kernSize));
  V1_conv_kernel<<<gridSize, blockSize, TILE_DIM * TILE_DIM * sizeof(int)>>>(
      d_in, width, height, d_temp2);
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaGetLastError());

  // Combine
  V1_sum<<<gridSize, blockSize>>>(d_temp1, d_temp2, width, height, d_out);
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaGetLastError());
  CHECK(cudaMemcpy(out, d_out, width * height * sizeof(int), cudaMemcpyDeviceToHost));
  CHECK(cudaFree(d_in));
  CHECK(cudaFree(d_temp1));
  CHECK(cudaFree(d_temp2));
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
  int pos = row * width + col;
  int prev = pos - width;

  int ans = d_dp[prev];
  int tr = col;

  for (int j = -1; j <= 1; j += 2) {
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
#ifdef V1_DEBUG
    cerr << "Row " << i << '\n';
#endif
    V1_dp_kernel<<<grid_size, block_size>>>(d_in, d_dp, d_trace, width, i);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
  }

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

  timer.Stop();

  delete[] trace;
  delete[] dp;
  CHECK(cudaFree(d_in));
  CHECK(cudaFree(d_dp));
  CHECK(cudaFree(d_trace));

#ifdef DEBUG
  cerr << "End of debug for V1_seam" << '\n';
  cerr << "==================================\n";
#endif

  return timer.Elapsed();
}

__global__ void V1_seam_removal_kernel(int *d_in, int height, int width,
                                       int *d_out) {}

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
