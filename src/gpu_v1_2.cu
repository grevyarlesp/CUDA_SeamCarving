#include "gpu_utils.h"
#include "gpu_v1.h"
#include <algorithm>
#include <iostream>

using std::cerr;

/*
   Dynamic programming kernel for finding seam
   */
__global__ void V1_2_dp_kernel(int *d_in, int *d_dp, int *d_trace, int width,
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
double V1_2_seam(int *in, int height, int width, int *out, int blocksize, int nStreams) {

#ifdef V1_DEBUG
  cerr << "==================================\n";
  cerr << "Debug for V1_seam" << '\n';
  cerr << "==================================\n";
#endif

  GpuTimer timer;
  timer.Start();

  dim3 grid_size((width - 1) / blocksize + 1);
  dim3 block_size(blocksize);

  int matBytes = height * width * sizeof(int);

  int *d_in;
  CHECK(cudaMalloc(&d_in, matBytes));
  CHECK(cudaMemcpy(d_in, in, matBytes,
                   cudaMemcpyHostToDevice));

  int *d_dp;
  CHECK(cudaMalloc(&d_dp, matBytes));

  int *d_trace;
  CHECK(cudaMalloc(&d_trace, matBytes));


  int *trace = new int[height * width];

  CHECK(cudaHostRegister(in, matBytes, cudaHostRegisterDefault));
  CHECK(cudaHostRegister(trace, matBytes, cudaHostRegisterDefault));


  cudaStream_t *streams;
  streams = (cudaStream_t *)malloc(sizeof(cudaStream_t) * nStreams);

  for (int i = 0; i < height; ++i) {
#ifdef V1_DEBUG
    cerr << "Row " << i << '\n';
#endif
    V1_2_dp_kernel<<<grid_size, block_size>>>(d_in, d_dp, d_trace, width, i);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
  }

  // trace back

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


