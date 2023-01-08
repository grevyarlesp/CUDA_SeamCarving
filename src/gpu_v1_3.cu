#include "gpu_utils.h"
#include "gpu_v1.h"
#include "gpu_v1_2.h"
#include <algorithm>
#include <iostream>

using std::cerr;

/*
   Dynamic programming kernel for finding seam
   */
__global__ void V1_2_dp_kernel(int *d_in, int *d_dp_prev, int *d_dp_cur,
                               int *d_trace, int width, int row) {

  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (col >= width)
    return;

  if (row == 0) {
    d_dp_cur[col] = d_in[col];
    return;
  }

  int pos = row * width + col;

  // middle
  int ans = d_dp_prev[col];
  int tr = col;

  for (int j = -1; j <= 1; j += 2) {
    int col_ = col + j;
    if (col_ < 0 || col_ >= width)
      continue;

    int tmp = d_dp_prev[col_];

    if (tmp < ans) {
      ans = tmp;
      tr = col_;
    }
  }

  d_trace[pos] = tr;
  d_dp_cur[col] = ans + d_in[pos];
}

/*
Input: n * m energy map
Output: result + time
*/
double V1_2_seam(int *in, int height, int width, int *out, int blocksize) {

  GpuTimer timer;
  timer.Start();

  dim3 grid_size((width - 1) / blocksize + 1);
  dim3 block_size(blocksize);

  int matBytes = height * width * sizeof(int);

  int *d_in;
  CHECK(cudaMalloc(&d_in, matBytes));
  CHECK(cudaMemcpy(d_in, in, matBytes, cudaMemcpyHostToDevice));

  // int *d_dp;
  // CHECK(cudaMalloc(&d_dp, matBytes));

  int row_sz = width * sizeof(int);

  int *d_dp_cur;
  CHECK(cudaMalloc(&d_dp_cur, row_sz));

  int *d_dp_prev;
  CHECK(cudaMalloc(&d_dp_prev, row_sz));

  int *d_trace;
  CHECK(cudaMalloc(&d_trace, matBytes));

  int *trace = new int[height * width];

  // CHECK(cudaHostRegister(in, matBytes, cudaHostRegisterDefault));
  // CHECK(cudaHostRegister(trace, matBytes, cudaHostRegisterDefault));

  // cudaStream_t *strems;
  // streams = (cudaStream_t *)malloc(sizeof(cudaStream_t) * nStreams);

  for (int i = 0, start = 0; i < height; ++i, start+=width) {
    V1_2_dp_kernel<<<grid_size, block_size>>>(d_in, d_dp_prev, d_dp_cur,
                                              d_trace, width, i);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());

    int *tmp = d_dp_prev;
    d_dp_prev = d_dp_cur;
    d_dp_cur = tmp;
  }

  CHECK(cudaMemcpy(trace, d_trace, matBytes, cudaMemcpyDeviceToHost));

  int *dp = new int[width];
  CHECK(cudaMemcpy(dp, d_dp_prev, row_sz, cudaMemcpyDeviceToHost));

  int pos = (int)(std::min_element(dp, dp + width) - dp);

#if V1_2_DEBUG
  std::cout << "Tracing\n";
#endif
  for (int i = height - 1; i >= 0; --i) {
    out[i] = pos;
#if V1_2_DEBUG
    cerr << i << ' ' << pos << ' ' << trace[i * width + pos] << '\n';
#endif
    if (i > 0)
      pos = trace[i * width + pos];
  }

  timer.Stop();

  delete[] trace;
  delete[] dp;
  CHECK(cudaFree(d_in));
  CHECK(cudaFree(d_dp_cur));
  CHECK(cudaFree(d_dp_prev));
  CHECK(cudaFree(d_trace));
  return timer.Elapsed();
}



double remove_seam(unsigned char *img, int num) {

    

}
