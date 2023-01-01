#include <math.h>
#include <stdio.h>

const int SOBEL_X[] = {
    1, 0, -1, 2, 0, -2, 1, 0, -1,
};

const int SOBEL_Y[] = {1, 2, 1, 0, 0, 0, -1, -2, -1};

/*
Input: n * m grayscale image, n *m rows
Output: n * m energy map
*/
void host_sobel_conv(int *in, int n, int m, int *out) {
  if (out == nullptr) {
    return;
  }

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      int sum1 = 0;
      int sum2 = 0;
      for (int i_ = -1, cnt = 0; i_ <= 1; ++i_) {
        for (int j_ = -1; j_ <= 1; ++j_, ++cnt) {
          int r_ = i - i_;
          int c_ = j - j_;
          // printf("%d %d\n", r_, c_);
          r_ = max(0, min(r_, n - 1));
          c_ = max(0, min(c_, m - 1));
          int pos = r_ * m + c_;
          sum1 += in[pos] * SOBEL_X[cnt];
          sum2 += in[pos] * SOBEL_Y[cnt];
          // printf("%d %d", sum1, sum2);
        }
      }
      out[i * m + j] = abs(sum1) + abs(sum2);
    }
  }
}

/*
Input: n * m  energy map
Output: Seam: 1D array of n elemenets, indicate the pixel for each row
   */
void host_dp_seam(int *in, int n, int m, int *out) {

}

void host_full(int *in, int n, int m, int* out) {
  
}
