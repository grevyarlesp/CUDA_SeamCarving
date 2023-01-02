#include <math.h>
#include <stdio.h>
#include <vector>
#include <algorithm>

using namespace std;

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
}

/*
Input: n * m  energy map
Output: Seam: 1D array of n elemenets, indicate the pixel for each row
   */
void host_dp_seam(int *in, int n, int m, int *out) {
  const int INF = 1e9;
  vector< vector<int> > dp(n, vector<int>(m, INF));
  vector< vector<int> > trace(n, vector<int>(m, -1));
  
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      if (i == 0) {
        dp[i][j] = in[j];
        continue;
      }
      for (int k = -1; k <= 1; ++k) {
        int prev_col = j + k;
        if (prev_col < 0 || prev_col >= m) continue;
        if (dp[i - 1][prev_col] < dp[i][j]) {
          dp[i][j] = dp[i - 1][prev_col];
          trace[i][j] = prev_col;
        }
      }
      dp[i][j] += in[i * m + j];
    }
  }

  // tracing back
  vector<int> ans;
  int pos = (int) (min_element(dp[n - 1].begin(), dp[n - 1].end()) - dp[n - 1].begin());

  for (int i = n - 1; i >= 0; --i) {
    out[i] = pos;
    if (i > 0) {
      pos = trace[i][pos];
    }
  }

}

void host_full(int *in, int n, int m, int *out) {

  return;
}
