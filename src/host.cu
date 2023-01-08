#include <algorithm>
#include <cassert>

#include <cstdlib>
#include "gpu_utils.h"
#include <cwchar>
#include <math.h>
#include <stdio.h>
#include <vector>
#include <iostream>

using namespace std;

const int SOBEL_X[] = {
    1, 0, -1, 2, 0, -2, 1, 0, -1,
};

const int SOBEL_Y[] = {1, 2, 1, 0, 0, 0, -1, -2, -1};

/*
  Converting to grayscale
  Input: 3 * height * width
  Output: height * width
 */
void host_to_grayscale(unsigned char *in, int height, int width, int *out) {

  for (int i = 0, cnt = 0; i < height * width * 3; i += 3, ++cnt) {
    int x = 0; 
    x += in[i] * 3;
    x += in[i + 1] * 6; 
    x += in[i + 2];
    x /= 10;
    out[cnt] = x;
  }
}

/*
   For highlighting seam:
Input: 3 * width * height, seam of [height] elements
Output: 3 * width * height, with seam highlighted in red
 */

void host_highlight_seam(unsigned char *out, int height, int width, int *seam) {
  for (int i = 0; i < height; ++i) {
    assert(seam[i] != -1);
    int pos = (i * width + seam[i]) * 3;
    out[pos + 0] = 255;
    out[pos + 1] = 0;
    out[pos + 2] = 0;
  }
}

/*
Input: n * m grayscale image, n *m rows
Output: n * m energy map
*/
void host_sobel_conv(int *in, int height, int width, int *out) {
  if (out == nullptr) {
    return;
  }

    for (int i = 0; i < height; ++i) {
      for (int j = 0; j < width; ++j) {
        int sum1 = 0;
        int sum2 = 0;
        for (int i_ = -1, cnt = 0; i_ <= 1; ++i_) {
          for (int j_ = -1; j_ <= 1; ++j_, ++cnt) {
            int r_ = i - i_;
            int c_ = j - j_;
            // printf("%d %d\n", r_, c_);
            r_ = max(0, min(r_, height - 1));
            c_ = max(0, min(c_, width - 1));
            int pos = r_ * width + c_;
            sum1 += in[pos] * SOBEL_X[cnt];
            sum2 += in[pos] * SOBEL_Y[cnt];
            // printf("%d %d", sum1, sum2);
          }
        }
        out[i * width + j] = abs(sum1) + abs(sum2);
      }
  }
}

/*
Input: n * m  energy map
Output: Seam: 1D array of n elemenets, indicate the pixel for each row
   */
void host_dp_seam(int *in, int height, int width, int *out) {
  const int INF = 1e9;
  vector<vector<int>> dp(height, vector<int>(width, INF));
  vector<vector<int>> trace(height, vector<int>(width, -1));

  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      if (i == 0) {
        dp[i][j] = in[j];
        continue;
      }

      dp[i][j] = dp[i - 1][j];
      trace[i][j] = j;

      for (int k = -1; k <= 1; k += 2) {
        int prev_col = j + k;
        if (prev_col < 0 || prev_col >= width)
          continue;
        if (dp[i - 1][prev_col] < dp[i][j]) {
          dp[i][j] = dp[i - 1][prev_col];
          trace[i][j] = prev_col;
        }
      }
      dp[i][j] += in[i * width + j];
    }
  }

  // tracing back
  int pos = (int)(min_element(dp[height - 1].begin(), dp[height - 1].end()) -
                  dp[height - 1].begin());

  for (int i = height - 1; i >= 0; --i) {
    out[i] = pos;
    if (i > 0) {
      pos = trace[i][pos];
    }
  }
}

void host_full(unsigned char *to_process, int height, int width, int *seam) {

  // to grayscale
  int *gray = new int[height * width];
  host_to_grayscale(to_process, height, width, gray);

  int *energy_map = new int[height * width];
  host_sobel_conv(gray, height, width, energy_map);

  GpuTimer timer;
  timer.Start();
  // seam = [height]
  host_dp_seam(energy_map, height, width, seam);
  timer.Stop();

  cout << "Seam time" << ' ' << timer.Elapsed() << '\n';

  // out: 3 * width * height

  host_highlight_seam(to_process, height, width, seam);

  delete[] gray;
  delete[] energy_map;
  return;
}
