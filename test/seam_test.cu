#include "gpu_v1.h"
#include "gpu_v2.h"
#include "host.h"
#include "host_utils.h"
#include <iostream>
#include <vector>
#define X first
#define Y second

using namespace std;

/*
Input: Energy map
Output: Seam
   */

vector<pair<int, int>> dat_sz = {{4, 6}, {5, 6}, {5, 5}, {1, 11}};

/*
    255, 200, 19, 20, 18, 17,
    255, 1,   19, 20, 18, 17,
    255, 200, 9,  20, 18, 17,
    255, 200, 0,  20, 18, 17,
    ---
    {1000, 1000, 1000, 1000, 1000,
     1000, 1000, 1000, 1000, 1000,
     1000, 1000, 1000, 1000, 1000,
     1000, 1000, 1000, 1000, 1000,
     1000, 1000, 0,    1000, 1000}
   */
vector<vector<int>> dat = {
    {
        255, 200, 19, 20, 18, 17, 255, 1,   19, 20, 18, 17,
        255, 200, 9,  20, 18, 17, 255, 200, 0,  20, 18, 17,
    },
    {1000, 19,  20, 30,   500, 20, 1000, 19,  0,  30,   500, 20, 1000, 19,  20,
     19,   500, 20, 1000, 19,  2,  30,   500, 20, 1000, 19,  0,  30,   500, 20},

    {1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,
     1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,
     1000, 1000, 1000, 1000, 0,    1000, 1000},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}};

vector<vector<int>> ans = {{2, 1, 2, 2},
                           {1, 2, 1, 2, 2},
                           {0, 0, 0, 1, 2},
                           {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};

void host_test() {
  for (size_t i = 0; i < dat.size(); ++i) {
    cout << "Case " << i << '\n';
    vector<int> &V = dat[i];

    pair<int, int> s = dat_sz[i];
    int *act = new int[s.X];

    host_dp_seam(V.data(), s.X, s.Y, act);

    check_answer(act, ans[i].data(), s.X, i);
    delete[] act;
  }
}

void gpu_test(int ver = 1) {
  if (ver == 1) {
    for (size_t i = 0; i < dat.size(); ++i) {

      cout << "Case " << i << '\n';
      vector<int> &V = dat[i];

      pair<int, int> s = dat_sz[i];
      int *act = new int[s.X];

      V1_seam(V.data(), s.X, s.Y, act);
      check_answer(act, ans[i].data(), s.X, i);
      delete[] act;
    }
  } else if (ver == 2) {
    for (size_t i = 0; i < dat.size(); ++i) {

      cout << "Case " << i << '\n';
      vector<int> &V = dat[i];

      pair<int, int> s = dat_sz[i];
      int *act = new int[s.X];

      V2_seam(V.data(), s.X, s.Y, act);
      check_answer(act, ans[i].data(), s.X, i);
      delete[] act;
    }
  }
}

void rand_test_2(int ver, int num = 2) {
  int H = 64;
  int W = 32;

  srand(123445);
  cout << "Random test" << '\n';
  int *A = new int[H * H];
  int *host_ans = new int[H];
  int *gpu_ans = new int[H];
  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < H * W; ++j) {
      A[i] = rand() % 4000;
    }

    if (ver == 1) 
      V1_seam(A, H, W, gpu_ans);
    else  if (ver == 2)
      V2_seam(A, H, W, gpu_ans);

    host_dp_seam(A, H, W, host_ans);
    check_answer(gpu_ans, host_ans, H, i);
  }

  delete[] A;
  delete[] host_ans;
  delete[] gpu_ans;
}



void rand_test(int ver, int num = 2) {
  srand(111222);

  cout << "Random test" << '\n';
  int *A = new int[128 * 128];
  int *host_ans = new int[128];
  int *gpu_ans = new int[128];
  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < 128 * 128; ++j) {
      A[i] = rand() % 4000;
    }

    if (ver == 1) 
      V1_seam(A, 128, 128, gpu_ans);
    else  if (ver == 2)
      V2_seam(A, 128, 128, gpu_ans);

    host_dp_seam(A, 128, 128, host_ans);
    check_answer(gpu_ans, host_ans, 128, i);
  }

  delete[] A;
  delete[] host_ans;
  delete[] gpu_ans;
}



int main(int argc, char **argv) {
  int ver = 0;
  if (argc == 2) {
    ver = atoi(argv[1]);
  }
  if (ver == 0) {
    host_test();
  } else {
    std::cout << "Testing Gpu ver " << ver << '\n';
    std::cout << "Normal test" << ver << '\n';
    gpu_test(ver);
    rand_test(ver, 1);
  }
  return 0;
}
