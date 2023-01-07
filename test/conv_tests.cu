#include "gpu_utils.h"
#include "gpu_v1.h"
#include "host.h"
#include "host_utils.h"

#include <cstdio>
#include <iostream>
#include <vector>

#define H first
#define W second

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace std;

const int SOBEL_X[] = {
    1, 0, -1, 2, 0, -2, 1, 0, -1,
};

const int SOBEL_Y[] = {1, 2, 1, 0, 0, 0, -1, -2, -1};

// 4 * 6
vector<vector<int>> dat = {{255, 166, 133, 222, 14, 9,  22, 11,
                            33,  44,  55,  22,  22, 33, 44, 55,
                            66,  77,  22,  55,  99, 10, 20, 30}};

vector<pair<int, int>> dat_sz = {{4, 6}, {5, 6}, {5, 5}, {1, 11}};

vector<vector<int>> ans = {{1132, 998, 734, 750, 744, 128, 932, 666,
                            622,  424, 240, 316, 88,  286, 174, 50,
                            138,  12,  132, 352, 200, 296, 266, 228}};

void host_test() {
  int *act1;
  act1 = (int *)malloc(4 * 6 * sizeof(int));
  host_sobel_conv(dat[0].data(), 4, 6, act1);

  check_answer(act1, ans[0].data(), 4 * 6, 1);
  free(act1);
}

void gpu_test(int ver) {
  srand(7123);

  if (ver == 1) {
    for (size_t i = 0; i < dat.size(); ++i) {

      cout << "Case " << i << '\n';
      vector<int> &V = dat[i];

      pair<int, int> s = dat_sz[i];
      int *act = new int[s.H * s.W];

      V1_conv(V.data(), s.H, s.W, act);
      check_answer(act, ans[i].data(), s.H * s.W, i);

      delete[] act;
    }

  } else {
  }
}

void rand_test(int num = 2) {
  srand(10000);
  cout << "Random test" << '\n';
  int *A = new int[128 * 128];
  int *host_ans = new int[128 * 128];
  int *gpu_ans = new int[128 * 128];
  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < 128 * 128; ++j) {
      A[i] = rand() % 2000;
    }
    V1_conv(A, 128, 128, gpu_ans);
    host_sobel_conv(A, 128, 128, host_ans);
    check_answer(gpu_ans, host_ans, 128 * 128, i);
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
    gpu_test(ver);
    rand_test();
  }
  return 0;
}
