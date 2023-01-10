#include "gpu_utils.h"
#include "gpu_v1.h"
#include "host.h"
#include "host_utils.h"
#include <cstdlib>

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

/*
const int SOBEL_X[] = {
    1, 0, -1, 2, 0, -2, 1, 0, -1,
};

const int SOBEL_Y[] = {1, 2, 1, 0, 0, 0, -1, -2, -1};
*/

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

  for (size_t i = 0; i < dat.size(); ++i) {

    cout << "Case " << i << '\n';
    vector<int> &V = dat[i];

    pair<int, int> s = dat_sz[i];
    int *act = new int[s.H * s.W];

    if (ver == 1)
      V1_conv(V.data(), s.H, s.W, act);
    else if (ver == 2) 
      Test_conv(V.data(), s.H, s.W, act);

    check_answer(act, ans[i].data(), s.H * s.W, i);

    delete[] act;
  }
}

void rand_test(int ver, int HEIGHT = 128, int WIDTH = 128, int num = 2) {
  std::cout << "Random test, size =  " << HEIGHT << " " << WIDTH << '\n';

  srand(222022);
  int *A = new int[HEIGHT * WIDTH];
  int *host_ans = new int[HEIGHT * WIDTH];
  int *gpu_ans = new int[HEIGHT * WIDTH];
  for (int i = 0; i < num; ++i) {

    cout << "Case " << i << '\n';

    int *host_ans = new int[HEIGHT * WIDTH];
    int *gpu_ans = new int[HEIGHT * WIDTH];

    for (int j = 0; j < HEIGHT * WIDTH; ++j) {
      A[i] = rand() % 4000;
    }
    if (ver == 1)
      V1_conv(A, HEIGHT, WIDTH, gpu_ans);
    else if (ver == 2)
      Test_conv(A, HEIGHT, WIDTH, gpu_ans);


    host_sobel_conv(A, HEIGHT, WIDTH, host_ans);

    cout << '\n';
    check_answer(gpu_ans, host_ans, HEIGHT * WIDTH, i);

    delete[] host_ans;
    delete[] gpu_ans;
  }

  delete[] A;
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
    rand_test(ver, 1024, 768, 20);
    rand_test(ver, 1024, 777, 20);
    rand_test(ver, 777, 777, 20);
    rand_test(ver, 238, 777, 20);
  }
  return 0;
}
