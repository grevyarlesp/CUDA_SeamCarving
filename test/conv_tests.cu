#include "gpu_utils.h"
#include "host.h"
#include <stdio.h>
#include "gpu_v1.h"

const int SOBEL_X[] = {
    1, 0, -1, 2, 0, -2, 1, 0, -1,
};

const int SOBEL_Y[] = {1, 2, 1, 0, 0, 0, -1, -2, -1};

// 4 * 6
int case1[24] = {255, 166, 133, 222, 14, 9,  22, 11, 33, 44, 55, 22,
                 22,  33,  44,  55,  66, 77, 22, 55, 99, 10, 20, 30};

int answer1[24] = {1132, 998, 734, 750, 744, 128, 932, 666, 622, 424, 240, 316,
                   88,   286, 174, 50,  138, 12,  132, 352, 200, 296, 266, 228};



bool check_answer(int *act, int *expected, int n, int ncase=-1) {
  printf("Case %d\n:", ncase);

  for (int i = 0; i < n; ++i) {
    if (act[i] != expected[i]) {
      printf("WRONG ANSWER\n");
      printf("Diff at %d, got %d, expected %d", i, act[i], expected[i]);
      return false;
    }
  }

  printf("CORRECT\n");
  return true;
}

void host_test() {
  int* act1;
  act1 = (int *)malloc(4 * 6 * sizeof(int));
  host_sobel_conv(case1, 4, 6, act1);

  check_answer(act1, answer1, 4 * 6, 1);
  free(act1);
}

void test_gpu(int ver) {

  
  
}

int main() {
  host_test();

  return 0;
}
