#include "gpu_v1.h"

const int SOBEL_X[] = {
    1, 0, -1, 2, 0, -2, 1, 0, -1,
};

const int SOBEL_Y[] = {1, 2, 1, 0, 0, 0, -1, -2, -1};

__global__ void V1_conv_kernel(int *in, int n, int m, int *out) {

}

void V1_conv(int *in, int n, int m, int *out) {

}
