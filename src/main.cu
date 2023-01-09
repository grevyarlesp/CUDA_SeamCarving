#include "gpu_utils.h"
#include <cstdio>
#include <iostream>
#include "gpu_v1.h"
#include "gpu_v1_2.h"
#include "gpu_v2.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using std::cout;

// 3 channels
// grayscale or RGB.
__global__ void remove_seam(unsigned char *img, int *d_seam, int height,
                            int width, unsigned char *d_out, int channels) {

  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int col = blockDim.y * blockIdx.y + threadIdx.y;

  if (row >= height || col >= width)
    return;

  int pos = row * width + col;

  int seam_x = d_seam[col];

  int target_col = col;

  if (seam_x == col) {
    return;
  }

  if (seam_x > col) {
    target_col -= 1;
  }

  int target_pos = row * (width - 1) + target_col;

  if (channels == 1) {
    d_out[target_pos] = img[pos];
  } else {
    d_out[target_pos] = img[pos];
    d_out[target_pos + 1] = img[pos + 1];
    d_out[target_pos + 2] = img[pos + 2];
  }
}
__global__ void remove_seam_2(int *gray, int *d_seam, int height,
                            int width, int *d_out) {

  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int col = blockDim.y * blockIdx.y + threadIdx.y;

  if (row >= height || col >= width)
    return;

  int pos = row * width + col;

  int seam_x = d_seam[col];

  int target_col = col;

  if (seam_x == col) {
    return;
  }

  if (seam_x > col) {
    target_col -= 1;
  }

  int target_pos = row * (width - 1) + target_col;

  d_out[target_pos] = gray[pos];
}



__global__ void add_seam(unsigned char *img, int *d_seam, int height, int width,
                         unsigned char *d_out, int channels) {
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int col = blockDim.y * blockIdx.y + threadIdx.y;

  if (row >= height || col >= width)
    return;

  int pos = row * width + col;

  int seam_x = d_seam[col];

  int target_col = col;

  if (seam_x == col) {
    return;
  }

  if (seam_x > col) {
    target_col -= 1;
  }

  int target_pos = row * (width - 1) + target_col;

  if (channels == 1) {
    d_out[target_pos] = img[pos];
  } else {
    d_out[target_pos] = img[pos];
    d_out[target_pos + 1] = img[pos + 1];
    d_out[target_pos + 2] = img[pos + 2];
  }
}

void resize_image(unsigned char *img, int height, int width, int target_width) {
  unsigned char *d_in;

  CHECK(cudaMalloc(&d_in, sizeof(unsigned char) * 3 * height * width));

  CHECK(cudaMemcpy(d_in, img, sizeof(unsigned char) * 3 * height * width,
                   cudaMemcpyHostToDevice));

  int *d_gray;
  CHECK(cudaMalloc(&d_gray, sizeof(int) * height * width));

  dim3 block_size(1024);
  dim3 grid_size((height * width - 1) / block_size.x + 1);
  V2_grayscale_kernel<<<grid_size, block_size>>>(d_in, height * width, d_gray);

  int *gray = new int[height * width];

  CHECK(cudaMemcpy(gray, d_gray, sizeof(int) * height * width,
                   cudaMemcpyDeviceToHost));


  if (width > target_width) {
    int to_remove = width - target_width;

    for (int cur_width = width; cur_width > target_width; --cur_width) {
      int *emap = new int[height * cur_width];
      V1_conv(gray, height, cur_width, emap);
      int *seam = new int[height];
      V1_2_seam(emap, height, cur_width, seam, 512);
      int *d_seam;
      CHECK(cudaMalloc(&d_seam, height * sizeof(int)));

      int *d_gray_out;
      CHECK(cudaMalloc(&d_gray_out, height * (cur_width - 1) * sizeof(int)));
      remove_seam_2<<<(height - 1) / 32 + 1, (cur_width - 1) / 32 + 1>>>(d_gray, d_seam, height, cur_width, d_gray_out);

      CHECK(cudaFree(gray));
      gray = d_gray_out;

      unsigned char  *d_in_out;
      CHECK(cudaMalloc(&d_in_out, height * (cur_width - 1) * sizeof(unsigned char) * 3));
      remove_seam<<<(height - 1) / 32 + 1, (cur_width - 1) / 32 + 1>>>(d_in, d_seam, height, cur_width, d_in_out, 3);
      CHECK(cudaFree(d_in));
      d_in = d_in_out;
      delete[] emap;
    }

    CHECK(cudaFree(gray));

  } else if (width < target_width) {
  }
}

int main(int argc, char **argv) {

  if (argc > 1) {
    int width, height, channels;
    unsigned char *img = stbi_load(argv[2], &width, &height, &channels, 0);
    if (img == NULL) {
      cout << "Error in loading the image\n";
      exit(1);
    }

    cout << "Loaded image with size of " << width << "x" << height
         << " channels " << channels << '\n';
  }
}
