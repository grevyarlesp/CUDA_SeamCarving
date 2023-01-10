#include "seam.h"
#include "gpu_utils.h"
#include "gpu_v1.h"
#include "gpu_v1_2.h"
#include "gpu_v2.h"
#include "host.h"
#include "host_utils.h"
#include <cstdio>
#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"



// 3 channels
// grayscale or RGB.
__global__ void remove_seam_rgb(unsigned char *img, int *d_seam, int height,
                                int width, unsigned char *d_out) {

  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int col = blockDim.x * blockIdx.x + threadIdx.x;

  if (col >= width || row >= height)
    return;

  __shared__ int seam_x;

  if (threadIdx.x == 0)
    seam_x = d_seam[row];
  __syncthreads();

  int pos = row * width + col;

  int target_col = col;

  if (seam_x == col) {
    return;
  }

  if (col > seam_x) {
    target_col = col - 1;
  }

  // reduced width
  int target_pos = row * (width - 1) + target_col;

  d_out[target_pos * 3] = img[pos * 3];
  d_out[target_pos * 3 + 1] = img[pos * 3 + 1];
  d_out[target_pos * 3 + 2] = img[pos * 3 + 2];
}
__global__ void remove_seam_gray(int *gray, int *d_seam, int height, int width,
                                 int *d_out) {

  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int col = blockDim.x * blockIdx.x + threadIdx.x;

  if (col >= width || row >= height)
    return;

  __shared__ int seam_x;

  if (threadIdx.x == 0)
    seam_x = d_seam[row];
  __syncthreads();

  int pos = row * width + col;

  int target_col = col;

  if (seam_x == col) {
    // remover
    return;
  }

  if (col > seam_x) {
    target_col = col - 1;
  }

  int target_pos = row * (width - 1) + target_col;

  d_out[target_pos] = gray[pos];
}

void shrink_image(unsigned char *img, int height, int width, int target_width,
                  std::string path) {

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

  int *d_seam;
  CHECK(cudaMalloc(&d_seam, height * sizeof(int)));

  for (int cur_width = width; cur_width > target_width; --cur_width) {

#ifdef SHRINK_DEBUG
    std::cerr << cur_width << '\n';
#endif

    int reduced_width = cur_width - 1;
#ifdef SHRINK_DEBUG
    std::cerr << "Gray hash        :" << calc_hash(gray, height * cur_width)
              << '\n';
#endif

    // remove 1 for cur_width
    int *emap = new int[height * cur_width];

#ifdef USE_HOST_FOR_CONV
    host_sobel_conv(gray, height, cur_width, emap);
#else
    V1_conv(gray, height, cur_width, emap);
#endif

#ifdef SHRINK_DEBUG

    std::cerr << "Energy map hash GPU:" << calc_hash(emap, height * cur_width) << '\n';
    // Debugging to find the wrong in hash
    {
      int *emap = new int[height * cur_width];

      host_sobel_conv(gray, height, cur_width, emap);

      std::cerr << "Energy map hash HOST:" << calc_hash(emap, height * cur_width) << '\n';

    }


#endif

    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());

    // seam,
    int *seam = new int[height];
    V1_2_seam(emap, height, cur_width, seam, 512);

    CHECK(
        cudaMemcpy(d_seam, seam, height * sizeof(int), cudaMemcpyHostToDevice));

#ifdef SHRINK_DEBUG
    {
      std::cerr << "Seam hash       :" << calc_hash(seam, height) << '\n';

      // unsigned char *out_seam = new unsigned char[height * cur_width * 3];
      // std::string out_path =
      //     add_ext(path, std::to_string(target_width) + "_" +
      //                       std::to_string(cur_width) + "_seam");
      //
      // CHECK(cudaMemcpy(out_seam, d_in, 3 * height * cur_width,
      //                  cudaMemcpyDeviceToHost));
      //
      // host_highlight_seam(out_seam, height, cur_width, seam);
      //
      // stbi_write_png(out_path.c_str(), cur_width, height, 3, out_seam,
      //                cur_width * 3);
      //
      // delete[] out_seam;
    };
#endif

    int *d_gray_rz;
    CHECK(cudaMalloc(&d_gray_rz, height * reduced_width * sizeof(int)));

    // resizing gray
    dim3 block_size(256);
    dim3 grid_size((cur_width - 1) / block_size.x + 1, height);
    remove_seam_gray<<<grid_size, block_size>>>(d_gray, d_seam, height,
                                                cur_width, d_gray_rz);

    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());

    CHECK(cudaFree(d_gray));
    d_gray = d_gray_rz;

    // copy back to host
    CHECK(cudaMemcpy(gray, d_gray_rz, height * reduced_width * sizeof(int),
                     cudaMemcpyDeviceToHost));
#ifdef SHRINK_DEBUG
    std::cerr << "Reduced gray hash " << calc_hash(gray, height * reduced_width)
              << '\n';
#endif

    unsigned char *d_out;
    CHECK(
        cudaMalloc(&d_out, height * reduced_width * sizeof(unsigned char) * 3));

    remove_seam_rgb<<<grid_size, block_size>>>(d_in, d_seam, height, cur_width,
                                               d_out);

    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());

#ifdef SHRINK_DEBUG
    {
        // std::string out_path = add_ext(path, std::to_string(target_width) +
        // "_" +
        //                                          std::to_string(reduced_width));
        //
        // out = new unsigned char[3 * height * reduced_width];
        // CHECK(cudaMemcpy(out, d_in, 3 * height * reduced_width,
        //                  cudaMemcpyDeviceToHost));
        // stbi_write_png(out_path.c_str(), reduced_width, height, 3, out,
        //                cur_width * 3);
        //
        // delete[] out;
        // unsigned char *tmp_gray = to_uchar(gray, height * reduced_width);
        // out_path = add_ext(path, std::to_string(target_width) + "_" +
        //                              std::to_string(reduced_width) +
        //                              "_gray");
        //
        // stbi_write_png(out_path.c_str(), reduced_width, height, 1, tmp_gray,
        //                reduced_width * 1);
        //
        // delete[] tmp_gray;
    };
#endif

    CHECK(cudaFree(d_in));
    d_in = d_out;

    delete[] emap;

    delete[] seam;
  }

  std::cerr << "Copying\n";
  unsigned char *out = new unsigned char[3 * height * target_width];

  CHECK(cudaMemcpy(out, d_in, 3 * height * target_width * sizeof(unsigned char),
                   cudaMemcpyDeviceToHost));

  std::string out_path = add_ext(path, std::to_string(target_width));
  stbi_write_png(out_path.c_str(), target_width, height, 3, out,
                 target_width * 3);
  std::cout << "Done writing to " << out_path << '\n';

  CHECK(cudaFree(d_in));
  CHECK(cudaFree(d_seam));
  CHECK(cudaFree(d_gray));
  delete[] gray;
  delete[] out;
}