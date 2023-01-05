// Generating answers for big test cases
// Using the GPU to generate for certain parts
#include "host.h"
#include "gpu_utils.h"
#include "host_utils.h"
#include "gpu_v1.h"

#include <iostream>
#include <string>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using std::cout;
using std::string;


/*
   Output result for each steps
   */

void test_v1_seam(string in_path, int blocksize=256) {
  int width, height, channels;

  cout << "Reading from " << in_path << '\n';

  unsigned char *img =
      stbi_load(in_path.c_str(), &width, &height, &channels, 3);

  assert(channels == 3);

  cout << "Channels "  << channels << " width " << width << " height "
       << height << '\n';

  string out_path = add_ext(in_path, "seam_v1");

  unsigned char *d_in;
  CHECK(cudaMalloc(&d_in, sizeof(unsigned char) * 3 * height * width));

  int *d_gray;
  CHECK(cudaMalloc(&d_gray, sizeof(char) * height * width));

  dim3 block_size(blocksize, blocksize);
  dim3 grid_size((height - 1) / blocksize + 1, (width - 1) / blocksize + 1);

  V1_grayscale_kernel<<<grid_size, block_size>>>(d_in, height, width, d_gray);



  delete[] img;
}



int main(int argc, char **argv) {
  if (argc < 2)
    return 0;
  string file_path(argv[1]);

  // grayscale(file_path);

  test_v1_seam(file_path);
}
