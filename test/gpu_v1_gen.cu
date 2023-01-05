// Generating answers for big test cases
// Using the GPU to generate for certain parts
#include "gpu_utils.h"
#include "gpu_v1.h"
#include "host.h"
#include "host_utils.h"

#include <iostream>
#include <string>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using std::cout;
using std::string;

unsigned char *to_uchar(int *in, int n) {
  unsigned char *ans = new unsigned char[n];
  for (int i = 0; i < n; ++i) {
    ans[i] = in[i];
  }
  return ans;
}

/*
   Output result for each steps
   */

void test_v1_seam(string in_path, int blocksize = 256) {
  int width, height, channels;

  cout << "Reading from " << in_path << '\n';

  unsigned char *img =
      stbi_load(in_path.c_str(), &width, &height, &channels, 3);

  assert(channels == 3);

  cout << "Channels " << channels << " width " << width << " height " << height
       << '\n';


  unsigned char *d_in;

  CHECK(cudaMalloc(&d_in, sizeof(unsigned char) * 3 * height * width));

  int *d_gray;
  CHECK(cudaMalloc(&d_gray, sizeof(int) * height * width));

  dim3 block_size(blocksize, blocksize);
  dim3 grid_size((width - 1) / blocksize + 1, (height - 1) / blocksize + 1);
  V1_grayscale_kernel<<<grid_size, block_size>>>(d_in, height, width, d_gray);

  int *gray = new int[height * width];

  CHECK(cudaMemcpy(gray, d_gray, sizeof(int) * height * width, cudaMemcpyDeviceToHost));
  

  string out_path = add_ext(in_path, "gray_v1");

  unsigned char *ugray = to_uchar(gray, height * width);
  stbi_write_png(out_path.c_str(), width, height, 1, ugray, width * 1);


  int *emap = new int[height * width];

  // TODO: replace conv kernel here
  host_sobel_conv(gray, height, width, emap);


  int *seam = new int[height];
  V1_seam(emap, height, width, seam);


  
  host_highlight_seam(img, height, width, seam);
  out_path = add_ext(in_path, "seam_v1");

  
  stbi_write_png(out_path.c_str(), width, height, 3, img, width * 3);
}

int main(int argc, char **argv) {
  if (argc < 2)
    return 0;
  printDeviceInfo();

  string file_path(argv[1]);

  // grayscale(file_path);

  test_v1_seam(file_path);
}
