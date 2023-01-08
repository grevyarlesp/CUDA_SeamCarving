// Generating answers for big test cases
// Using the GPU to generate for certain parts
#include "gpu_utils.h"
#include "gpu_v1.h"
#include "gpu_v1_1.h"
#include "gpu_v1_2.h"
#include "gpu_v1_3.h"
#include "gpu_v2.h"
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

/*
   Output result for each steps
   */

void test_v1_seam(string in_path, int minorver, int blocksize, bool write_to_file = false) {
  int width, height, channels;

  cout << "Reading from " << in_path << '\n';

  unsigned char *img =
      stbi_load(in_path.c_str(), &width, &height, &channels, 3);

  assert(channels == 3);

  GpuTimer timer;

  cout << "Channels " << channels << " width " << width << " height " << height
       << '\n';

  timer.Start();

  unsigned char *d_in;

  CHECK(cudaMalloc(&d_in, sizeof(unsigned char) * 3 * height * width));

  CHECK(cudaMemcpy(d_in, img, sizeof(unsigned char) * 3 * height * width,
                   cudaMemcpyHostToDevice));

  int *d_gray;
  CHECK(cudaMalloc(&d_gray, sizeof(int) * height * width));

  dim3 block_size(1024);
  dim3 grid_size((height * width - 1) / block_size.x + 1);
  V2_grayscale_kernel<<<grid_size, block_size>>>(d_in, height * width, d_gray);

  cout << "Channels " << channels << " width " << width << " height " << height
       << '\n';

  cout << "Block size" << ' ' << blocksize << '\n';

  CHECK(cudaDeviceSynchronize());
  CHECK(cudaGetLastError());

  int *gray = new int[height * width];

  CHECK(cudaMemcpy(gray, d_gray, sizeof(int) * height * width,
                   cudaMemcpyDeviceToHost));

  if (write_to_file) {
    string out_path = add_ext(in_path, "gray_v1");
    unsigned char *ugray = to_uchar(gray, height * width);
    stbi_write_png(out_path.c_str(), width, height, 1, ugray, width * 1);
    delete[] ugray;
  }

  int *emap = new int[height * width];

  V1_conv(gray, height, width, emap);

  int *seam = new int[height];

  double tseam;
  if (minorver == 0) {
    cout << "Ver 1.0\n";

    tseam = V1_seam(emap, height, width, seam, blocksize);
  } else if (minorver == 1) {

    cout << "Ver 1.1\n";
    tseam = V1_1_seam(emap, height, width, seam, blocksize);

  }  else if (minorver == 2) {

    cout << "Ver 1.2\n";
    tseam = V1_2_seam(emap, height, width, seam, blocksize);
    
  } else if (minorver == 3) {

    cout << "Ver 1.3\n";
    tseam = V1_3_seam(emap, height, width, seam, blocksize);

  }

  cout << "Seam time = "  << tseam << '\n';

  timer.Stop();

  host_highlight_seam(img, height, width, seam);

  string out_path = add_ext(in_path, "seam_v1");

  stbi_write_png(out_path.c_str(), width, height, 3, img, width * 3);

  cout << "Complete in " << timer.Elapsed() << '\n';

}

int main(int argc, char **argv) {
  if (argc < 2)
    return 0;
  // printDeviceInfo();

  string file_path(argv[1]);

  int minorver = 0;

  if (argc > 2) 
    minorver = atoi(argv[2]);

  int blocksize = 256;
  if (argc > 3) 
    blocksize = atoi(argv[3]);
    

  // grayscale(file_path);

  test_v1_seam(file_path, minorver, blocksize);
}
