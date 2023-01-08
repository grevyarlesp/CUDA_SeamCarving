// Generating answers for big test cases
#include "host.h"
#include "host_utils.h"
#include "gpu_utils.h"
#include <iostream>
#include <string>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using std::cout;
using std::string;

void grayscale(string in_path) {
  int width, height, channels;

  cout << "Generating grayscale" << in_path << '\n';
  cout << "Reading from " << in_path << '\n';

  unsigned char *img =
      stbi_load(in_path.c_str(), &width, &height, &channels, 3);
  cout << "Channels: " << ' ' << channels << " width " << width << " height "
       << height << '\n';


  assert(channels == 3);

  int *gray = new int[height * width];

  host_to_grayscale(img, height, width, gray);

  unsigned char * out = new unsigned char[height * width];

  for (int i = 0; i < height * width; ++i) 
    out[i] = gray[i];

  string out_path = add_ext(in_path, "gray");

  stbi_write_png(out_path.c_str(), width, height, 1, out, width * 1);

  int *emap = new int[height * width];
  host_sobel_conv(gray, height, width, emap);

  int mx = 0;
  for (int i = 0; i < height * width; ++i) {
    mx = max(mx, emap[i]);
  }

  out_path = add_ext(in_path, "emap");
  for (int i = 0; i < height * width; ++i) {
    float x = 1.0 * emap[i] / mx * 255;
    out[i] = (unsigned char)(x);
  }

  stbi_write_png(out_path.c_str(), width, height, 1, out, width * 1);

}

void seam(string in_path) {

  
  int width, height, channels;

  cout << "Reading from " << in_path << '\n';

  unsigned char *img =
      stbi_load(in_path.c_str(), &width, &height, &channels, 3);
  assert(channels == 3);

  cout << "Channels "  << channels << " width " << width << " height "
       << height << '\n';

  string out_path = add_ext(in_path, "seam");

  int *seam = new int[height];
  GpuTimer timer;
  timer.Start();
  host_full(img, height, width, seam);
  timer.Stop();

  cout << "Compute time:  " << timer.Elapsed() << '\n';

  stbi_write_png(out_path.c_str(), width, height, 3, img, width * 3);



  delete[] seam;
  delete[] img;
}



int main(int argc, char **argv) {
  if (argc < 2)
    return 0;
  string file_path(argv[1]);

  //grayscale(file_path);
  seam(file_path);
}
