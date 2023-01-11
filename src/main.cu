#include "gpu_utils.h"
#include "host_utils.h"
#include "seam.h"
#include <cstdio>
#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using std::cout;
int main(int argc, char **argv) {

  std::string in_path(argv[1]);

  int target_width = atoi(argv[2]);

  int width, height, channels;
  unsigned char *img = stbi_load(argv[1], &width, &height, &channels, 0);
  if (img == NULL) {
    cout << "Error in loading the image\n";
    exit(1);
  }

  cout << "Loaded image with size of " << width << "x" << height << " channels "
       << channels << '\n';
  cout << "Target Width = " << target_width << '\n';

  unsigned char *out;
  if (target_width < width) {

    cout << "Shrinking to = " << target_width << '\n';
    out = new unsigned char[3 * height * target_width];
    shrink_image(img, height, width, target_width, out);
  } else {
    // enlarge

    out = new unsigned char[3 * height * target_width];

    enlarge_image(img, height, width, target_width, out);
  }
  std::string out_path = add_ext(in_path, std::to_string(target_width));
}
