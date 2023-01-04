#include "gpu_utils.h"
#include "host.h"
#include <cstdio>
#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using std::cout;

int main(int argc, char **argv) {

  int ver = 0;
  if (argc > 2) {
    ver = atoi(argv[1]);
  }

  if (ver > 0)
    printDeviceInfo();

  if (argc > 2) {
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
