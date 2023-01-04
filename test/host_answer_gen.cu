// Generating answers for big test cases
#include "host.h"
#include <iostream>
#include <string>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using std::cout;
using std::string;

string add_ext(const string &in_path, string to_add) {

  size_t lastindex = in_path.find_last_of(".");

  to_add = "_" + to_add;

  string out = in_path;
  out.insert(lastindex, to_add);
  return out;
}

/*

*/

void seam(string in_path) {
  int width, height, channels;

  cout << "Reading from " << in_path << '\n';
  ;
  unsigned char *img =
      stbi_load(in_path.c_str(), &width, &height, &channels, 0);
  assert(channels == 3);

  cout << "Channels: " << ' ' << channels << " width " << width << " height "
       << height << '\n';

  string out_path = add_ext(in_path, "seam");

  int *seam = new int[height];
  host_full(img, height, width, seam);

  cout << "Done, writing... " << out_path << '\n';

  stbi_write_png(out_path.c_str(), width, height, 3, img, 100);

  free(img);
}

int main(int argc, char **argv) {
  if (argc < 2)
    return 0;
  string file_path(argv[1]);

  seam(file_path);
}
