#ifndef SEAM_H
#define SEAM_H

#include <string>

void shrink_image(unsigned char *img, int height, int width, int target_width,
                  unsigned char *out);

void enlarge_image(unsigned char *img, int height, int width, int target_width,
                   unsigned char *out);
#endif
