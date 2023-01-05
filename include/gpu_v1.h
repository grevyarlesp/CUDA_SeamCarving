#ifndef GPU_V1_H
#define GPU_V1_H 


void V1_conv(int *in, int w, int h, bool sobelx, int *out);
double V1_seam(int *in, int n, int m, int *out, int blocksize  = 256);
void V1_grayscale(unsigned char *in, int height, int width, int channels, int *out);


void v1_in_to_seam(unsigned char *in, int height, int width, char *out,
                   int blocksize = 256);

#endif /* GPU_V1_H */

