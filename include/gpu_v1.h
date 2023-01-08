#ifndef GPU_V1_H
#define GPU_V1_H 


void V1_conv(int *in, int w, int h, int *out, int block_size = 32);
double V1_seam(int *in, int n, int m, int *out, int blocksize  = 256);

// 1024 max
void V1_grayscale(unsigned char *in, int height, int width, int *out, int blocksize = 32);
int V1_min(int* in, int* ind, int n, int block_size=32);

__global__ void V1_grayscale_kernel(unsigned char *d_in, int height, int width,
                                    int *out);

#endif /* GPU_V1_H */

