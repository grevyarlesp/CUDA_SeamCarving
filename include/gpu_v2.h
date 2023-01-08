#ifndef GPU_V2_H
#define GPU_V2_H

__global__ void V2_grayscale_kernel(unsigned char *d_in, int num_pixels,
                                    int *out);


double V2_seam(int *in, int height, int width, int *out, int blocksize = 256);
#endif
