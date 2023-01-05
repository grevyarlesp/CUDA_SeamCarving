#ifndef __HOST_H__
#define __HOST_H__

void host_sobel_conv(int *in, int n, int m, int *out);

void host_dp_seam(int *in, int n, int m, int *out);

void host_to_grayscale(unsigned char *in, int height, int width, int *out);

void host_full(unsigned char *in, int height, int width, int *seam);

__global__ void V1_grayscale_kernel(unsigned char *d_in, int height, int width,
                                    int *out);

#endif
