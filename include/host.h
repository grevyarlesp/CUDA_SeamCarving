#ifndef __HOST_H__
#define __HOST_H__

void host_sobel_conv(int *in, int n, int m, int* out);

void host_dp_seam(int *in, int n, int m, int *out);

void host_full(unsigned char *in, int height, int width, int *seam);

#endif
