#ifndef GPU_V1_H
#define GPU_V1_H 


void V1_conv(int *in, int n, int m, int *out);
double V1_seam(int *in, int n, int m, int *out, int blocksize  = 256);

#endif /* GPU_V1_H */

