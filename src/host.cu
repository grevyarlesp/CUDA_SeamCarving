#include <stdio.h>


const int SOBEL_X[] = {
  1, 0, -1, 
  2, 0, -2, 
  1, 0, -1,
};

const int SOBEL_Y[] = {
  1, 2, 1,
  0 ,0 ,0,
  -1, -2, -1
};


/*
Input: n * m grayscale image
Output: n * m energy map
*/
int* sobel_conv(int *in, int n, int m) {
  int *out =  (int*) malloc(n * m * sizeof(int));

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      
    }
  }
  
}

/*
Input: n * m * 3 array
Output: Processed seam
   */
void host_generate_seam(int *in, int n, int m, int *out) {
  out =  (int*) malloc(3 * n * m * sizeof(int));

}


