#include "host_utils.h"
#include <iostream>

using namespace std;

bool check_answer(int *act, int *expected, int n, int ncase=-1) {
  cout << "Case " << ncase << '\n';

  for (int i = 0; i < n; ++i) {
    if (act[i] != expected[i]) {
      cout << "Wrong answer\n";
      cout << "Dfif at " << i << ", got " << act[i] << ' ' << expected[i] << '\n';
      return false;
    }
  }

  cout << "CORRECT\n";
  return true;
}


