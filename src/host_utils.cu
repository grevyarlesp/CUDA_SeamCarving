#include "host_utils.h"
#include <iostream>

using namespace std;

bool check_answer(int *act, int *expected, int n, int ncase) {
  bool b = true;

  for (int i = 0; i < n; ++i) {
    if (act[i] != expected[i]) {
      b = false;
      break;
    }
  }

  if (!b) {
    cout << "Actual = \n";
    for (int i = 0; i < n; ++i) {
      cout << act[i] << ' ';
    }
    cout << '\n';
    cout << "Expected = \n";
    for (int i = 0; i < n; ++i) {
      cout << expected[i] << ' ';
    }
    cout << '\n';
    return false;
  }

  cout << "CORRECT\n";
  return true;
}
