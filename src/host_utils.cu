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

string add_ext(const string &in_path, string to_add) {

  size_t lastindex = in_path.find_last_of(".");

  to_add = "_" + to_add;

  string out = in_path;
  out.insert(lastindex, to_add);
  return out;
}


unsigned char *to_uchar(int *in, int n) {
  unsigned char *ans = new unsigned char[n];
  for (int i = 0; i < n; ++i) {
    ans[i] = in[i];
  }
  return ans;
}


