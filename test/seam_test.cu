#include "host.h"
#include "host_utils.h"
#include <vector>
#define X first
#define Y second

using namespace std;

/*
Input: Energy map
Output: Seam
   */

vector<pair<int, int>> dat_sz = {
  {4, 6}, 
  {5, 5}, 
  {10, 5}};

vector<vector<int> > dat = {
  {
    255, 200, 19, 20, 18, 17,
    255, 1  , 19, 20, 18, 17,
    255, 200, 9 , 20, 18, 17,
    255, 200, 0 , 20, 18, 17,
  }, 
  {
    1000, 19, 20, 30, 500, 20,
    1000, 19, 0, 30, 500, 20,
    1000, 19, 20, 19, 500, 20,
    1000, 19, 2, 30, 500, 20,
    1000, 19, 0, 30, 500, 20
  }
};

vector< vector<int>  > ans = {
  {2, 1, 2, 2},
  {1, 2, 1, 2, 2}
};

void host_test() {
  for (size_t i = 0; i < dat.size(); ++i) {
    vector<int>& V = dat[i];

    pair<int, int> s = dat_sz[i];
    int *act = new int[s.X];

    host_dp_seam(V.data(), s.X, s.Y, act);

    check_answer(act, ans[i].data(), s.X, i);
    delete[] act;
  }
}



int main() {
  host_test();
  return 0;

}
