#ifndef HOST_UTILS_H
#define HOST_UTILS_H

#include <string>
bool check_answer(int *act, int *expected, int n, int ncase=-1);

std::string add_ext(const std::string &in_path, std::string to_add);

unsigned char *to_uchar(int *in, int n);

#endif
