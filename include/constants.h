#pragma once
#include <iostream>
using namespace std;

//#define a_factor 0.990545
//#define b_factor 0.000200945
#define a_factor 0.9
#define b_factor 0.002
#define w_random 0.01

inline bool cudaCheck(cudaError_t ret, const char *fileName,
                      unsigned int lineNo) {
  if (ret != cudaSuccess) {
    std::cerr << "CUDA error in " << fileName << ":" << lineNo << std::endl
              << "\t (" << cudaGetErrorString(ret) << ")" << std::endl;
    exit(-1);
  }

  return ret != cudaSuccess;
}

#define cuCall(err) cudaCheck(err, __FILE__, __LINE__)
