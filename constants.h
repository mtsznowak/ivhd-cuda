#pragma once

#define shrink_near 0.0
#define shrink_far 1.0
#define sammon_k 1.0
#define sammon_m 2.0
#define sammon_w 0.0
#define a_factor 0.990545
#define b_factor 0.000200945
#define w_near 1.0
#define w_random 0.01
#define w_far 1.0
#define only2d true

inline bool cudaCheck(cudaError_t ret, const char *fileName, unsigned int lineNo) {
    if (ret != cudaSuccess) {
        std::cout << "CUDA error in " << fileName << ":" << lineNo << std::endl
                  << "\t (" << cudaGetErrorString(ret) << ")" << std::endl;
    }

    return ret != cudaSuccess;
}

#define cuCall(err)       cudaCheck(err, __FILE__, __LINE__)
