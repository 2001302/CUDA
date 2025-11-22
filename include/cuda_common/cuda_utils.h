#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#ifdef __cplusplus
extern "C" {
#endif

// CUDA function declarations
void add_vectors_cuda(float* a, float* b, float* c, int n);

#ifdef __cplusplus
}
#endif

#endif // CUDA_UTILS_H

