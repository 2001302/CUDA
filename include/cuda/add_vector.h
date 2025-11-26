#ifndef ADD_VECTOR_H
#define ADD_VECTOR_H

#ifdef __cplusplus
extern "C" {
#endif

// CUDA function to add two vectors
void add_vectors_cuda(float* a, float* b, float* c, int n);

#ifdef __cplusplus
}
#endif

#endif // ADD_VECTOR_H

