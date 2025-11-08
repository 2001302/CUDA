#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

// CUDA 커널: 벡터 덧셈
__global__ void vectorAdd(float *A, float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

// 호스트 함수: CUDA 에러 체크
void checkCudaError(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(error));
        exit(1);
    }
}

#define CHECK_CUDA_ERROR(err) checkCudaError(err, __FILE__, __LINE__)

int main() {
    const int N = 1000000;  // 벡터 크기
    const size_t size = N * sizeof(float);

    // 호스트 메모리 할당
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    // 초기화
    for (int i = 0; i < N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // 디바이스 메모리 할당
    float *d_A, *d_B, *d_C;
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_A, size));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_B, size));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_C, size));

    // 호스트에서 디바이스로 데이터 복사
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // 커널 실행 설정
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // 커널 실행
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // 디바이스에서 호스트로 결과 복사
    CHECK_CUDA_ERROR(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // 결과 검증
    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (fabs(h_C[i] - 3.0f) > 1e-5) {
            printf("Error at index %d: expected 3.0, got %f\n", i, h_C[i]);
            correct = false;
            break;
        }
    }

    if (correct) {
        printf("Vector addition completed successfully!\n");
        printf("First 10 results: ");
        for (int i = 0; i < 10; i++) {
            printf("%.1f ", h_C[i]);
        }
        printf("\n");
    } else {
        printf("Vector addition failed!\n");
    }

    // 메모리 해제
    free(h_A);
    free(h_B);
    free(h_C);
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_C));

    return 0;
}


