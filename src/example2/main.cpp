#include <iostream>
#include <vector>
#include "cuda_common/cuda_utils.h"

int main() {
    const int n = 5;
    std::vector<float> a(n, 1.0f);
    std::vector<float> b(n, 2.0f);
    std::vector<float> c(n, 0.0f);

    std::cout << "Before CUDA computation:" << std::endl;
    std::cout << "a = ";
    for (int i = 0; i < n; ++i) {
        std::cout << a[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "b = ";
    for (int i = 0; i < n; ++i) {
        std::cout << b[i] << " ";
    }
    std::cout << std::endl;

    // Call CUDA function
    add_vectors_cuda(a.data(), b.data(), c.data(), n);

    std::cout << "\nAfter CUDA computation (a + b):" << std::endl;
    std::cout << "c = ";
    for (int i = 0; i < n; ++i) {
        std::cout << c[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}

