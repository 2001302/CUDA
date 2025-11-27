#include "algorithm/algorithm.h"
#include "cuda/add_vector.h"
#include "common/logger.h"
#include <stdexcept>

namespace algorithm {

std::vector<float> add_vectors(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        common::Logger::get_instance().error("Vector sizes do not match");
        throw std::invalid_argument("Vector sizes must be equal");
    }
    
    if (a.empty()) {
        common::Logger::get_instance().warning("Empty vectors provided");
        return std::vector<float>();
    }
    
    common::Logger::get_instance().info("Adding vectors of size " + std::to_string(a.size()));
    
    std::vector<float> result(a.size());
    
    // Convert vectors to raw pointers for CUDA
    std::vector<float> a_copy = a;
    std::vector<float> b_copy = b;
    
    add_vectors_cuda(a_copy.data(), b_copy.data(), result.data(), static_cast<int>(a.size()));
    
    common::Logger::get_instance().info("Vector addition completed");
    
    return result;
}

} // namespace algorithm

