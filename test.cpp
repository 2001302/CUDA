#include <iostream>
#include <vector>
#include <string>

int main() {
    std::cout << "Hello, World!" << std::endl;
    
    // 벡터 예제
    std::vector<int> numbers = {1, 2, 3, 4, 5};
    std::cout << "Numbers: ";
    for (const auto& num : numbers) {
        std::cout << num << " ";
    }
    std::cout << std::endl;
    
    // 문자열 예제
    std::string message = "C++ Sample Code";
    std::cout << "Message: " << message << std::endl;
    
    // 간단한 계산
    int sum = 0;
    for (const auto& num : numbers) {
        sum += num;
    }
    std::cout << "Sum of numbers: " << sum << std::endl;
    
    return 0;
}

