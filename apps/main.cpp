#include <iostream>

#include "neuralforge/tensor.hpp"

int main() {
    using namespace neuralforge;

    std::cout << "NeuralForge v0.2 – tensor + matrix multiplication demo\n\n";

    Tensor A{{1.f, 2.f, 3.f},
             {4.f, 5.f, 6.f}};
    Tensor B{{7.f, 8.f},
             {9.f, 10.f},
             {11.f, 12.f}};

    std::cout << "A = " << A.to_string() << "\n";
    std::cout << "B = " << B.to_string() << "\n";

    const auto C = matmul(A, B);
    std::cout << "C = A @ B = " << C.to_string() << "\n";
    std::cout << "  C[0,0] = " << C.at(0, 0) << " (expected 58)\n";
    std::cout << "  C[1,1] = " << C.at(1, 1) << " (expected 154)\n";
    std::cout << "\nDemo finished.\n";
    return 0;
}
