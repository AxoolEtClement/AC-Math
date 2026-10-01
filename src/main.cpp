#include "mathLibCPP/Vec3.h"
#include "mathLibSIMD/include/Vec4.hpp"
#include <iostream>
#include <string>

using namespace simd;
int main() {
    Vec4 v1(1.0f, 2.0f, 3.0f, 4.0f);
    Vec4 v2(1.0f, 2.0f, 3.0f, 4.0f);

    
    std::cout << "v1dotv2: " << v1.Dot(v2) << std::endl;

    return 0;
}