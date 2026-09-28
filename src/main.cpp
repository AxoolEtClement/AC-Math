#include "mathLibCPP/Vec3.h"
#include <iostream>

using namespace Maths;
int main() {
    Vec3 v1(1.0f, 2.0f, 3.0f);
    Vec3 v2(4.0f, 5.0f, 6.0f);

    Vec3 v3 = v1 + v2;
    Vec3 v4 = v1 - v2;
    Vec3 v5 = v1 * 2.0f;
    Vec3 v6 = v1 / 2.0f;

    return 0;
}