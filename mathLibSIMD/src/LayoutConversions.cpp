#include "Vec4.hpp"
#include "Vec4SOA.hpp"
#include <immintrin.h>

namespace simd {

void ToSOA(const Vec4* input, Vec4SOA* output, std::size_t count)
{
    if (!input || !output || count == 0) return;

    for (std::size_t i = 0; i < count; i += 4)
    {
        __m128 x_res = _mm_set_ps(
            (i+3 < count) ? input[i+3]._data.m128_f32[0] : 0.f,
            (i+2 < count) ? input[i+2]._data.m128_f32[0] : 0.f,
            (i+1 < count) ? input[i+1]._data.m128_f32[0] : 0.f,
            input[i]._data.m128_f32[0]
        );
        __m128 y_res = _mm_set_ps(
            (i+3 < count) ? input[i+3]._data.m128_f32[1] : 0.f,
            (i+2 < count) ? input[i+2]._data.m128_f32[1] : 0.f,
            (i+1 < count) ? input[i+1]._data.m128_f32[1] : 0.f,
            input[i]._data.m128_f32[1]
        );
        __m128 z_res = _mm_set_ps(
            (i+3 < count) ? input[i+3]._data.m128_f32[2] : 0.f,
            (i+2 < count) ? input[i+2]._data.m128_f32[2] : 0.f,
            (i+1 < count) ? input[i+1]._data.m128_f32[2] : 0.f,
            input[i]._data.m128_f32[2]
        );
        __m128 w_res = _mm_set_ps(
            (i+3 < count) ? input[i+3]._data.m128_f32[3] : 0.f,
            (i+2 < count) ? input[i+2]._data.m128_f32[3] : 0.f,
            (i+1 < count) ? input[i+1]._data.m128_f32[3] : 0.f,
            input[i]._data.m128_f32[3]
        );

        output[i / 4] = Vec4SOA(x_res, y_res, z_res, w_res);
    }
}

void ToAOS(const Vec4SOA* input, Vec4* output, std::size_t count)
{
    if (!input || !output || count == 0) return;

    for (std::size_t i = 0; i < count; i += 4)
    {
        __m128 x = input[i / 4].getX();
        __m128 y = input[i / 4].getY();
        __m128 z = input[i / 4].getZ();
        __m128 w = input[i / 4].getW();

        for (std::size_t j = 0; j < 4 && (i + j) < count; ++j)
        {
            output[i + j] = Vec4(
                ((float*)&x)[j],
                ((float*)&y)[j],
                ((float*)&z)[j],
                ((float*)&w)[j]
            );
        }
    }
}

}
