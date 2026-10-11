#include "Vec3f.hpp"
#include "Mat4x4.hpp"
#include <cmath>
#include <immintrin.h>

namespace simd {

namespace
{
    __m128 LoadComponent(const Vec3f* values, std::size_t offset,
                         std::size_t count, int component)
    {
        const auto componentAt = [values, offset, count, component](std::size_t lane) {
            if (offset + lane >= count)
                return 0.0f;

            const Vec3f& value = values[offset + lane];
            return component == 0 ? value.getX()
                : component == 1 ? value.getY()
                : value.getZ();
        };

        return _mm_set_ps(componentAt(3), componentAt(2),
                          componentAt(1), componentAt(0));
    }

    void StoreComponents(__m128 x, __m128 y, __m128 z,
                         Vec3f* output, std::size_t offset, std::size_t count)
    {
        alignas(16) float xs[4];
        alignas(16) float ys[4];
        alignas(16) float zs[4];
        _mm_store_ps(xs, x);
        _mm_store_ps(ys, y);
        _mm_store_ps(zs, z);

        for (std::size_t lane = 0; lane < 4 && offset + lane < count; ++lane)
            output[offset + lane] = Vec3f{xs[lane], ys[lane], zs[lane]};
    }
}

void BatchDotReference(const Vec3f* a, const Vec3f* b, float* results,
                       std::size_t count)
{
    if (!a || !b || !results || count == 0) return;
    for (std::size_t i = 0; i < count; ++i)
        results[i] = a[i].getX() * b[i].getX()
            + a[i].getY() * b[i].getY()
            + a[i].getZ() * b[i].getZ();
}

void BatchNormalizeReference(const Vec3f* input, Vec3f* output,
                             std::size_t count)
{
    if (!input || !output || count == 0) return;
    for (std::size_t i = 0; i < count; ++i)
    {
        const float lengthSquared = input[i].MagnitudeSquared();
        output[i] = lengthSquared > 1e-12f
            ? input[i] * (1.0f / std::sqrt(lengthSquared))
            : Vec3f::Zero;
    }
}

void BatchTransformPointReference(const Matrix4x4& mat, const Vec3f* input,
                                  Vec3f* output, std::size_t count)
{
    if (!input || !output || count == 0) return;
    for (std::size_t i = 0; i < count; ++i)
        output[i] = mat.TransformPoint(input[i]);
}

void BatchDot(const Vec3f* a, const Vec3f* b, float* results, std::size_t count)
{
    if (!a || !b || !results || count == 0) return;

    std::size_t i = 0;
    for (; i + 3 < count; i += 4)
    {
        const __m128 ax = LoadComponent(a, i, count, 0);
        const __m128 ay = LoadComponent(a, i, count, 1);
        const __m128 az = LoadComponent(a, i, count, 2);
        const __m128 bx = LoadComponent(b, i, count, 0);
        const __m128 by = LoadComponent(b, i, count, 1);
        const __m128 bz = LoadComponent(b, i, count, 2);
        const __m128 result = _mm_add_ps(
            _mm_add_ps(_mm_mul_ps(ax, bx), _mm_mul_ps(ay, by)),
            _mm_mul_ps(az, bz));
        _mm_storeu_ps(results + i, result);
    }

    for (; i < count; ++i)
        results[i] = a[i].Dot(b[i]);
}

void BatchNormalize(const Vec3f* input, Vec3f* output, std::size_t count)
{
    if (!input || !output || count == 0) return;

    std::size_t i = 0;
    for (; i + 3 < count; i += 4)
    {
        const __m128 x = LoadComponent(input, i, count, 0);
        const __m128 y = LoadComponent(input, i, count, 1);
        const __m128 z = LoadComponent(input, i, count, 2);
        const __m128 lengthSquared = _mm_add_ps(
            _mm_add_ps(_mm_mul_ps(x, x), _mm_mul_ps(y, y)),
            _mm_mul_ps(z, z));
        const __m128 valid = _mm_cmpgt_ps(lengthSquared, _mm_set1_ps(1e-12f));
        const __m128 inverseLength = _mm_div_ps(
            _mm_set1_ps(1.0f), _mm_sqrt_ps(lengthSquared));
        StoreComponents(_mm_and_ps(_mm_mul_ps(x, inverseLength), valid),
                        _mm_and_ps(_mm_mul_ps(y, inverseLength), valid),
                        _mm_and_ps(_mm_mul_ps(z, inverseLength), valid),
                        output, i, count);
    }

    for (; i < count; ++i)
    {
        const float lengthSquared = input[i].MagnitudeSquared();
        output[i] = lengthSquared > 1e-12f
            ? input[i] * (1.0f / std::sqrt(lengthSquared))
            : Vec3f::Zero;
    }
}

void BatchTransformPoint(const Matrix4x4& mat, const Vec3f* input, Vec3f* output, std::size_t count)
{
    if (!input || !output || count == 0) return;

    const __m128 m00 = _mm_set1_ps(mat(0, 0));
    const __m128 m01 = _mm_set1_ps(mat(0, 1));
    const __m128 m02 = _mm_set1_ps(mat(0, 2));
    const __m128 m03 = _mm_set1_ps(mat(0, 3));
    const __m128 m10 = _mm_set1_ps(mat(1, 0));
    const __m128 m11 = _mm_set1_ps(mat(1, 1));
    const __m128 m12 = _mm_set1_ps(mat(1, 2));
    const __m128 m13 = _mm_set1_ps(mat(1, 3));
    const __m128 m20 = _mm_set1_ps(mat(2, 0));
    const __m128 m21 = _mm_set1_ps(mat(2, 1));
    const __m128 m22 = _mm_set1_ps(mat(2, 2));
    const __m128 m23 = _mm_set1_ps(mat(2, 3));

    std::size_t i = 0;
    for (; i + 3 < count; i += 4)
    {
        const __m128 x = LoadComponent(input, i, count, 0);
        const __m128 y = LoadComponent(input, i, count, 1);
        const __m128 z = LoadComponent(input, i, count, 2);
        const __m128 outX = _mm_add_ps(
            _mm_add_ps(_mm_mul_ps(x, m00), _mm_mul_ps(y, m01)),
            _mm_add_ps(_mm_mul_ps(z, m02), m03));
        const __m128 outY = _mm_add_ps(
            _mm_add_ps(_mm_mul_ps(x, m10), _mm_mul_ps(y, m11)),
            _mm_add_ps(_mm_mul_ps(z, m12), m13));
        const __m128 outZ = _mm_add_ps(
            _mm_add_ps(_mm_mul_ps(x, m20), _mm_mul_ps(y, m21)),
            _mm_add_ps(_mm_mul_ps(z, m22), m23));
        StoreComponents(outX, outY, outZ, output, i, count);
    }

    for (; i < count; ++i)
        output[i] = mat.TransformPoint(input[i]);
}

}
