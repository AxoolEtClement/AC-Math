#include "Mat3.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace
{
    // ligne de A*B = A[i][0]*B.ligne0 + A[i][1]*B.ligne1 + A[i][2]*B.ligne2
    inline __m128 MulRow(__m128 a, __m128 b0, __m128 b1, __m128 b2)
    {
        const __m128 x = _mm_shuffle_ps(a, a, _MM_SHUFFLE(0, 0, 0, 0));
        const __m128 y = _mm_shuffle_ps(a, a, _MM_SHUFFLE(1, 1, 1, 1));
        const __m128 z = _mm_shuffle_ps(a, a, _MM_SHUFFLE(2, 2, 2, 2));
        return _mm_add_ps(_mm_add_ps(_mm_mul_ps(x, b0), _mm_mul_ps(y, b1)), _mm_mul_ps(z, b2));
    }

    inline float& At(__m128& row, std::size_t c)
    {
        return reinterpret_cast<float*>(&row)[c];
    }

    inline const float& At(const __m128& row, std::size_t c)
    {
        return reinterpret_cast<const float*>(&row)[c];
    }
}

namespace simd
{

    Matrix3x3::Matrix3x3() : _data{}
    {
        _data[0] = _mm_set_ps(0.0f, 0.0f, 0.0f, 1.0f);
        _data[1] = _mm_set_ps(0.0f, 0.0f, 1.0f, 0.0f);
        _data[2] = _mm_set_ps(0.0f, 1.0f, 0.0f, 0.0f);
    }


    Matrix3x3::Matrix3x3(const std::array<float, 9>& elements) : _data{}
    {
        for (std::size_t row = 0; row < 3; ++row)
        {
            _data[row] = _mm_set_ps(0.0f,
                                    elements[row * 3 + 2],
                                    elements[row * 3 + 1],
                                    elements[row * 3 + 0]);
        }
    }


    Matrix3x3 Matrix3x3::Identity()
    {
        return Matrix3x3();
    }


    Matrix3x3 Matrix3x3::Zero()
    {
        Matrix3x3 result;
        result._data[0] = result._data[1] = result._data[2] = _mm_setzero_ps();
        return result;
    }


    float& Matrix3x3::operator()(std::size_t row, std::size_t column)
    {
        if (row >= 3 || column >= 3)
        {
            throw std::out_of_range("Matrix index is out of range");
        }
        return At(_data[row], column);
    }


    const float& Matrix3x3::operator()(std::size_t row, std::size_t column) const
    {
        if (row >= 3 || column >= 3)
        {
            throw std::out_of_range("Matrix index is out of range");
        }
        return reinterpret_cast<const float*>(&_data[row])[column];
    }


    Matrix3x3 Matrix3x3::operator+(const Matrix3x3& rhs) const
    {
        Matrix3x3 result;
        for (std::size_t row = 0; row < 3; ++row)
        {
            result._data[row] = _mm_add_ps(_data[row], rhs._data[row]);
        }
        return result;
    }


    Matrix3x3 Matrix3x3::operator-(const Matrix3x3& rhs) const
    {
        Matrix3x3 result;
        for (std::size_t row = 0; row < 3; ++row)
        {
            result._data[row] = _mm_sub_ps(_data[row], rhs._data[row]);
        }
        return result;
    }


    Matrix3x3 Matrix3x3::operator*(const Matrix3x3& rhs) const
    {
        Matrix3x3 result;
        for (int i = 0; i < 3; ++i)
        {
            const __m128 a0 = _mm_shuffle_ps(_data[i], _data[i], _MM_SHUFFLE(0, 0, 0, 0));
            const __m128 a1 = _mm_shuffle_ps(_data[i], _data[i], _MM_SHUFFLE(1, 1, 1, 1));
            const __m128 a2 = _mm_shuffle_ps(_data[i], _data[i], _MM_SHUFFLE(2, 2, 2, 2));

            result._data[i] = _mm_add_ps(
                _mm_add_ps(_mm_mul_ps(a0, rhs._data[0]), _mm_mul_ps(a1, rhs._data[1])),
                _mm_mul_ps(a2, rhs._data[2])
            );
        }
        return result;
    }


    Vec3f Matrix3x3::operator*(Vec3f rhs) const
    {
        __m128 c0 = _data[0];
        __m128 c1 = _data[1];
        __m128 c2 = _data[2];
        __m128 c3 = _mm_setzero_ps();

        _MM_TRANSPOSE4_PS(c0, c1, c2, c3);

        __m128 vx = _mm_shuffle_ps(rhs._data, rhs._data, _MM_SHUFFLE(0, 0, 0, 0));
        __m128 vy = _mm_shuffle_ps(rhs._data, rhs._data, _MM_SHUFFLE(1, 1, 1, 1));
        __m128 vz = _mm_shuffle_ps(rhs._data, rhs._data, _MM_SHUFFLE(2, 2, 2, 2));

        return _mm_add_ps(
            _mm_add_ps(_mm_mul_ps(vx, c0), _mm_mul_ps(vy, c1)),
            _mm_mul_ps(vz, c2)
        );
    }


    Matrix3x3 Matrix3x3::operator*(float scalar) const
    {
        Matrix3x3 result;
        __m128 scalarVec = _mm_set1_ps(scalar);
        for (std::size_t row = 0; row < 3; ++row)
        {
            result._data[row] = _mm_mul_ps(_data[row], scalarVec);
        }
        return result;
    }


    Matrix3x3& Matrix3x3::operator*=(const Matrix3x3& rhs)
    {
        // operator* construit un résultat séparé : sûr même si &rhs == this.
        *this = *this * rhs;
        return *this;
    }


    bool Matrix3x3::operator==(const Matrix3x3& rhs) const
    {
        for (std::size_t row = 0; row < 3; ++row)
        {
            if (_mm_movemask_ps(_mm_cmpeq_ps(_data[row], rhs._data[row])) != 0xF)
            {
                return false;
            }
        }
        return true;
    }


    bool Matrix3x3::operator!=(const Matrix3x3& rhs) const
    {
        return !(*this == rhs);
    }


    Matrix3x3 Matrix3x3::Transpose() const
    {
        __m128 r0 = _data[0];
        __m128 r1 = _data[1];
        __m128 r2 = _data[2];
        __m128 r3 = _mm_setzero_ps();
        _MM_TRANSPOSE4_PS(r0, r1, r2, r3);

        Matrix3x3 result;
        result._data[0] = r0;
        result._data[1] = r1;
        result._data[2] = r2;
        return result;
    }


    float Matrix3x3::Determinant() const
    {
        Matrix3x3 working = *this;

        float determinant = At(working._data[0], 0) * (At(working._data[1], 1) * At(working._data[2], 2) - At(working._data[1], 2) * At(working._data[2], 1)) -
                            At(working._data[0], 1) * (At(working._data[1], 0) * At(working._data[2], 2) - At(working._data[1], 2) * At(working._data[2], 0)) +
                            At(working._data[0], 2) * (At(working._data[1], 0) * At(working._data[2], 1) - At(working._data[1], 1) * At(working._data[2], 0));
        return determinant;
    }


    Matrix3x3 Matrix3x3::Inverse() const
    {
        auto cross = [](__m128 a, __m128 b) {
            const __m128 a_yzx = _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 0, 2, 1));
            const __m128 b_zxy = _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 1, 0, 2));
            const __m128 a_zxy = _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 1, 0, 2));
            const __m128 b_yzx = _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 0, 2, 1));
            return _mm_sub_ps(_mm_mul_ps(a_yzx, b_zxy), _mm_mul_ps(a_zxy, b_yzx));
        };

        __m128 c0 = cross(_data[1], _data[2]);
        __m128 c1 = cross(_data[2], _data[0]);
        __m128 c2 = cross(_data[0], _data[1]);

        // Déterminant = dot(_data[0], c0)
        __m128 m = _mm_mul_ps(_data[0], c0);
        __m128 det = _mm_add_ps(m, _mm_shuffle_ps(m, m, _MM_SHUFFLE(1, 1, 1, 1)));
        det = _mm_add_ps(det, _mm_shuffle_ps(m, m, _MM_SHUFFLE(2, 2, 2, 2)));

        const float det_val = _mm_cvtss_f32(det);
        if (std::abs(det_val) < 1e-8f)
        {
            throw std::runtime_error("Matrix is singular");
        }

        const __m128 inv_det = _mm_set1_ps(1.0f / det_val);

        __m128 r0 = _mm_mul_ps(c0, inv_det);
        __m128 r1 = _mm_mul_ps(c1, inv_det);
        __m128 r2 = _mm_mul_ps(c2, inv_det);
        __m128 r3 = _mm_setzero_ps();

        _MM_TRANSPOSE4_PS(r0, r1, r2, r3);

        Matrix3x3 res;
        res._data[0] = r0;
        res._data[1] = r1;
        res._data[2] = r2;
        return res;
    }


    Matrix3x3 Matrix3x3::Scale(const Vec3f& scale)
    {
        Matrix3x3 result;
        result._data[0] = _mm_set_ps(0.0f, 0.0f, 0.0f, scale.getX());
        result._data[1] = _mm_set_ps(0.0f, 0.0f, scale.getY(), 0.0f);
        result._data[2] = _mm_set_ps(0.0f, scale.getZ(), 0.0f, 0.0f);
        return result;
    }


    Matrix3x3 Matrix3x3::RotationX(float radians)
    {
        float c = std::cos(radians);
        float s = std::sin(radians);

        Matrix3x3 res;
        res._data[0] = _mm_setr_ps(1.0f, 0.0f, 0.0f, 0.0f);
        res._data[1] = _mm_setr_ps(0.0f,    c,   -s, 0.0f);
        res._data[2] = _mm_setr_ps(0.0f,    s,    c, 0.0f);
        return res;
    }


    Matrix3x3 Matrix3x3::RotationY(float radians)
    {
        float c = std::cos(radians);
        float s = std::sin(radians);
        Matrix3x3 res;
        res._data[0] = _mm_setr_ps(   c, 0.0f,    s, 0.0f);
        res._data[1] = _mm_setr_ps(0.0f, 1.0f, 0.0f, 0.0f);
        res._data[2] = _mm_setr_ps(  -s, 0.0f,    c, 0.0f);
        return res;
    }

    Matrix3x3 Matrix3x3::RotationZ(float radians)
    {
        float c = std::cos(radians);
        float s = std::sin(radians);
        Matrix3x3 res;
        res._data[0] = _mm_setr_ps(   c,  -s, 0.0f, 0.0f);
        res._data[1] = _mm_setr_ps(   s,   c, 0.0f, 0.0f);
        res._data[2] = _mm_setr_ps(0.0f, 0.0f, 1.0f, 0.0f);
        return res;
    }

}