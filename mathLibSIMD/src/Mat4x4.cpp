#include "Mat4x4.hpp"

namespace simd
{
    Matrix4x4::Matrix4x4() 
        : _data{ 
        _mm_setr_ps(1.f, 0.f, 0.f, 0.f),
        _mm_setr_ps(0.f, 1.f, 0.f, 0.f),
        _mm_setr_ps(0.f, 0.f, 1.f, 0.f),
        _mm_setr_ps(0.f, 0.f, 0.f, 1.f)
        }
    {}

    Matrix4x4::Matrix4x4(const std::array<float, 16>& elements) 
        : _data{
            _mm_loadu_ps(&elements[0]),
            _mm_loadu_ps(&elements[4]),
            _mm_loadu_ps(&elements[8]),
            _mm_loadu_ps(&elements[12])
        }
    {}

    Matrix4x4 Matrix4x4::Identity()
    {
        return Matrix4x4();
    }

    Matrix4x4 Matrix4x4::Zero()
    {
        return Matrix4x4(std::array<float, 16>{});
    }

    float& Matrix4x4::operator()(std::size_t row, std::size_t column)
    {
        if (row >= 4 || column >= 4)
        {
            throw std::out_of_range("Matrix index is out of range");
        }
        return _data[row].m128_f32[column];
    }

    const float& Matrix4x4::operator()(std::size_t row, std::size_t column) const
    {
        if (row >= 4 || column >= 4)
        {
            throw std::out_of_range("Matrix index is out of range");
        }
        return _data[row].m128_f32[column];
    }

    Matrix4x4 Matrix4x4::operator+(const Matrix4x4& rhs) const
    {
        Matrix4x4 result = Zero();
        for (std::size_t row = 0; row < 4; ++row)
        {
            result._data[row] = _mm_add_ps(_data[row], rhs._data[row]);
        }
        return result;
    }

    Matrix4x4 Matrix4x4::operator-(const Matrix4x4& rhs) const
    {
        Matrix4x4 result = Zero();
        for (std::size_t row = 0; row < 4; ++row)
        {
            result._data[row] = _mm_sub_ps(_data[row], rhs._data[row]);
        }
        return result;
    }

    Matrix4x4 Matrix4x4::operator*(const Matrix4x4& rhs) const
    {
        Matrix4x4 result;

        __m128 col0 = rhs._data[0];
        __m128 col1 = rhs._data[1];
        __m128 col2 = rhs._data[2];
        __m128 col3 = rhs._data[3];

        _MM_TRANSPOSE4_PS(col0, col1, col2, col3);

        constexpr int MASK = 0xFF;

        for (int i = 0; i < 4; ++i) {
            __m128 row = this->_data[i];

            __m128 dp0 = _mm_dp_ps(row, col0, MASK); // Result: [r0, r0, r0, r0]
            __m128 dp1 = _mm_dp_ps(row, col1, MASK); // Result: [r1, r1, r1, r1]
            __m128 dp2 = _mm_dp_ps(row, col2, MASK); // Result: [r2, r2, r2, r2]
            __m128 dp3 = _mm_dp_ps(row, col3, MASK); // Result: [r3, r3, r3, r3]

            __m128 low = _mm_unpacklo_ps(dp0, dp1); // [r0, r1, r0, r1]
            __m128 high = _mm_unpacklo_ps(dp2, dp3); // [r2, r3, r2, r3]
            result._data[i] = _mm_movelh_ps(low, high); // [r0, r1, r2, r3]
        }

        return result;
    }

    Vec4 Matrix4x4::operator*(const Vec4& rhs) const
    {
        return {
            _mm_cvtss_f32(_mm_dp_ps(_data[0], rhs._data, 0xF1)),
            _mm_cvtss_f32(_mm_dp_ps(_data[1], rhs._data, 0xF1)),
            _mm_cvtss_f32(_mm_dp_ps(_data[2], rhs._data, 0xF1)),
            _mm_cvtss_f32(_mm_dp_ps(_data[3], rhs._data, 0xF1))
        };
    }

    Matrix4x4 Matrix4x4::operator*(float scalar) const
    {
        Matrix4x4 result = Zero();

		result._data[0] = _mm_mul_ps(_data[0], _mm_set1_ps(scalar));
		result._data[1] = _mm_mul_ps(_data[1], _mm_set1_ps(scalar));
		result._data[2] = _mm_mul_ps(_data[2], _mm_set1_ps(scalar));
		result._data[3] = _mm_mul_ps(_data[3], _mm_set1_ps(scalar));

        return result;
    }

    Matrix4x4& Matrix4x4::operator*=(const Matrix4x4& rhs)
    {
        *this = *this * rhs;
        return *this;
    }

    bool Matrix4x4::operator==(const Matrix4x4& rhs) const
    {
        for (std::size_t row = 0; row < 4; ++row)
        {
            if (_mm_movemask_ps(_mm_cmpeq_ps(_data[row], rhs._data[row])) != 0xF)
                return false;
        }
        return true;
    }

    bool Matrix4x4::operator!=(const Matrix4x4& rhs) const
    {
        return !(*this == rhs);
    }

    Matrix4x4 Matrix4x4::Transpose() const
    {
        Matrix4x4 result = Zero();

		__m128 r0 = _data[0];
        __m128 r1 = _data[1];
        __m128 r2 = _data[2];
        __m128 r3 = _data[3];

		_MM_TRANSPOSE4_PS(r0, r1, r2, r3);

		result._data[0] = r0;
		result._data[1] = r1;
		result._data[2] = r2;
		result._data[3] = r3;

        return result;
    }

    float Matrix4x4::Determinant() const
    {
        float m[4][4];
        _mm_storeu_ps(m[0], _data[0]);
        _mm_storeu_ps(m[1], _data[1]);
        _mm_storeu_ps(m[2], _data[2]);
        _mm_storeu_ps(m[3], _data[3]);

		// Helper lambda to compute the determinant of the sub 3x3 matrix
        auto det3 = [](float m00, float m01, float m02,
            float m10, float m11, float m12,
            float m20, float m21, float m22) {
                return m00 * (m11 * m22 - m12 * m21) -
                    m01 * (m10 * m22 - m12 * m20) +
                    m02 * (m10 * m21 - m11 * m20);
            };

        float sub0 = det3(m[1][1], m[1][2], m[1][3], m[2][1], m[2][2], m[2][3], m[3][1], m[3][2], m[3][3]);
        float sub1 = det3(m[1][0], m[1][2], m[1][3], m[2][0], m[2][2], m[2][3], m[3][0], m[3][2], m[3][3]);
        float sub2 = det3(m[1][0], m[1][1], m[1][3], m[2][0], m[2][1], m[2][3], m[3][0], m[3][1], m[3][3]);
        float sub3 = det3(m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2], m[3][0], m[3][1], m[3][2]);

        return m[0][0] * sub0 - m[0][1] * sub1 + m[0][2] * sub2 - m[0][3] * sub3;
    }

    Matrix4x4 Matrix4x4::Inverse(float relativeTolerance) const
    {
        if (!std::isfinite(relativeTolerance) || relativeTolerance < 0.f || relativeTolerance >= 1.f)
        {
            throw std::invalid_argument("Inverse tolerance must be finite and in [0, 1)");
        }

        float l[4][4];
        _mm_storeu_ps(l[0], _data[0]);
        _mm_storeu_ps(l[1], _data[1]);
        _mm_storeu_ps(l[2], _data[2]);
        _mm_storeu_ps(l[3], _data[3]);

        float r[4][4] = {
            {1.f, 0.f, 0.f, 0.f},
            {0.f, 1.f, 0.f, 0.f},
            {0.f, 0.f, 1.f, 0.f},
            {0.f, 0.f, 0.f, 1.f}
        };

        float scales[4]{};
        for (std::size_t row = 0; row < 4; ++row)
        {
            for (std::size_t column = 0; column < 4; ++column)
            {
                if (!std::isfinite(l[row][column]))
                {
                    throw std::domain_error("Cannot invert a non-finite matrix");
                }
                scales[row] = std::max(scales[row], std::abs(l[row][column]));
            }
            if (scales[row] == 0.f)
            {
                throw std::domain_error("Cannot invert a singular matrix");
            }
        }

        for (std::size_t column = 0; column < 4; ++column)
        {
            std::size_t pivot = column;
            for (std::size_t row = column + 1; row < 4; ++row)
            {
                if (std::abs(l[row][column]) / scales[row] >
                    std::abs(l[pivot][column]) / scales[pivot])
                {
                    pivot = row;
                }
            }
            if (std::abs(l[pivot][column]) / scales[pivot] <= relativeTolerance)
            {
                throw std::domain_error("Matrix is singular or too ill-conditioned");
            }
            for (std::size_t j = 0; j < 4; ++j)
            {
                std::swap(l[column][j], l[pivot][j]);
                std::swap(r[column][j], r[pivot][j]);
            }
            std::swap(scales[column], scales[pivot]);

            const float diagonal = l[column][column];
            for (std::size_t j = 0; j < 4; ++j)
            {
                l[column][j] /= diagonal;
                r[column][j] /= diagonal;
            }
            for (std::size_t row = 0; row < 4; ++row)
            {
                if (row == column)
                {
                    continue;
                }
                const float factor = l[row][column];
                for (std::size_t j = 0; j < 4; ++j)
                {
                    l[row][j] -= factor * l[column][j];
                    r[row][j] -= factor * r[column][j];
                }
            }
        }

        for (std::size_t row = 0; row < 4; ++row)
        {
            for (std::size_t column = 0; column < 4; ++column)
            {
                if (!std::isfinite(r[row][column]))
                {
                    throw std::domain_error("Inverse is not representable");
                }
            }
        }

        Matrix4x4 result;
        result._data[0] = _mm_loadu_ps(r[0]);
        result._data[1] = _mm_loadu_ps(r[1]);
        result._data[2] = _mm_loadu_ps(r[2]);
        result._data[3] = _mm_loadu_ps(r[3]);

        return result;
    }

    Matrix4x4 Matrix4x4::Scale(const Vec3f& scale)
    {
        Matrix4x4 result;
		result._data[0] = _mm_setr_ps(scale.getX(), 0.f, 0.f, 0.f);
		result._data[1] = _mm_setr_ps(0.f, scale.getX(), 0.f, 0.f);
		result._data[2] = _mm_setr_ps(0.f, 0.f, scale.getZ(), 0.f);

        return result;
    }

    Matrix4x4 Matrix4x4::RotationX(float radians)
    {
        Matrix4x4 result;
        const float cosine = std::cos(radians);
        const float sine = std::sin(radians);

        result._data[0] = _mm_setr_ps(1.f, 0.f, 0.f, 0.f);
        result._data[1] = _mm_setr_ps(0.f, cosine, -sine, 0.f);
        result._data[2] = _mm_setr_ps(0.f, sine, cosine, 0.f);
        result._data[3] = _mm_setr_ps(0.f, 0.f, 0.f, 1.f);
        return result;
    }

    Matrix4x4 Matrix4x4::RotationY(float radians)
    {
        Matrix4x4 result;
        const float cosine = std::cos(radians);
        const float sine = std::sin(radians);

        result._data[0] = _mm_setr_ps(cosine, 0.f, sine, 0.f);
        result._data[1] = _mm_setr_ps(0.f, 1.f, 0.f, 0.f);
        result._data[2] = _mm_setr_ps(-sine, 0.f, cosine, 0.f);
        result._data[3] = _mm_setr_ps(0.f, 0.f, 0.f, 1.f);
        return result;
    }

    Matrix4x4 Matrix4x4::RotationZ(float radians)
    {
        Matrix4x4 result;
        const float cosine = std::cos(radians);
        const float sine = std::sin(radians);

        result._data[0] = _mm_setr_ps(cosine, -sine, 0.f, 0.f);
        result._data[1] = _mm_setr_ps(sine, cosine, 0.f, 0.f);
        result._data[2] = _mm_setr_ps(0.f, 0.f, 1.f, 0.f);
        result._data[3] = _mm_setr_ps(0.f, 0.f, 0.f, 1.f);
        return result;
    }

    Matrix4x4 Matrix4x4::Translation(const Vec3f& offset)
    {
        Matrix4x4 result;

        result._data[0] = _mm_setr_ps(1.f, 0.f, 0.f, offset.getX());
        result._data[1] = _mm_setr_ps(0.f, 1.f, 0.f, offset.getY());
        result._data[2] = _mm_setr_ps(0.f, 0.f, 1.f, offset.getZ());
        result._data[3] = _mm_setr_ps(0.f, 0.f, 0.f, 1.f);
        return result;
    }

    Vec3f Matrix4x4::TransformPoint(const Vec3f& point) const
    {
        float row3[4];
        _mm_storeu_ps(row3, _data[3]);

        if (row3[0] != 0.f || row3[1] != 0.f ||
            row3[2] != 0.f || row3[3] != 1.f)
        {
            throw std::domain_error("Transform requires an affine matrix");
        }

        const Vec4 result = *this * Vec4(point.getX(), point.getY(), point.getZ(), 1.f);
        return { result.getX(), result.getY(), result.getZ() };
    }

    Vec3f Matrix4x4::TransformDirection(const Vec3f& direction) const
    {
        float row3[4];
        _mm_storeu_ps(row3, _data[3]);

        if (row3[0] != 0.f || row3[1] != 0.f ||
            row3[2] != 0.f || row3[3] != 1.f)
        {
            throw std::domain_error("Transform requires an affine matrix");
        }

        const Vec4 result = *this * Vec4(direction.getX(), direction.getY(), direction.getZ(), 0.f);
        return { result.getX(), result.getY(), result.getZ()};
    }
}