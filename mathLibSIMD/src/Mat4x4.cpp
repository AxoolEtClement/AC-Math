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
        return _data[row][column];
    }

    const float& Matrix4x4::operator()(std::size_t row, std::size_t column) const
    {
        if (row >= 4 || column >= 4)
        {
            throw std::out_of_range("Matrix index is out of range");
        }
        return _data[row][column];
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
        for (std::size_t row = 0; row < 4; ++row)
        {
            result._data[row] = _mm_mul_ps(_data[row], _mm_set1_ps(scalar));
        }
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
        for (std::size_t row = 0; row < 4; ++row)
        {
            for (std::size_t column = 0; column < 4; ++column)
            {
                result.values[column][row] = values[row][column];
            }
        }
        return result;
    }

    T Matrix4x4::Determinant() const
    {
        Matrix4x4 working = *this;
        T determinant = T{ 1 };
        for (std::size_t column = 0; column < 4; ++column)
        {
            std::size_t pivot = column;
            for (std::size_t row = column + 1; row < 4; ++row)
            {
                if (std::abs(working.values[row][column]) > std::abs(working.values[pivot][column]))
                {
                    pivot = row;
                }
            }
            if (working.values[pivot][column] == T{ 0 })
            {
                return T{ 0 };
            }
            if (pivot != column)
            {
                for (std::size_t j = 0; j < 4; ++j)
                {
                    std::swap(working.values[pivot][j], working.values[column][j]);
                }
                determinant = -determinant;
            }
            const T diagonal = working.values[column][column];
            determinant *= diagonal;
            for (std::size_t row = column + 1; row < 4; ++row)
            {
                const T factor = working.values[row][column] / diagonal;
                for (std::size_t j = column + 1; j < 4; ++j)
                {
                    working.values[row][j] -= factor * working.values[column][j];
                }
            }
        }
        return determinant;
    }

    Matrix4x4 Matrix4x4::Inverse(T relativeTolerance) const
    {
        if (!std::isfinite(relativeTolerance) || relativeTolerance < T{ 0 } || relativeTolerance >= T{ 1 })
        {
            throw std::invalid_argument("Inverse tolerance must be finite and in [0, 1)");
        }
        Matrix4x4 left = *this;
        Matrix4x4 right;
        T scales[4]{};
        for (std::size_t row = 0; row < 4; ++row)
        {
            for (std::size_t column = 0; column < 4; ++column)
            {
                if (!std::isfinite(left.values[row][column]))
                {
                    throw std::domain_error("Cannot invert a non-finite matrix");
                }
                scales[row] = std::max(scales[row], std::abs(left.values[row][column]));
            }
            if (scales[row] == T{ 0 })
            {
                throw std::domain_error("Cannot invert a singular matrix");
            }
        }
        for (std::size_t column = 0; column < 4; ++column)
        {
            std::size_t pivot = column;
            for (std::size_t row = column + 1; row < 4; ++row)
            {
                if (std::abs(left.values[row][column]) / scales[row] >
                    std::abs(left.values[pivot][column]) / scales[pivot])
                {
                    pivot = row;
                }
            }
            if (std::abs(left.values[pivot][column]) / scales[pivot] <= relativeTolerance)
            {
                throw std::domain_error("Matrix is singular or too ill-conditioned");
            }
            for (std::size_t j = 0; j < 4; ++j)
            {
                std::swap(left.values[column][j], left.values[pivot][j]);
                std::swap(right.values[column][j], right.values[pivot][j]);
            }
            std::swap(scales[column], scales[pivot]);
            const T diagonal = left.values[column][column];
            for (std::size_t j = 0; j < 4; ++j)
            {
                left.values[column][j] /= diagonal;
                right.values[column][j] /= diagonal;
            }
            for (std::size_t row = 0; row < 4; ++row)
            {
                if (row == column)
                {
                    continue;
                }
                const T factor = left.values[row][column];
                for (std::size_t j = 0; j < 4; ++j)
                {
                    left.values[row][j] -= factor * left.values[column][j];
                    right.values[row][j] -= factor * right.values[column][j];
                }
            }
        }
        for (std::size_t row = 0; row < 4; ++row)
        {
            for (std::size_t column = 0; column < 4; ++column)
            {
                if (!std::isfinite(right.values[row][column]))
                {
                    throw std::domain_error("Inverse is not representable");
                }
            }
        }
        return right;
    }

    Matrix4x4 Matrix4x4::Scale(const Vec3f& scale)
    {
        Matrix4x4 result;
        result.values[0][0] = scale.x;
        result.values[1][1] = scale.y;
        result.values[2][2] = scale.z;
        return result;
    }

    Matrix4x4 Matrix4x4::RotationX(T radians)
    {
        Matrix4x4 result;
        const T cosine = std::cos(radians);
        const T sine = std::sin(radians);
        result.values[1][1] = cosine;
        result.values[2][2] = cosine;
        result.values[1][2] = -sine;
        result.values[2][1] = sine;
        return result;
    }

    Matrix4x4 Matrix4x4::RotationY(T radians)
    {
        Matrix4x4 result;
        const T cosine = std::cos(radians);
        const T sine = std::sin(radians);
        result.values[2][2] = cosine;
        result.values[0][0] = cosine;
        result.values[2][0] = -sine;
        result.values[0][2] = sine;
        return result;
    }

    Matrix4x4 Matrix4x4::RotationZ(T radians)
    {
        Matrix4x4 result;
        const T cosine = std::cos(radians);
        const T sine = std::sin(radians);
        result.values[0][0] = cosine;
        result.values[1][1] = cosine;
        result.values[0][1] = -sine;
        result.values[1][0] = sine;
        return result;
    }

    Matrix4x4 Matrix4x4::Translation(const Vec3f& offset)
    {
        Matrix4x4 result;
        result.values[0][3] = offset.x;
        result.values[1][3] = offset.y;
        result.values[2][3] = offset.z;
        return result;
    }

    Vec3f Matrix4x4::TransformPoint(const Vec3f& point) const
    {
        if (values[3][0] != T{ 0 } || values[3][1] != T{ 0 } ||
            values[3][2] != T{ 0 } || values[3][3] != T{ 1 })
        {
            throw std::domain_error("Transform requires an affine matrix");
        }
        const Vec4 result = *this * Vec4(point.x, point.y, point.z, T{ 1 });
        return { result.x, result.y, result.z };
    }

    Vec3f Matrix4x4::TransformDirection(const Vec3f& direction) const
    {
        if (values[3][0] != T{ 0 } || values[3][1] != T{ 0 } ||
            values[3][2] != T{ 0 } || values[3][3] != T{ 1 })
        {
            throw std::domain_error("Transform requires an affine matrix");
        }
        const Vec4 result = *this * Vec4(direction.x, direction.y, direction.z, T{ 0 });
        return { result.x, result.y, result.z };
    }
}