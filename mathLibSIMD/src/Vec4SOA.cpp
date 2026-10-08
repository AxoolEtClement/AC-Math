#include "Vec4SOA.hpp"
#include <stdexcept>

namespace simd
{
    Vec4SOA::Vec4SOA() 
        : _dataX{_mm_setzero_ps()}, _dataY{_mm_setzero_ps()}, _dataZ{_mm_setzero_ps()}, _dataW{_mm_setzero_ps()}
    {
    }

    Vec4SOA::Vec4SOA(const float x[4], const float y[4], const float z[4], const float w[4]) 
        : _dataX{_mm_loadu_ps(x)}, _dataY{_mm_loadu_ps(y)}, _dataZ{_mm_loadu_ps(z)}, _dataW{_mm_loadu_ps(w)}
    {
    }

    Vec4SOA::Vec4SOA(const Vec4SOA& other) 
        : _dataX{other._dataX}, _dataY{other._dataY}, _dataZ{other._dataZ}, _dataW{other._dataW}
    {
    }

    Vec4SOA::Vec4SOA(__m128 x, __m128 y, __m128 z, __m128 w) 
        : _dataX{x}, _dataY{y}, _dataZ{z}, _dataW{w}
    {
    }

    __m128 Vec4SOA::getX() const { return _dataX; }
    __m128 Vec4SOA::getY() const { return _dataY; }
    __m128 Vec4SOA::getZ() const { return _dataZ; }
    __m128 Vec4SOA::getW() const { return _dataW; }

    void Vec4SOA::setX(__m128 value) { _dataX = value; }
    void Vec4SOA::setY(__m128 value) { _dataY = value; }
    void Vec4SOA::setZ(__m128 value) { _dataZ = value; }
    void Vec4SOA::setW(__m128 value) { _dataW = value; }

    Vec4SOA Vec4SOA::operator+(const Vec4SOA& rhs) const
    {
        return { _mm_add_ps(_dataX, rhs._dataX), _mm_add_ps(_dataY, rhs._dataY), _mm_add_ps(_dataZ, rhs._dataZ), _mm_add_ps(_dataW, rhs._dataW) };
    }

    Vec4SOA Vec4SOA::operator-(const Vec4SOA& rhs) const
    {
        return { _mm_sub_ps(_dataX, rhs._dataX), _mm_sub_ps(_dataY, rhs._dataY), _mm_sub_ps(_dataZ, rhs._dataZ), _mm_sub_ps(_dataW, rhs._dataW) };
    }

    Vec4SOA Vec4SOA::operator*(const Vec4SOA& rhs) const
    {
        return { _mm_mul_ps(_dataX, rhs._dataX), _mm_mul_ps(_dataY, rhs._dataY), _mm_mul_ps(_dataZ, rhs._dataZ), _mm_mul_ps(_dataW, rhs._dataW) };
    }

    Vec4SOA Vec4SOA::operator/(const Vec4SOA& rhs) const
    {
        return { _mm_div_ps(_dataX, rhs._dataX), _mm_div_ps(_dataY, rhs._dataY), _mm_div_ps(_dataZ, rhs._dataZ), _mm_div_ps(_dataW, rhs._dataW) };
    }

    Vec4SOA Vec4SOA::operator-() const
    {
        __m128 zero = _mm_setzero_ps();
        return { _mm_sub_ps(zero, _dataX), _mm_sub_ps(zero, _dataY), _mm_sub_ps(zero, _dataZ), _mm_sub_ps(zero, _dataW) };
    }

    Vec4SOA Vec4SOA::operator*(float scalar) const
    {
        __m128 s = _mm_set1_ps(scalar);
        return { _mm_mul_ps(_dataX, s), _mm_mul_ps(_dataY, s), _mm_mul_ps(_dataZ, s), _mm_mul_ps(_dataW, s) };
    }

    Vec4SOA Vec4SOA::operator/(float scalar) const
    {
        if (scalar == 0.0f)
            throw std::domain_error("Cannot divide by zero");

        __m128 inv = _mm_set1_ps(1.0f / scalar);
        return { _mm_mul_ps(_dataX, inv), _mm_mul_ps(_dataY, inv), _mm_mul_ps(_dataZ, inv), _mm_mul_ps(_dataW, inv) };
    }

    Vec4SOA& Vec4SOA::operator+=(const Vec4SOA& rhs)
    {
        _dataX = _mm_add_ps(_dataX, rhs._dataX);
        _dataY = _mm_add_ps(_dataY, rhs._dataY);
        _dataZ = _mm_add_ps(_dataZ, rhs._dataZ);
        _dataW = _mm_add_ps(_dataW, rhs._dataW);
        return *this;
    }

    Vec4SOA& Vec4SOA::operator-=(const Vec4SOA& rhs)
    {
        _dataX = _mm_sub_ps(_dataX, rhs._dataX);
        _dataY = _mm_sub_ps(_dataY, rhs._dataY);
        _dataZ = _mm_sub_ps(_dataZ, rhs._dataZ);
        _dataW = _mm_sub_ps(_dataW, rhs._dataW);
        return *this;
    }

    Vec4SOA& Vec4SOA::operator*=(const Vec4SOA& rhs)
    {
        _dataX = _mm_mul_ps(_dataX, rhs._dataX);
        _dataY = _mm_mul_ps(_dataY, rhs._dataY);
        _dataZ = _mm_mul_ps(_dataZ, rhs._dataZ);
        _dataW = _mm_mul_ps(_dataW, rhs._dataW);
        return *this;
    }

    Vec4SOA& Vec4SOA::operator/=(const Vec4SOA& rhs)
    {
        _dataX = _mm_div_ps(_dataX, rhs._dataX);
        _dataY = _mm_div_ps(_dataY, rhs._dataY);
        _dataZ = _mm_div_ps(_dataZ, rhs._dataZ);
        _dataW = _mm_div_ps(_dataW, rhs._dataW);
        return *this;
    }

    Vec4SOA& Vec4SOA::operator*=(float scalar)
    {
        __m128 s = _mm_set1_ps(scalar);
        _dataX = _mm_mul_ps(_dataX, s);
        _dataY = _mm_mul_ps(_dataY, s);
        _dataZ = _mm_mul_ps(_dataZ, s);
        _dataW = _mm_mul_ps(_dataW, s);
        return *this;
    }

    Vec4SOA& Vec4SOA::operator/=(float scalar)
    {
        *this = *this / scalar;
        return *this;
    }

    bool Vec4SOA::operator==(const Vec4SOA& rhs) const
    {
        __m128 eqX = _mm_cmpeq_ps(_dataX, rhs._dataX);
        __m128 eqY = _mm_cmpeq_ps(_dataY, rhs._dataY);
        __m128 eqZ = _mm_cmpeq_ps(_dataZ, rhs._dataZ);
        __m128 eqW = _mm_cmpeq_ps(_dataW, rhs._dataW);
        return _mm_movemask_ps(_mm_and_ps(_mm_and_ps(eqX, eqY), _mm_and_ps(eqZ, eqW))) == 0xF;
    }

    bool Vec4SOA::operator!=(const Vec4SOA& rhs) const
    {
        return !(*this == rhs);
    }

    __m128 Vec4SOA::Dot(const Vec4SOA& rhs) const
    {
        __m128 xx = _mm_mul_ps(_dataX, rhs._dataX);
        __m128 yy = _mm_mul_ps(_dataY, rhs._dataY);
        __m128 zz = _mm_mul_ps(_dataZ, rhs._dataZ);
        __m128 ww = _mm_mul_ps(_dataW, rhs._dataW);
        return _mm_add_ps(_mm_add_ps(xx, yy), _mm_add_ps(zz, ww));
    }

    __m128 Vec4SOA::MagnitudeSquared() const
    {
        return Dot(*this);
    }

    __m128 Vec4SOA::Magnitude() const
    {
        return _mm_sqrt_ps(MagnitudeSquared());
    }

    Vec4SOA Vec4SOA::Normalize() const
    {
        __m128 invLen = _mm_div_ps(_mm_set1_ps(1.0f), Magnitude());
        return { _mm_mul_ps(_dataX, invLen), _mm_mul_ps(_dataY, invLen), _mm_mul_ps(_dataZ, invLen), _mm_mul_ps(_dataW, invLen) };
    }

    __m128 Vec4SOA::DistanceSquared(const Vec4SOA& rhs) const
    {
        return (*this - rhs).MagnitudeSquared();
    }

    __m128 Vec4SOA::Distance(const Vec4SOA& rhs) const
    {
        return _mm_sqrt_ps(DistanceSquared(rhs));
    }

    Vec4SOA Vec4SOA::Lerp(const Vec4SOA& a, const Vec4SOA& b, float t)
    {
        __m128 vt = _mm_set1_ps(t);
        return {
            _mm_add_ps(a._dataX, _mm_mul_ps(_mm_sub_ps(b._dataX, a._dataX), vt)),
            _mm_add_ps(a._dataY, _mm_mul_ps(_mm_sub_ps(b._dataY, a._dataY), vt)),
            _mm_add_ps(a._dataZ, _mm_mul_ps(_mm_sub_ps(b._dataZ, a._dataZ), vt)),
            _mm_add_ps(a._dataW, _mm_mul_ps(_mm_sub_ps(b._dataW, a._dataW), vt))
        };
    }

    Vec4SOA Vec4SOA::Min(const Vec4SOA& a, const Vec4SOA& b)
    {
        return { _mm_min_ps(a._dataX, b._dataX), _mm_min_ps(a._dataY, b._dataY), _mm_min_ps(a._dataZ, b._dataZ), _mm_min_ps(a._dataW, b._dataW) };
    }

    Vec4SOA Vec4SOA::Max(const Vec4SOA& a, const Vec4SOA& b)
    {
        return { _mm_max_ps(a._dataX, b._dataX), _mm_max_ps(a._dataY, b._dataY), _mm_max_ps(a._dataZ, b._dataZ), _mm_max_ps(a._dataW, b._dataW) };
    }

    const Vec4SOA Vec4SOA::Zero{_mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps()};
    const Vec4SOA Vec4SOA::One{_mm_set1_ps(1.0f), _mm_set1_ps(1.0f), _mm_set1_ps(1.0f), _mm_set1_ps(1.0f)};

    Vec4SOA operator*(float scalar, const Vec4SOA& vector)
    {
        return vector * scalar;
    }
}