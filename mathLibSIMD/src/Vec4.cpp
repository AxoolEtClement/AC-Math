#include "Vec4.hpp"

namespace
{
    inline __m128 DotVec4(__m128 a, __m128 b)
    {
        const __m128 m = _mm_mul_ps(a, b);
        const __m128 shuf = _mm_shuffle_ps(m, m, _MM_SHUFFLE(1, 0, 3, 2));
        const __m128 sums = _mm_add_ps(m, shuf);
        const __m128 shuf2 = _mm_shuffle_ps(sums, sums, _MM_SHUFFLE(2, 3, 0, 1));
        return _mm_add_ps(sums, shuf2);
    }
}

namespace simd
{

    Vec4::Vec4() : _data{_mm_setzero_ps()}
    {

    }


    Vec4::Vec4(float _x, float _y, float _z, float _w) : _data{_mm_set_ps(_w, _z, _y, _x)}
    {

    }

    Vec4::Vec4(__m128 data) : _data{data}
    {

    }

    float Vec4::getX() const { return _mm_cvtss_f32(_data); }
    float Vec4::getY() const { return _mm_cvtss_f32(_mm_shuffle_ps(_data, _data, _MM_SHUFFLE(1, 1, 1, 1))); }
    float Vec4::getZ() const { return _mm_cvtss_f32(_mm_shuffle_ps(_data, _data, _MM_SHUFFLE(2, 2, 2, 2))); }
    float Vec4::getW() const { return _mm_cvtss_f32(_mm_shuffle_ps(_data, _data, _MM_SHUFFLE(3, 3, 3, 3))); }

    void Vec4::setX(float value) { _data = _mm_insert_ps(_data, _mm_set_ss(value), 0x00); }
    void Vec4::setY(float value) { _data = _mm_insert_ps(_data, _mm_set_ss(value), 0x10); }
    void Vec4::setZ(float value) { _data = _mm_insert_ps(_data, _mm_set_ss(value), 0x20); }
    void Vec4::setW(float value) { _data = _mm_insert_ps(_data, _mm_set_ss(value), 0x30); }


    Vec4 Vec4::store(__m128 val)
    {
        alignas(16) float components[4];

        _mm_store_ps(components, val);

        return {
            components[0],
            components[1],
            components[2],
            components[3]
        };
    }


    Vec4 Vec4::operator+(const Vec4& rhs) const
    {
        return _mm_add_ps(this->_data, rhs._data);
    }


    Vec4 Vec4::operator-(const Vec4& rhs) const
    {
        return _mm_sub_ps(this->_data, rhs._data);
    }


    Vec4 Vec4::operator*(const Vec4& rhs) const
    {
        return _mm_mul_ps(this->_data, rhs._data);
    }


    Vec4 Vec4::operator/(const Vec4& rhs) const
    {
        if (rhs.getX() == 0.0f || rhs.getY() == 0.0f || rhs.getZ() == 0.0f || rhs.getW() == 0.0f)
        {
            throw std::domain_error("Cannot divide by a zero component");
        }

        return _mm_div_ps(this->_data, rhs._data);
    }


    Vec4 Vec4::operator-() const
    {
        // Negate the vector by subtracting it from zero
        return _mm_sub_ps(_mm_set1_ps(0.0f), this->_data);
    }


    Vec4 Vec4::operator*(float scalar) const
    {
        return _mm_mul_ps(this->_data, _mm_set1_ps(scalar));
    }


    Vec4 Vec4::operator/(float scalar) const
    {
        if (scalar == 0.0f)
        {
            throw std::domain_error("Cannot divide by zero");
        }

        return _mm_div_ps(this->_data, _mm_set1_ps(scalar));
    }


    Vec4& Vec4::operator+=(const Vec4& rhs)
    {
        _data = _mm_add_ps(_data, rhs._data);
        return *this;
    }


    Vec4& Vec4::operator-=(const Vec4& rhs)
    {
        _data = _mm_sub_ps(_data, rhs._data);
        return *this;
    }


    Vec4& Vec4::operator*=(const Vec4& rhs)
    {
        _data = _mm_mul_ps(_data, rhs._data);
        return *this;
    }


    Vec4& Vec4::operator/=(const Vec4& rhs)
    {
        // operator/ lève l'exception avant toute modification de *this
        *this = *this / rhs;
        return *this;
    }


    Vec4& Vec4::operator*=(float scalar)
    {
        _data = _mm_mul_ps(_data, _mm_set1_ps(scalar));
        return *this;
    }


    Vec4& Vec4::operator/=(float scalar)
    {
        *this = *this / scalar;
        return *this;
    }


    bool Vec4::operator==(const Vec4& rhs) const
    {
        __m128 mask = _mm_cmpeq_ps(this->_data, rhs._data);
        return _mm_movemask_ps(mask) == 0xF;
    }


    bool Vec4::operator!=(const Vec4& rhs) const
    {
        __m128 mask = _mm_cmpneq_ps(this->_data, rhs._data);
        return _mm_movemask_ps(mask) != 0x0;
    }


    float Vec4::Dot(const Vec4& rhs) const
    {
        return _mm_cvtss_f32(DotVec4(this->_data, rhs._data));
    }


    float Vec4::MagnitudeSquared() const
    {
        return _mm_cvtss_f32(DotVec4(this->_data, this->_data));
    }


    float Vec4::Magnitude() const
    {
        return _mm_cvtss_f32(_mm_sqrt_ps(DotVec4(this->_data, this->_data)));
    }


    Vec4 Vec4::Normalize() const
    {
        if (!std::isfinite(this->getX()) || !std::isfinite(this->getY()) || !std::isfinite(this->getZ()) || !std::isfinite(this->getW()))
        {
            throw std::domain_error("Cannot normalize non-finite components");
        }

        const float scale = std::max({std::abs(this->getX()), std::abs(this->getY()), std::abs(this->getZ()), std::abs(this->getW())});
        if (scale == 0.0f)
        {
            throw std::domain_error("Cannot normalize the zero vector");
        }

        const __m128 scaled = _mm_div_ps(this->_data, _mm_set1_ps(scale));
        return _mm_div_ps(scaled, _mm_sqrt_ps(DotVec4(scaled, scaled)));
    }


    float Vec4::DistanceSquared(const Vec4& rhs) const
    {
        __m128 diff = _mm_sub_ps(this->_data, rhs._data);
        return _mm_cvtss_f32(DotVec4(diff, diff));
    }


    float Vec4::Distance(const Vec4& rhs) const
    {
        __m128 diff = _mm_sub_ps(this->_data, rhs._data);
        return _mm_cvtss_f32(_mm_sqrt_ps(DotVec4(diff, diff)));
    }


    float Vec4::Angle(const Vec4& rhs) const
    {
        __m128 len1_sq = _mm_dp_ps(this->_data, this->_data, 0xFF);
        __m128 len2_sq = _mm_dp_ps(rhs._data, rhs._data, 0xFF);

        if (_mm_cvtss_f32(len1_sq) == 0.0f || _mm_cvtss_f32(len2_sq) == 0.0f)
        {
            throw std::domain_error("Cannot compute the angle with a zero vector");
        }

        __m128 inv_len1 = _mm_div_ps(_mm_set1_ps(1.0f), _mm_sqrt_ps(len1_sq));
        __m128 inv_len2 = _mm_div_ps(_mm_set1_ps(1.0f), _mm_sqrt_ps(len2_sq));

        __m128 n1 = _mm_mul_ps(this->_data, inv_len1);
        __m128 n2 = _mm_mul_ps(rhs._data, inv_len2);

        __m128 dot_prod = _mm_dp_ps(n1, n2, 0xFF);

        float cosine;
        _mm_store_ss(&cosine, dot_prod);


        cosine = std::clamp(cosine, -1.0f, 1.0f);

        return std::acos(cosine);
    }


    Vec4 Vec4::Lerp(const Vec4& a, const Vec4& b, float t)
    {
        return _mm_add_ps(a._data, _mm_mul_ps(_mm_sub_ps(b._data, a._data), _mm_set1_ps(t)));
    }


    Vec4 Vec4::Min(const Vec4& a, const Vec4& b)
    {
        return _mm_min_ps(a._data, b._data);
    }


    Vec4 Vec4::Max(const Vec4& a, const Vec4& b)
    {
        return _mm_max_ps(a._data, b._data);
    }


    const Vec4 Vec4::Zero{0, 0, 0, 0};


    const Vec4 Vec4::One{1, 1, 1, 1};


    const Vec4 Vec4::UnitX{1, 0, 0, 0};


    const Vec4 Vec4::UnitY{0, 1, 0, 0};


    const Vec4 Vec4::UnitZ{0, 0, 1, 0};


    const Vec4 Vec4::UnitW{0, 0, 0, 1};


    Vec4 operator*(float scalar, const Vec4& vector)
    {
        return vector * scalar;
    }
}