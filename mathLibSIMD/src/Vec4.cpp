#include "Vec4.hpp"

namespace simd
{
    
    Vec4::Vec4() : x(0), y(0), z(0), w(0)
    {

    }

    
    Vec4::Vec4(float _x, float _y, float _z, float _w) : x(_x), y(_y), z(_z), w(_w)
    {

    }

    
    Vec4::Vec4(const Vec4& other) : x(static_cast<float>(other.x)), y(static_cast<float>(other.y)), z(static_cast<float>(other.z)), w(static_cast<float>(other.w))
    {

    }

    
    __m128 Vec4::load(const Vec4& v)
    {
        return _mm_set_ps(v.w, v.z, v.y, v.x);
    }

    
    Vec4 Vec4::store(__m128 val)
    {
        float components[4];

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
        return store(_mm_add_ps(load(*this), load(rhs)));
    }

    
    Vec4 Vec4::operator-(const Vec4& rhs) const
    {
        return store(_mm_sub_ps(load(*this), load(rhs)));
    };

    
    Vec4 Vec4::operator*(const Vec4& rhs) const
    {
        return store(_mm_mul_ps(load(*this), load(rhs)));
    }

    
    Vec4 Vec4::operator/(const Vec4& rhs) const
    {
        if (rhs.x == 0.0f || rhs.y == 0.0f || rhs.z == 0.0f || rhs.w == 0.0f)
        {
            throw std::domain_error("Cannot divide by a zero component");
        }

        return store(_mm_div_ps(load(*this), load(rhs)));
    }

    
    Vec4 Vec4::operator-() const
    {
        // Negate the vector by subtracting it from zero
        return store(_mm_sub_ps(_mm_set1_ps(0.0f), load(*this)));
    }

    
    Vec4 Vec4::operator*(float scalar) const
    {
        return store(_mm_mul_ps(load(*this), _mm_set1_ps(scalar)));
    }

    
    Vec4 Vec4::operator/(float scalar) const
    {
        if (scalar == 0.0f)
        {
            throw std::domain_error("Cannot divide by zero");
        }

        return store(_mm_div_ps(load(*this), _mm_set1_ps(scalar)));
    }

    
    Vec4& Vec4::operator+=(const Vec4& rhs)
    {
        *this = store(_mm_add_ps(load(*this), load(rhs)));
        return *this;
    }

    
    Vec4& Vec4::operator-=(const Vec4& rhs)
    {
        *this = store(_mm_sub_ps(load(*this), load(rhs)));
        return *this;
    }

    
    Vec4& Vec4::operator*=(const Vec4& rhs)
    {
        *this = store(_mm_mul_ps(load(*this), load(rhs)));
        return *this;
    }

    
    Vec4& Vec4::operator/=(const Vec4& rhs)
    {
        *this = store(_mm_div_ps(load(*this), load(rhs)));
        return *this;
    }

    
    Vec4& Vec4::operator*=(float scalar)
    {
        *this = store(_mm_mul_ps(load(*this), _mm_set1_ps(scalar)));
        return *this;
    }

    
    Vec4& Vec4::operator/=(float scalar)
    {
        *this = store(_mm_div_ps(load(*this), _mm_set1_ps(scalar)));
        return *this;
    }

    
    bool Vec4::operator==(const Vec4& rhs) const
    {
        __m128 mask = _mm_cmpeq_ps(load(*this), load(rhs));
        return _mm_movemask_ps(mask) == 0xF;
    }

    
    bool Vec4::operator!=(const Vec4& rhs) const
    {
        __m128 mask = _mm_cmpneq_ps(load(*this), load(rhs));
        return _mm_movemask_ps(mask) != 0x0;
    }

    
    float Vec4::Dot(const Vec4& rhs) const
    {
        return store(_mm_dp_ps(load(*this), load(rhs), 0xFF)).x;
    }

    
    float Vec4::MagnitudeSquared() const
    {
        return store(_mm_dp_ps(load(*this), load(*this), 0xFF)).x;
    }

    
    float Vec4::Magnitude() const
    {
        return store(_mm_sqrt_ps(_mm_dp_ps(load(*this), load(*this), 0xFF))).x;
    }

    
    Vec4 Vec4::Normalize() const
    {
        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z) || !std::isfinite(w))
        {
            throw std::domain_error("Cannot normalize non-finite components");
        }

        // Scaling avoids overflowing/underflowing the squared magnitude.
        const float scale = std::max({std::abs(x), std::abs(y), std::abs(z), std::abs(w)});
        if (scale == 0.0f)
        {
            throw std::domain_error("Cannot normalize the zero vector");
        }

        
        return store(_mm_div_ps(load(*this), _mm_set1_ps(Magnitude())));
    }

    
    float Vec4::DistanceSquared(const Vec4& rhs) const
    {
        return store(_mm_dp_ps(_mm_sub_ps(load(*this), load(rhs)), _mm_sub_ps(load(*this), load(rhs)), 0xFF)).x;
    }

    
    float Vec4::Distance(const Vec4& rhs) const
    {
        __m128 v1 = load(*this);
        __m128 v2 = load(rhs);
    
        __m128 diff = _mm_sub_ps(v1, v2);
    
        __m128 dp = _mm_dp_ps(diff, diff, 0xFF);
    
        __m128 sq = _mm_sqrt_ps(dp);
    
        float result;
        _mm_store_ss(&result, sq);
        return result;
    }

    
    float Vec4::Angle(const Vec4& rhs) const
    {
        __m128 v1 = load(*this);
        __m128 v2 = load(rhs);

        __m128 len1_sq = _mm_dp_ps(v1, v1, 0xFF);
        __m128 len2_sq = _mm_dp_ps(v2, v2, 0xFF);

        __m128 inv_len1 = _mm_div_ps(_mm_set1_ps(1.0f), _mm_sqrt_ps(len1_sq));
        __m128 inv_len2 = _mm_div_ps(_mm_set1_ps(1.0f), _mm_sqrt_ps(len2_sq));

        __m128 n1 = _mm_mul_ps(v1, inv_len1);
        __m128 n2 = _mm_mul_ps(v2, inv_len2);

        __m128 dot_prod = _mm_dp_ps(n1, n2, 0xFF);

        float cosine;
        _mm_store_ss(&cosine, dot_prod);
    
        
        cosine = std::clamp(cosine, -1.0f, 1.0f);

        return std::acos(cosine);
    }

    
    Vec4 Vec4::Lerp(const Vec4& a, const Vec4& b, float t)
    {
        return store(_mm_add_ps(_mm_mul_ps(_mm_sub_ps(load(b), load(a)), _mm_set1_ps(t)), load(a)));
    }

    
    Vec4 Vec4::Min(const Vec4& a, const Vec4& b)
    {
        return store(_mm_min_ps(load(a), load(b)));
    }

    
    Vec4 Vec4::Max(const Vec4& a, const Vec4& b)
    {
        return store(_mm_max_ps(load(a), load(b)));
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