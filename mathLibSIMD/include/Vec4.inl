/*
namespace simd
{
    template <std::floating_point T>
    Vec4<T>::Vec4() : x(0), y(0), z(0), w(0)
    {

    }

    template <std::floating_point T>
    Vec4<T>::Vec4(T _x, T _y, T _z, T _w) : x(_x), y(_y), z(_z), w(_w)
    {

    }

    template <std::floating_point T>
    template <std::floating_point U>
    Vec4<T>::Vec4(const Vec4<U>& other) : x(static_cast<T>(other.x)), y(static_cast<T>(other.y)), z(static_cast<T>(other.z)), w(static_cast<T>(other.w))
    {

    }

    template <std::floating_point T>
    __m128 Vec4<T>::load(const Vec4& v)
    {
        return _mm_set_ps(v.w, v.z, v.y, v.x);
    }

    template <std::floating_point T>
    Vec4<T> Vec4<T>::store(__m128 val)
    {
        float components[4];

        _mm_storeu_ps(components, val);

        return {
            components[0],
            components[1],
            components[2],
            components[3]
        };
    }

    template <std::floating_point T>
    Vec4<T> Vec4<T>::operator+(const Vec4& rhs) const
    {
        return store(_mm_add_ps(load(*this), load(rhs)));
    }

    template <std::floating_point T>
    Vec4<T> Vec4<T>::operator-(const Vec4& rhs) const
    {
        return store(_mm_sub_ps(load(*this), load(rhs)));
    };

    template <std::floating_point T>
    Vec4<T> Vec4<T>::operator*(const Vec4& rhs) const
    {
        return store(_mm_mul_ps(load(*this), load(rhs)));
    }

    template <std::floating_point T>
    Vec4<T> Vec4<T>::operator/(const Vec4& rhs) const
    {
        if (rhs.x == T{0} || rhs.y == T{0} || rhs.z == T{0} || rhs.w == T{0})
        {
            throw std::domain_error("Cannot divide by a zero component");
        }

        return store(_mm_div_ps(load(*this), load(rhs)));
    }

    template <std::floating_point T>
    Vec4<T> Vec4<T>::operator-() const
    {
        // Negate the vector by subtracting it from zero
        return store(_mm_sub_ps(_mm_set1_ps(0.0f), load(*this)));
    }

    template <std::floating_point T>
    Vec4<T> Vec4<T>::operator*(T scalar) const
    {
        return store(_mm_mul_ps(load(*this), _mm_set1_ps(scalar)));
    }

    template <std::floating_point T>
    Vec4<T> Vec4<T>::operator/(T scalar) const
    {
        if (scalar == T{0})
        {
            throw std::domain_error("Cannot divide by zero");
        }

        return store(_mm_div_ps(load(*this), _mm_set1_ps(scalar)));
    }

    template <std::floating_point T>
    Vec4<T>& Vec4<T>::operator+=(const Vec4& rhs)
    {
        return store(_mm_add_ps(load(*this), load(rhs)));
    }

    template <std::floating_point T>
    Vec4<T>& Vec4<T>::operator-=(const Vec4& rhs)
    {
        return store(_mm_sub_ps(load(*this), load(rhs)));
    }

    template <std::floating_point T>
    Vec4<T>& Vec4<T>::operator*=(const Vec4& rhs)
    {
        return store(_mm_mul_ps(load(*this), load(rhs)));
    }

    template <std::floating_point T>
    Vec4<T>& Vec4<T>::operator/=(const Vec4& rhs)
    {
        return store(_mm_div_ps(load(*this), load(rhs)));
    }

    template <std::floating_point T>
    Vec4<T>& Vec4<T>::operator*=(T scalar)
    {
        return store(_mm_mul_ps(load(*this), _mm_set1_ps(scalar)));
    }

    template <std::floating_point T>
    Vec4<T>& Vec4<T>::operator/=(T scalar)
    {
        return store(_mm_div_ps(load(*this), _mm_set1_ps(scalar)));
    }

    template <std::floating_point T>
    bool Vec4<T>::operator==(const Vec4& rhs) const
    {
        __m128 mask = _mm_cmpeq_ps(load(*this), load(rhs));
        return _mm_movemask_ps(mask) == 0xF;
    }

    template <std::floating_point T>
    bool Vec4<T>::operator!=(const Vec4& rhs) const
    {
        __m128 mask = _mm_cmpneq_ps(load(*this), load(rhs));
        return _mm_movemask_ps(mask) != 0x0;
    }

    template <std::floating_point T>
    T Vec4<T>::Dot(const Vec4& rhs) const
    {
        return store(_mm_dp_ps(load(*this), load(rhs), 0xFF)).x;
    }

    template <std::floating_point T>
    T Vec4<T>::MagnitudeSquared() const
    {
        return store(_mm_dp_ps(load(*this), load(*this), 0xFF)).x;
    }

    template <std::floating_point T>
    T Vec4<T>::Magnitude() const
    {
        return store(_mm_sqrt_ps(_mm_dp_ps(load(*this), load(*this), 0xFF))).x;
    }

    template <std::floating_point T>
    Vec4<T> Vec4<T>::Normalize() const
    {
        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z) || !std::isfinite(w))
        {
            throw std::domain_error("Cannot normalize non-finite components");
        }

        // Scaling avoids overflowing/underflowing the squared magnitude.
        const T scale = std::max({std::abs(x), std::abs(y), std::abs(z), std::abs(w)});
        if (scale == T{0})
        {
            throw std::domain_error("Cannot normalize the zero vector");
        }

        
        return store(_mm_div_ps(load(*this), _mm_set1_ps(Magnitude())));
    }

    template <std::floating_point T>
    T Vec4<T>::DistanceSquared(const Vec4& rhs) const
    {
        return store(_mm_dp_ps(_mm_sub_ps(load(*this), load(rhs)), _mm_sub_ps(load(*this), load(rhs)), 0xFF)).x;
    }

    template <std::floating_point T>
    T Vec4<T>::Distance(const Vec4& rhs) const
    {
        return store(_mm_sqrt_ps(_mm_dp_ps(_mm_sub_ps(load(*this), load(rhs)), _mm_sub_ps(load(*this), load(rhs)), 0xFF))).x;
    }

    template <std::floating_point T>
    T Vec4<T>::Angle(const Vec4& rhs) const
    {
        const T cosine = Normalize().Dot(rhs.Normalize());
        return store(_mm_acos_ps(_mm_set1_ps(std::clamp(cosine, T{-1}, T{1}))));
    }

    template <std::floating_point T>
    Vec4<T> Vec4<T>::Lerp(const Vec4& a, const Vec4& b, T t)
    {
        return store(_mm_add_ps(_mm_mul_ps(_mm_sub_ps(load(b), load(a)), _mm_set1_ps(t)), load(a)));
    }

    template <std::floating_point T>
    Vec4<T> Vec4<T>::Min(const Vec4& a, const Vec4& b)
    {
        return store(_mm_min_ps(load(a), load(b)));
    }

    template <std::floating_point T>
    Vec4<T> Vec4<T>::Max(const Vec4& a, const Vec4& b)
    {
        return store(_mm_max_ps(load(a), load(b)));
    }

    template <std::floating_point T>
    const Vec4<T> Vec4<T>::Zero{0, 0, 0, 0};

    template <std::floating_point T>
    const Vec4<T> Vec4<T>::One{1, 1, 1, 1};

    template <std::floating_point T>
    const Vec4<T> Vec4<T>::UnitX{1, 0, 0, 0};

    template <std::floating_point T>
    const Vec4<T> Vec4<T>::UnitY{0, 1, 0, 0};

    template <std::floating_point T>
    const Vec4<T> Vec4<T>::UnitZ{0, 0, 1, 0};

    template <std::floating_point T>
    const Vec4<T> Vec4<T>::UnitW{0, 0, 0, 1};

    template <std::floating_point T>
    Vec4<T> operator*(T scalar, const Vec4<T>& vector)
    {
        return vector * scalar;
    }

    using Vec4f = Vec4<float>;
    using Vec4d = Vec4<double>;
}
*/