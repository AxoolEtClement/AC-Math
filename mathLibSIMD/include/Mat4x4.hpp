/**
 * @file Mat4x4.hpp
 * @brief Implémentation d'une matrice 4x4 alignée pour transformations affines.
 */

#pragma once

#include "Vec4.hpp"
#include "Vec3f.hpp"
#include <array>
#include <limits>
#include <utility>
#include <immintrin.h>

namespace simd
{
    // Row-major storage, column vectors: result = matrix * vector.
    // A * B applies B first. Angles are radians, right-handed rotations.
    /**
     * @class Matrix4x4
     * @brief Matrice de transformation 4x4 optimisée SIMD.
     *
     * \page conventions Conventions :
     * - Stockage : Row-major (ligne par ligne).
     * - Vecteurs : Considérés comme des vecteurs colonnes.
     * - Composition : Le résultat est calculé comme \f$ v' = M \times v \f$.
     *   L'application de plusieurs matrices \f$ A \times B \times v \f$ applique B puis A.
     * - Alignement : Stockée sous forme de 4 registres __m128 alignés.
     */
    class Matrix4x4
    {
    public:
        Matrix4x4(); // Identity.
        explicit Matrix4x4(const std::array<float, 16>& elements);
        static Matrix4x4 Identity();
        static Matrix4x4 Zero();
        float& operator()(std::size_t row, std::size_t column);
        const float& operator()(std::size_t row, std::size_t column) const;
        Matrix4x4 operator+(const Matrix4x4& rhs) const;
        Matrix4x4 operator-(const Matrix4x4& rhs) const;
        Matrix4x4 operator*(const Matrix4x4& rhs) const;
        Vec4 operator*(const Vec4& rhs) const;
        Matrix4x4 operator*(float scalar) const;
        Matrix4x4& operator*=(const Matrix4x4& rhs);
        bool operator==(const Matrix4x4& rhs) const;
        bool operator!=(const Matrix4x4& rhs) const;
        Matrix4x4 Transpose() const;
        float Determinant() const;
        // Gauss-Jordan with scaled partial pivoting. Rejects singular/ill-conditioned input.
        Matrix4x4 Inverse(float relativeTolerance = float{ 64 } *std::numeric_limits<float>::epsilon()) const;
        static Matrix4x4 Scale(const Vec3f& scale);
        static Matrix4x4 RotationX(float radians);
        static Matrix4x4 RotationY(float radians);
        static Matrix4x4 RotationZ(float radians);
        static Matrix4x4 Translation(const Vec3f& offset);
        /**
         * @brief Transforme un point 3D.
         *
         * Utilise une coordonnée homogène \f$ w = 1 \f$ pour les points.
         * Transformation affine uniquement, sans division perspective.
         * @param[in] point Le point à transformer.
         * @return Le point transformé.
         * @see conventions
         */
        Vec3f TransformPoint(const Vec3f& point) const; // Affine only, w = 1.
        Vec3f TransformDirection(const Vec3f& direction) const; // Affine only, w = 0.

    private:
        std::array<__m128, 4> _data;
    };

    /**
     * @brief Transforme un lot de points 3D par une même matrice.
     *
     * La transformation utilise une coordonnée homogène w = 1 et ne réalise
     * pas de division perspective.
     *
     * @param[in] mat La matrice de transformation.
     * @param[in] input Pointeur vers le tableau de points d'entrée.
     * @param[out] output Pointeur vers le tableau de points de sortie.
     * @param[in] count Nombre de points à transformer.
     * @pre input et output doivent être alignés sur 16 octets.
     * @note Gère les tailles de lots non multiples de la largeur SIMD.
     * @see Matrix4x4::TransformPoint
     */
    void BatchTransformPoint(const Matrix4x4& mat, const Vec3f* input, Vec3f* output, std::size_t count);

    /**
     * @brief Référence scalaire de BatchTransformPoint pour les comparaisons.
     * @param[in] mat Matrice affine commune.
     * @param[in] input Tableau de points.
     * @param[out] output Tableau destination.
     * @param[in] count Nombre de points.
     */
    void BatchTransformPointReference(const Matrix4x4& mat, const Vec3f* input,
                                      Vec3f* output, std::size_t count);
}
