#pragma once

#include "idr/vector3d.hpp"
#include "idr/quaternion.hpp"
#include <array>
#include <cmath>
#include <cstddef>
#include <algorithm>

namespace idr {

/**
 * @brief Zero-dependency, stack-allocated, fixed-size matrix class template.
 *
 * Guaranteed 100% portable across 32-bit MinGW, GCC, Clang, and MSVC.
 * Zero heap allocation, cache-friendly, optimized for small dimension filtering (up to 15x15).
 */
template <size_t Rows, size_t Cols>
class Matrix {
public:
    std::array<double, Rows * Cols> data{};

    constexpr Matrix() = default;

    static constexpr Matrix zeros() noexcept {
        return Matrix{};
    }

    static constexpr Matrix eye() noexcept {
        Matrix m;
        constexpr size_t min_dim = (Rows < Cols) ? Rows : Cols;
        for (size_t i = 0; i < min_dim; ++i) {
            m(i, i) = 1.0;
        }
        return m;
    }

    constexpr double& operator()(size_t r, size_t c) noexcept {
        return data[r * Cols + c];
    }

    constexpr const double& operator()(size_t r, size_t c) const noexcept {
        return data[r * Cols + c];
    }

    Matrix operator+(const Matrix& o) const noexcept {
        Matrix res;
        for (size_t i = 0; i < Rows * Cols; ++i) {
            res.data[i] = data[i] + o.data[i];
        }
        return res;
    }

    Matrix& operator+=(const Matrix& o) noexcept {
        for (size_t i = 0; i < Rows * Cols; ++i) {
            data[i] += o.data[i];
        }
        return *this;
    }

    Matrix operator-(const Matrix& o) const noexcept {
        Matrix res;
        for (size_t i = 0; i < Rows * Cols; ++i) {
            res.data[i] = data[i] - o.data[i];
        }
        return res;
    }

    Matrix& operator-=(const Matrix& o) noexcept {
        for (size_t i = 0; i < Rows * Cols; ++i) {
            data[i] -= o.data[i];
        }
        return *this;
    }

    Matrix operator*(double s) const noexcept {
        Matrix res;
        for (size_t i = 0; i < Rows * Cols; ++i) {
            res.data[i] = data[i] * s;
        }
        return res;
    }

    Matrix operator/(double s) const noexcept {
        Matrix res;
        const double inv = 1.0 / s;
        for (size_t i = 0; i < Rows * Cols; ++i) {
            res.data[i] = data[i] * inv;
        }
        return res;
    }

    template <size_t OtherCols>
    Matrix<Rows, OtherCols> operator*(const Matrix<Cols, OtherCols>& o) const noexcept {
        Matrix<Rows, OtherCols> res;
        for (size_t i = 0; i < Rows; ++i) {
            for (size_t k = 0; k < Cols; ++k) {
                const double s = (*this)(i, k);
                if (std::abs(s) < 1e-15) continue;
                for (size_t j = 0; j < OtherCols; ++j) {
                    res(i, j) += s * o(k, j);
                }
            }
        }
        return res;
    }

    Matrix<Cols, Rows> transpose() const noexcept {
        Matrix<Cols, Rows> res;
        for (size_t r = 0; r < Rows; ++r) {
            for (size_t c = 0; c < Cols; ++c) {
                res(c, r) = (*this)(r, c);
            }
        }
        return res;
    }

    double trace() const noexcept {
        constexpr size_t min_dim = (Rows < Cols) ? Rows : Cols;
        double tr = 0.0;
        for (size_t i = 0; i < min_dim; ++i) {
            tr += (*this)(i, i);
        }
        return tr;
    }

    Matrix symmetrized() const noexcept {
        static_assert(Rows == Cols, "Symmetrization requires a square matrix");
        Matrix res;
        for (size_t i = 0; i < Rows; ++i) {
            res(i, i) = (*this)(i, i);
            for (size_t j = i + 1; j < Cols; ++j) {
                const double avg = 0.5 * ((*this)(i, j) + (*this)(j, i));
                res(i, j) = avg;
                res(j, i) = avg;
            }
        }
        return res;
    }

    template <size_t BlockRows, size_t BlockCols>
    void setBlock(size_t start_r, size_t start_c, const Matrix<BlockRows, BlockCols>& block) noexcept {
        for (size_t r = 0; r < BlockRows; ++r) {
            for (size_t c = 0; c < BlockCols; ++c) {
                (*this)(start_r + r, start_c + c) = block(r, c);
            }
        }
    }

    template <size_t BlockRows, size_t BlockCols>
    Matrix<BlockRows, BlockCols> getBlock(size_t start_r, size_t start_c) const noexcept {
        Matrix<BlockRows, BlockCols> block;
        for (size_t r = 0; r < BlockRows; ++r) {
            for (size_t c = 0; c < BlockCols; ++c) {
                block(r, c) = (*this)(start_r + r, start_c + c);
            }
        }
        return block;
    }

    static Matrix<3, 3> skewSymmetric(const Vector3d& v) noexcept {
        Matrix<3, 3> m;
        m(0, 0) = 0.0;  m(0, 1) = -v.z; m(0, 2) = v.y;
        m(1, 0) = v.z;  m(1, 1) = 0.0;  m(1, 2) = -v.x;
        m(2, 0) = -v.y; m(2, 1) = v.x;  m(2, 2) = 0.0;
        return m;
    }

    static Matrix<3, 3> fromQuaternion(const Quaternion& q) noexcept {
        const double w = q.w, x = q.x, y = q.y, z = q.z;
        Matrix<3, 3> R;
        R(0, 0) = 1.0 - 2.0 * (y * y + z * z);
        R(0, 1) = 2.0 * (x * y - w * z);
        R(0, 2) = 2.0 * (x * z + w * y);

        R(1, 0) = 2.0 * (x * y + w * z);
        R(1, 1) = 1.0 - 2.0 * (x * x + z * z);
        R(1, 2) = 2.0 * (y * z - w * x);

        R(2, 0) = 2.0 * (x * z - w * y);
        R(2, 1) = 2.0 * (y * z + w * x);
        R(2, 2) = 1.0 - 2.0 * (x * x + y * y);
        return R;
    }
};

template <size_t Rows, size_t Cols>
inline Matrix<Rows, Cols> operator*(double s, const Matrix<Rows, Cols>& m) noexcept {
    return m * s;
}

/**
 * @brief Analytical 2x2 matrix inversion.
 */
inline bool invert2x2(const Matrix<2, 2>& m, Matrix<2, 2>& inv) noexcept {
    const double det = m(0, 0) * m(1, 1) - m(0, 1) * m(1, 0);
    if (std::abs(det) < 1e-15) {
        return false;
    }
    const double inv_det = 1.0 / det;
    inv(0, 0) =  m(1, 1) * inv_det;
    inv(0, 1) = -m(0, 1) * inv_det;
    inv(1, 0) = -m(1, 0) * inv_det;
    inv(1, 1) =  m(0, 0) * inv_det;
    return true;
}

/**
 * @brief Analytical 3x3 matrix inversion via Cramer's rule / cofactor adjugate.
 */
inline bool invert3x3(const Matrix<3, 3>& m, Matrix<3, 3>& inv) noexcept {
    const double a00 = m(0, 0), a01 = m(0, 1), a02 = m(0, 2);
    const double a10 = m(1, 0), a11 = m(1, 1), a12 = m(1, 2);
    const double a20 = m(2, 0), a21 = m(2, 1), a22 = m(2, 2);

    const double c00 = a11 * a22 - a12 * a21;
    const double c01 = -(a10 * a22 - a12 * a20);
    const double c02 = a10 * a21 - a11 * a20;

    const double det = a00 * c00 + a01 * c01 + a02 * c02;
    if (std::abs(det) < 1e-15) {
        return false;
    }
    const double inv_det = 1.0 / det;

    inv(0, 0) = c00 * inv_det;
    inv(0, 1) = (a02 * a21 - a01 * a22) * inv_det;
    inv(0, 2) = (a01 * a12 - a02 * a11) * inv_det;

    inv(1, 0) = c01 * inv_det;
    inv(1, 1) = (a00 * a22 - a02 * a20) * inv_det;
    inv(1, 2) = (a02 * a10 - a00 * a12) * inv_det;

    inv(2, 0) = c02 * inv_det;
    inv(2, 1) = (a01 * a20 - a00 * a21) * inv_det;
    inv(2, 2) = (a00 * a11 - a01 * a10) * inv_det;

    return true;
}

} // namespace idr
