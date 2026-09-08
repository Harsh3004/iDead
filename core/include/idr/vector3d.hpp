#pragma once

#include <cmath>

namespace idr {

struct Vector3d {
    double x{0.0};
    double y{0.0};
    double z{0.0};

    constexpr Vector3d() = default;
    constexpr Vector3d(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}

    Vector3d operator+(const Vector3d& o) const noexcept {
        return {x + o.x, y + o.y, z + o.z};
    }

    Vector3d operator-(const Vector3d& o) const noexcept {
        return {x - o.x, y - o.y, z - o.z};
    }

    Vector3d operator*(double s) const noexcept {
        return {x * s, y * s, z * s};
    }

    Vector3d operator/(double s) const noexcept {
        return {x / s, y / s, z / s};
    }

    Vector3d& operator+=(const Vector3d& o) noexcept {
        x += o.x;
        y += o.y;
        z += o.z;
        return *this;
    }

    double dot(const Vector3d& o) const noexcept {
        return x * o.x + y * o.y + z * o.z;
    }

    Vector3d cross(const Vector3d& o) const noexcept {
        return {
            y * o.z - z * o.y,
            z * o.x - x * o.z,
            x * o.y - y * o.x
        };
    }

    double normSq() const noexcept {
        return x * x + y * y + z * z;
    }

    double norm() const noexcept {
        return std::sqrt(normSq());
    }

    Vector3d normalized() const noexcept {
        const double n = norm();
        if (n > 1e-15) {
            return *this / n;
        }
        return *this;
    }
};

inline Vector3d operator*(double s, const Vector3d& v) noexcept {
    return v * s;
}

} // namespace idr
