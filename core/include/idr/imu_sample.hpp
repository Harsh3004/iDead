#pragma once

namespace idr {

struct ImuSample {
    double t;                 // seconds, monotonic
    double ax, ay, az;        // m/s^2, phone/body frame
    double gx, gy, gz;        // rad/s, phone/body frame
    double mx, my, mz;        // uT, phone/body frame (NaN if unavailable)
};

} // namespace idr
