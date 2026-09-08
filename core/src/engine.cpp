#include "idr/engine.hpp"

namespace idr {

IdrEngine::IdrEngine() = default;

void IdrEngine::processImu(const ImuSample& sample) {
    last_imu_ = sample;
    has_imu_ = true;

    if (strapdown_.isInitialized()) {
        strapdown_.update(sample);
    }
}

void IdrEngine::processGnss(const GnssFix& fix) {
    last_gnss_ = fix;
    has_gnss_ = true;
}

void IdrEngine::initializeStrapdown(
    double t0,
    double lat0,
    double lon0,
    double alt0,
    double speed_ms,
    double heading_deg,
    const Vector3d& initial_accel
) noexcept {
    strapdown_.initialize(t0, lat0, lon0, alt0, speed_ms, heading_deg, initial_accel);
}

const ImuSample& IdrEngine::getLastImu() const {
    return last_imu_;
}

const GnssFix& IdrEngine::getLastGnss() const {
    return last_gnss_;
}

bool IdrEngine::hasImu() const noexcept {
    return has_imu_;
}

bool IdrEngine::hasGnss() const noexcept {
    return has_gnss_;
}

void IdrEngine::reset() noexcept {
    last_imu_ = ImuSample{};
    last_gnss_ = GnssFix{};
    has_imu_ = false;
    has_gnss_ = false;
    strapdown_.reset();
}

} // namespace idr

