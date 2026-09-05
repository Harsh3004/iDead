#include "idr/engine.hpp"

namespace idr {

IdrEngine::IdrEngine() = default;

void IdrEngine::processImu(const ImuSample& sample) {
    last_imu_ = sample;
    has_imu_ = true;
}

void IdrEngine::processGnss(const GnssFix& fix) {
    last_gnss_ = fix;
    has_gnss_ = true;
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
}

} // namespace idr
