#include "idr/engine.hpp"
#include <cmath>
#include <iostream>
#include <cassert>

#if __has_include(<catch2/catch_test_macros.hpp>)
#include <catch2/catch_test_macros.hpp>

TEST_CASE("IdrEngine processes synthetic ImuSample", "[engine]") {
    idr::IdrEngine engine;

    SECTION("Engine initializes with no samples") {
        REQUIRE_FALSE(engine.hasImu());
        REQUIRE_FALSE(engine.hasGnss());
    }

    SECTION("Process IMU sample and retain values") {
        idr::ImuSample sample{
            100.5,              // t (seconds)
            0.12, -0.05, 9.81,  // ax, ay, az (m/s^2)
            0.01, -0.02, 0.005, // gx, gy, gz (rad/s)
            22.4, -14.2, 41.8   // mx, my, mz (uT)
        };

        engine.processImu(sample);

        REQUIRE(engine.hasImu());
        REQUIRE(engine.getLastImu().t == 100.5);
        REQUIRE(engine.getLastImu().az == 9.81);
        REQUIRE(engine.getLastImu().gx == 0.01);
        REQUIRE(engine.getLastImu().mz == 41.8);
    }

    SECTION("Process GNSS fix and retain values") {
        idr::GnssFix fix{
            101.0,
            12.9716, 77.5946, 920.0,
            5.5,
            2.1,
            14,
            true
        };

        engine.processGnss(fix);

        REQUIRE(engine.hasGnss());
        REQUIRE(engine.getLastGnss().t == 101.0);
        REQUIRE(engine.getLastGnss().lat == 12.9716);
        REQUIRE(engine.getLastGnss().valid);
    }
}
#else
// Fallback test runner when compiled without Catch2 dependency
int main() {
    idr::IdrEngine engine;
    assert(!engine.hasImu());
    assert(!engine.hasGnss());

    idr::ImuSample sample{
        100.5,
        0.12, -0.05, 9.81,
        0.01, -0.02, 0.005,
        22.4, -14.2, 41.8
    };

    engine.processImu(sample);
    assert(engine.hasImu());
    assert(engine.getLastImu().t == 100.5);
    assert(engine.getLastImu().az == 9.81);
    assert(engine.getLastImu().gx == 0.01);
    assert(engine.getLastImu().mz == 41.8);

    idr::GnssFix fix{
        101.0,
        12.9716, 77.5946, 920.0,
        5.5,
        2.1,
        14,
        true
    };
    engine.processGnss(fix);
    assert(engine.hasGnss());
    assert(engine.getLastGnss().t == 101.0);
    assert(engine.getLastGnss().lat == 12.9716);
    assert(engine.getLastGnss().valid);

    std::cout << "[PASS] IdrEngine unit tests passed successfully.\n";
    return 0;
}
#endif
