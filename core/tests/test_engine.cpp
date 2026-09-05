#include "idr/engine.hpp"
#include <iostream>
#include <cmath>
#include <cassert>

#if defined(IDR_USE_GTEST)
#include <gtest/gtest.h>

TEST(IdrEngineTest, InitialStateIsEmpty) {
    idr::IdrEngine engine;
    EXPECT_FALSE(engine.hasImu());
    EXPECT_FALSE(engine.hasGnss());
}

TEST(IdrEngineTest, ProcessImuSample) {
    idr::IdrEngine engine;
    idr::ImuSample sample{
        100.5,              // t (seconds)
        0.12, -0.05, 9.81,  // ax, ay, az (m/s^2)
        0.01, -0.02, 0.005, // gx, gy, gz (rad/s)
        22.4, -14.2, 41.8   // mx, my, mz (uT)
    };

    engine.processImu(sample);

    EXPECT_TRUE(engine.hasImu());
    EXPECT_DOUBLE_EQ(engine.getLastImu().t, 100.5);
    EXPECT_DOUBLE_EQ(engine.getLastImu().az, 9.81);
    EXPECT_DOUBLE_EQ(engine.getLastImu().gx, 0.01);
    EXPECT_DOUBLE_EQ(engine.getLastImu().mz, 41.8);
}

TEST(IdrEngineTest, ProcessGnssFix) {
    idr::IdrEngine engine;
    idr::GnssFix fix{
        101.0,
        12.9716, 77.5946, 920.0,
        5.5,
        2.1,
        14,
        true
    };

    engine.processGnss(fix);

    EXPECT_TRUE(engine.hasGnss());
    EXPECT_DOUBLE_EQ(engine.getLastGnss().t, 101.0);
    EXPECT_DOUBLE_EQ(engine.getLastGnss().lat, 12.9716);
    EXPECT_TRUE(engine.getLastGnss().valid);
}

#elif defined(IDR_USE_CATCH2)
#include <catch2/catch_test_macros.hpp>

TEST_CASE("IdrEngine basic processing", "[engine]") {
    idr::IdrEngine engine;
    idr::ImuSample sample{100.5, 0.12, -0.05, 9.81, 0.01, -0.02, 0.005, 22.4, -14.2, 41.8};
    engine.processImu(sample);
    REQUIRE(engine.hasImu());
    REQUIRE(engine.getLastImu().t == 100.5);
}

#else
// Self-contained, dependency-free test runner
// Executes identically on platforms without external testing libraries
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

    std::cout << "[PASSED] All IdrEngine unit tests succeeded.\n";
    return 0;
}
#endif
