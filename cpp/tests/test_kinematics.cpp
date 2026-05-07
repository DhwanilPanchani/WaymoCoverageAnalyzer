// WaymoCoverageAnalyzer — Google Test suite for C++ kinematics engine
// Purpose: Verify KinematicsEngine correctness on synthetic trajectories and edge cases
// Author: <your-name>

#include "kinematics.h"

#include <cmath>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

namespace {

constexpr double kDt = 0.1;
constexpr double kTol = 1e-6;

/// Build an AgentTrajectory from std::vectors (valid for the duration of the test).
/// valid uses uint8_t instead of bool to avoid std::vector<bool>'s bitfield
/// specialisation, which is not contiguous and cannot construct std::span<const bool>.
struct TrajHolder {
    std::vector<double>   px, py, vx, vy, hdg;
    std::vector<uint8_t>  valid;  // 1 = valid, 0 = invalid

    AgentTrajectory view(double dt = kDt) const {
        return AgentTrajectory{
            .positions_x   = std::span<const double>(px),
            .positions_y   = std::span<const double>(py),
            .velocities_vx = std::span<const double>(vx),
            .velocities_vy = std::span<const double>(vy),
            .headings      = std::span<const double>(hdg),
            .valid         = std::span<const bool>(
                                 reinterpret_cast<const bool*>(valid.data()),
                                 valid.size()),
            .dt            = dt
        };
    }
};

/// Constant-velocity trajectory: moves in +x at speed v for n_steps steps.
TrajHolder make_constant_velocity(double speed, int n_steps) {
    TrajHolder holder;
    holder.px.resize(n_steps);
    holder.py.resize(n_steps, 0.0);
    holder.vx.resize(n_steps, speed);
    holder.vy.resize(n_steps, 0.0);
    holder.hdg.resize(n_steps, 0.0);
    holder.valid.resize(n_steps, true);
    for (int i = 0; i < n_steps; ++i) {
        holder.px[i] = static_cast<double>(i) * speed * kDt;
    }
    return holder;
}

/// Braking trajectory: decelerates from initial_speed to 0 over n_steps steps.
TrajHolder make_braking(double initial_speed, int n_steps) {
    TrajHolder holder;
    holder.px.resize(n_steps);
    holder.py.resize(n_steps, 0.0);
    holder.vx.resize(n_steps);
    holder.vy.resize(n_steps, 0.0);
    holder.hdg.resize(n_steps, 0.0);
    holder.valid.resize(n_steps, true);

    const double decel = initial_speed / (static_cast<double>(n_steps - 1) * kDt);
    double pos = 0.0;
    for (int i = 0; i < n_steps; ++i) {
        const double speed = std::max(0.0, initial_speed - decel * static_cast<double>(i) * kDt);
        holder.vx[i] = speed;
        holder.px[i] = pos;
        pos += speed * kDt;
    }
    return holder;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Constant velocity: mean_speed and max_speed should equal the input speed.
// ---------------------------------------------------------------------------
TEST(KinematicsEngine, ConstantVelocitySpeed) {
    constexpr double kSpeed = 10.0;
    constexpr int    kSteps = 91;
    auto holder = make_constant_velocity(kSpeed, kSteps);
    const auto features = KinematicsEngine::compute(holder.view());

    EXPECT_NEAR(features.mean_speed, kSpeed, kTol);
    EXPECT_NEAR(features.max_speed,  kSpeed, kTol);
    // Constant velocity → zero longitudinal acceleration and jerk
    EXPECT_NEAR(features.mean_acceleration,          0.0, kTol);
    EXPECT_NEAR(features.max_acceleration_magnitude, 0.0, kTol);
    EXPECT_NEAR(features.mean_jerk,                  0.0, kTol);
}

// ---------------------------------------------------------------------------
// Braking: max_acceleration should be approximately the expected deceleration.
// ---------------------------------------------------------------------------
TEST(KinematicsEngine, BrakingMaxAcceleration) {
    constexpr double kInitSpeed = 10.0;
    constexpr int    kSteps     = 91;
    auto holder = make_braking(kInitSpeed, kSteps);
    const auto features = KinematicsEngine::compute(holder.view());

    // Expected deceleration magnitude: speed / ((n-1)*dt)
    const double expected_decel = kInitSpeed / (static_cast<double>(kSteps - 1) * kDt);
    EXPECT_NEAR(features.max_acceleration_magnitude, expected_decel, 1e-4);
    EXPECT_LT(features.mean_speed, kInitSpeed);
    EXPECT_GT(features.mean_speed, 0.0);
}

// ---------------------------------------------------------------------------
// All-invalid trajectory: must return zeroed struct without throwing.
// ---------------------------------------------------------------------------
TEST(KinematicsEngine, AllInvalidTimesteps) {
    constexpr int kSteps = 10;
    TrajHolder holder;
    holder.px.resize(kSteps, 0.0);
    holder.py.resize(kSteps, 0.0);
    holder.vx.resize(kSteps, 5.0);
    holder.vy.resize(kSteps, 0.0);
    holder.hdg.resize(kSteps, 0.0);
    holder.valid.resize(kSteps, false);

    ASSERT_NO_THROW({
        const auto features = KinematicsEngine::compute(holder.view());
        EXPECT_NEAR(features.mean_speed,        0.0, kTol);
        EXPECT_NEAR(features.max_speed,         0.0, kTol);
        EXPECT_NEAR(features.mean_acceleration, 0.0, kTol);
        EXPECT_NEAR(features.path_length,       0.0, kTol);
    });
}

// ---------------------------------------------------------------------------
// TTC: agents on direct collision course should yield a finite, small TTC.
// ---------------------------------------------------------------------------
TEST(KinematicsEngine, TTCCollisionCourse) {
    constexpr int    kSteps  = 91;
    constexpr double kSpeed  = 10.0;
    constexpr double kSep    = 50.0;  // meters apart initially

    // ego moves +x at kSpeed; other moves -x at kSpeed, starting kSep ahead.
    TrajHolder ego, other;
    ego.px.resize(kSteps);   ego.py.resize(kSteps, 0.0);
    ego.vx.resize(kSteps, kSpeed);  ego.vy.resize(kSteps, 0.0);
    ego.hdg.resize(kSteps, 0.0);   ego.valid.resize(kSteps, true);

    other.px.resize(kSteps); other.py.resize(kSteps, 0.0);
    other.vx.resize(kSteps, -kSpeed); other.vy.resize(kSteps, 0.0);
    other.hdg.resize(kSteps, M_PI); other.valid.resize(kSteps, true);

    for (int i = 0; i < kSteps; ++i) {
        ego.px[i]   = static_cast<double>(i) * kSpeed * kDt;
        other.px[i] = kSep - static_cast<double>(i) * kSpeed * kDt;
    }

    const double ttc = KinematicsEngine::compute_ttc(ego.view(), other.view());
    // The engine returns the MINIMUM positive TTC across all timesteps (spec-defined).
    // As the agents advance through the trajectory they get closer, so the minimum TTC
    // approaches the single-timestep look-ahead (dt) just before they cross — which is
    // strictly > 0 and << 999.  We verify it is finite and below the initial-separation TTC.
    const double initial_ttc = kSep / (2.0 * kSpeed);  // = 2.5 s
    EXPECT_LT(ttc, initial_ttc);
    EXPECT_GT(ttc, 0.0);
    EXPECT_LT(ttc, 999.0);
}

// ---------------------------------------------------------------------------
// TTC: diverging agents should return 999.0.
// ---------------------------------------------------------------------------
TEST(KinematicsEngine, TTCDivergingAgents) {
    constexpr int    kSteps = 91;
    constexpr double kSpeed = 10.0;

    // Both agents move in +x; other starts ahead and stays ahead.
    TrajHolder ego, other;
    ego.px.resize(kSteps);   ego.py.resize(kSteps, 0.0);
    ego.vx.resize(kSteps, kSpeed);  ego.vy.resize(kSteps, 0.0);
    ego.hdg.resize(kSteps, 0.0);   ego.valid.resize(kSteps, true);

    other.px.resize(kSteps); other.py.resize(kSteps, 0.0);
    other.vx.resize(kSteps, kSpeed * 2.0); other.vy.resize(kSteps, 0.0);
    other.hdg.resize(kSteps, 0.0); other.valid.resize(kSteps, true);

    for (int i = 0; i < kSteps; ++i) {
        ego.px[i]   = static_cast<double>(i) * kSpeed * kDt;
        other.px[i] = 10.0 + static_cast<double>(i) * kSpeed * 2.0 * kDt;
    }

    const double ttc = KinematicsEngine::compute_ttc(ego.view(), other.view());
    EXPECT_NEAR(ttc, 999.0, kTol);
}

// ---------------------------------------------------------------------------
// Lateral acceleration: straight-line constant velocity → zero.
// ---------------------------------------------------------------------------
TEST(KinematicsEngine, LateralAccelerationStraightLine) {
    constexpr double kSpeed = 15.0;
    constexpr int    kSteps = 91;
    auto holder = make_constant_velocity(kSpeed, kSteps);
    const auto features = KinematicsEngine::compute(holder.view());

    // No turning → zero cross product → zero lateral acceleration.
    EXPECT_NEAR(features.mean_lateral_acceleration, 0.0, kTol);
    EXPECT_NEAR(features.max_lateral_acceleration,  0.0, kTol);
    // Sanity check: curvature also zero on a straight path.
    EXPECT_NEAR(features.mean_curvature, 0.0, kTol);
}

// ---------------------------------------------------------------------------
// Lateral acceleration: circular motion → a_lat = speed² / radius.
// ---------------------------------------------------------------------------
TEST(KinematicsEngine, LateralAccelerationCircularMotion) {
    // Constant-speed circular arc: vx = -speed*sin(omega*t), vy = speed*cos(omega*t)
    // omega = speed / radius; expected lateral accel = speed^2 / radius
    constexpr double kRadius = 20.0;   // metres
    constexpr double kSpeed  = 10.0;   // m/s
    constexpr double kOmega  = kSpeed / kRadius;
    constexpr int    kSteps  = 91;

    TrajHolder holder;
    holder.px.resize(kSteps);
    holder.py.resize(kSteps);
    holder.vx.resize(kSteps);
    holder.vy.resize(kSteps);
    holder.hdg.resize(kSteps, 0.0);
    holder.valid.resize(kSteps, 1);

    for (int i = 0; i < kSteps; ++i) {
        const double t   = static_cast<double>(i) * kDt;
        const double ang = kOmega * t;
        holder.px[i] = kRadius * std::sin(ang);
        holder.py[i] = kRadius * (1.0 - std::cos(ang));
        holder.vx[i] = kSpeed * std::cos(ang);
        holder.vy[i] = kSpeed * std::sin(ang);
    }

    const auto features = KinematicsEngine::compute(holder.view());
    const double expected_lat_accel = kSpeed * kSpeed / kRadius;  // = 5.0 m/s²
    EXPECT_NEAR(features.mean_lateral_acceleration, expected_lat_accel, 0.1);
    EXPECT_GT(features.max_lateral_acceleration, 0.0);
    // Longitudinal acceleration should be near zero (constant speed).
    EXPECT_NEAR(features.mean_acceleration, 0.0, 1e-4);
}
