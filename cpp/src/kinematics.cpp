// WaymoCoverageAnalyzer — C++ kinematic feature engine implementation
// Purpose: Compute speed, acceleration, jerk, curvature, path_length, TTC per trajectory
// Author: <your-name>

#include "kinematics.h"

#include <Eigen/Dense>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

namespace {

constexpr double kTtcNotFound = 999.0;
constexpr double kEpsilon = 1e-9;

/// Collect indices of valid timesteps.
std::vector<std::size_t> valid_indices(const AgentTrajectory& traj) {
    std::vector<std::size_t> indices;
    indices.reserve(traj.valid.size());
    for (std::size_t idx = 0; idx < traj.valid.size(); ++idx) {
        if (traj.valid[idx]) {
            indices.push_back(idx);
        }
    }
    return indices;
}

/// Compute speed magnitude at each valid index.
std::vector<double> compute_speeds(
    const AgentTrajectory& traj,
    const std::vector<std::size_t>& valid_idx
) {
    std::vector<double> speeds;
    speeds.reserve(valid_idx.size());
    for (std::size_t idx : valid_idx) {
        const double vx = traj.velocities_vx[idx];
        const double vy = traj.velocities_vy[idx];
        speeds.push_back(std::sqrt(vx * vx + vy * vy));
    }
    return speeds;
}

/// Forward-difference derivative of a sequence: result[i] = (seq[i+1]-seq[i]) / dt
std::vector<double> finite_diff(const std::vector<double>& seq, double dt) {
    if (seq.size() < 2) return {};
    std::vector<double> diff;
    diff.reserve(seq.size() - 1);
    for (std::size_t i = 0; i + 1 < seq.size(); ++i) {
        diff.push_back((seq[i + 1] - seq[i]) / dt);
    }
    return diff;
}

double safe_mean(const std::vector<double>& values) {
    if (values.empty()) return 0.0;
    return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
}

double safe_max_abs(const std::vector<double>& values) {
    if (values.empty()) return 0.0;
    double result = 0.0;
    for (double v : values) {
        result = std::max(result, std::abs(v));
    }
    return result;
}

/// Circular variance of a sequence of headings (radians).
///
/// Uses the mean resultant length R of unit vectors on the circle:
///   R = |mean(exp(i * theta))|
///   circular_variance = 1 - R  (range [0, 1]; 0 = all identical, 1 = maximally dispersed)
///
/// This is wrap-safe: headings at 3.1 and -3.1 rad produce near-zero variance,
/// unlike raw linear variance which would give ~38 rad².
double circular_variance(const std::vector<double>& headings) {
    if (headings.size() < 2) return 0.0;
    double sum_sin = 0.0;
    double sum_cos = 0.0;
    for (double h : headings) {
        sum_sin += std::sin(h);
        sum_cos += std::cos(h);
    }
    const double n = static_cast<double>(headings.size());
    const double mean_resultant_length = std::sqrt(sum_sin * sum_sin + sum_cos * sum_cos) / n;
    return 1.0 - mean_resultant_length;  // range [0, 1]
}

} // anonymous namespace

KinematicFeatures KinematicsEngine::compute(const AgentTrajectory& traj) {
    KinematicFeatures features{};

    const auto valid_idx = valid_indices(traj);
    if (valid_idx.size() < 2) {
        // Single timestep or no data — return zeroed struct without crashing.
        return features;
    }

    // --- Speed ---
    const auto speeds = compute_speeds(traj, valid_idx);
    features.mean_speed = safe_mean(speeds);
    features.max_speed  = *std::max_element(speeds.begin(), speeds.end());

    // Longitudinal acceleration only (rate of change of speed magnitude).
    // Lateral/centripetal component is not captured here — it appears in curvature instead.
    const auto accels = finite_diff(speeds, traj.dt);
    // mean_acceleration is signed: negative means net deceleration over the trajectory.
    // max_acceleration_magnitude is the unsigned peak — use for severity/safety metrics.
    features.mean_acceleration          = safe_mean(accels);
    features.max_acceleration_magnitude = safe_max_abs(accels);

    // --- Jerk (forward difference on longitudinal acceleration) ---
    const auto jerks = finite_diff(accels, traj.dt);
    features.mean_jerk          = safe_mean(jerks);
    features.max_jerk_magnitude = safe_max_abs(jerks);

    // --- Curvature and lateral acceleration, evaluated at each valid step ---
    // 2D cross product: cross_z = vx*ay - vy*ax
    // curvature[i]         = |cross_z| / |v|^3
    // lat_acceleration[i]  = speed^2 * curvature = |cross_z| / |v|
    //   (centripetal acceleration; independent of the longitudinal component)
    {
        std::vector<double> curvatures;
        std::vector<double> lat_accels;
        curvatures.reserve(valid_idx.size() - 1);
        lat_accels.reserve(valid_idx.size() - 1);

        for (std::size_t i = 0; i + 1 < valid_idx.size(); ++i) {
            const std::size_t cur = valid_idx[i];
            const std::size_t nxt = valid_idx[i + 1];
            const double vx    = traj.velocities_vx[cur];
            const double vy    = traj.velocities_vy[cur];
            const double ax    = (traj.velocities_vx[nxt] - vx) / traj.dt;
            const double ay    = (traj.velocities_vy[nxt] - vy) / traj.dt;
            const double speed = std::sqrt(vx * vx + vy * vy);

            if (speed > kEpsilon) {
                const double cross_z   = vx * ay - vy * ax;
                const double abs_cross = std::abs(cross_z);
                curvatures.push_back(abs_cross / (speed * speed * speed));
                lat_accels.push_back(abs_cross / speed);  // = speed^2 * curvature
            } else {
                curvatures.push_back(0.0);
                lat_accels.push_back(0.0);
            }
        }

        features.mean_curvature           = safe_mean(curvatures);
        features.max_curvature            = curvatures.empty() ? 0.0
                                              : *std::max_element(curvatures.begin(), curvatures.end());
        features.mean_lateral_acceleration = safe_mean(lat_accels);
        features.max_lateral_acceleration  = lat_accels.empty() ? 0.0
                                              : *std::max_element(lat_accels.begin(), lat_accels.end());
    }

    // --- Path length and displacement ---
    {
        double path_len = 0.0;
        for (std::size_t i = 0; i + 1 < valid_idx.size(); ++i) {
            const std::size_t cur = valid_idx[i];
            const std::size_t nxt = valid_idx[i + 1];
            const double dx = traj.positions_x[nxt] - traj.positions_x[cur];
            const double dy = traj.positions_y[nxt] - traj.positions_y[cur];
            path_len += std::sqrt(dx * dx + dy * dy);
        }
        features.path_length = path_len;

        const std::size_t first = valid_idx.front();
        const std::size_t last  = valid_idx.back();
        const double disp_x = traj.positions_x[last] - traj.positions_x[first];
        const double disp_y = traj.positions_y[last] - traj.positions_y[first];
        features.displacement = std::sqrt(disp_x * disp_x + disp_y * disp_y);
    }

    // --- Lane change score: circular heading variance × mean_speed ---
    // Uses circular variance (range [0, 1]) so wrap-around at ±π cannot inflate the score.
    // A straight-line agent gets ~0; a lane-changing agent at highway speed gets a large value.
    {
        std::vector<double> headings;
        headings.reserve(valid_idx.size());
        for (std::size_t idx : valid_idx) {
            headings.push_back(traj.headings[idx]);
        }
        features.lane_change_score = circular_variance(headings) * features.mean_speed;
    }

    return features;
}

double KinematicsEngine::compute_ttc(
    const AgentTrajectory& ego,
    const AgentTrajectory& other
) {
    const std::size_t n_steps = std::min(ego.valid.size(), other.valid.size());
    double min_ttc = kTtcNotFound;

    for (std::size_t i = 0; i < n_steps; ++i) {
        if (!ego.valid[i] || !other.valid[i]) continue;

        const Eigen::Vector2d rel_pos{
            other.positions_x[i] - ego.positions_x[i],
            other.positions_y[i] - ego.positions_y[i]
        };
        const Eigen::Vector2d rel_vel{
            other.velocities_vx[i] - ego.velocities_vx[i],
            other.velocities_vy[i] - ego.velocities_vy[i]
        };

        const double denom = rel_vel.dot(rel_vel);
        if (denom < kEpsilon) continue;

        // Point-mass approximation: no bounding-box term.
        // Returns time to closest approach, not true physical collision time.
        // True TTC would subtract the sum of agent radii from |rel_pos| before dividing.
        const double ttc = -rel_pos.dot(rel_vel) / denom;
        if (ttc > 0.0) {
            min_ttc = std::min(min_ttc, ttc);
        }
    }

    return min_ttc;
}
