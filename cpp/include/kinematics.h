// WaymoCoverageAnalyzer — C++ kinematic feature engine header
// Purpose: Defines KinematicFeatures, AgentTrajectory, and KinematicsEngine API
// Author: <your-name>

#pragma once

#include <span>
#include <vector>
#include <cstdint>

/// All scalar kinematic features extracted from one agent trajectory.
struct KinematicFeatures {
    double mean_speed;
    double max_speed;

    // Longitudinal (speed-magnitude) acceleration only. mean is signed (negative = net
    // braking); max_acceleration_magnitude is the unsigned peak for severity metrics.
    // Lateral/centripetal acceleration is captured separately via curvature.
    double mean_acceleration;
    double max_acceleration_magnitude;

    // Longitudinal jerk (derivative of longitudinal acceleration). mean is signed;
    // max_jerk_magnitude is the unsigned peak.
    double mean_jerk;
    double max_jerk_magnitude;

    double mean_curvature;
    double max_curvature;

    // Lateral (centripetal) acceleration: a_lat = speed² × curvature = |v × a| / speed.
    // Complements longitudinal acceleration — captures cornering g-forces that the
    // speed-magnitude derivative misses entirely.
    double mean_lateral_acceleration;
    double max_lateral_acceleration;

    double path_length;
    double displacement;

    // Circular heading variance (range [0,1], wrap-safe) × mean_speed.
    // Straight-line agents → ~0; lane-changing agents at speed → large value.
    double lane_change_score;
    // interaction_score is a scene-level metric that requires all agents' positions
    // simultaneously; it cannot be computed from a single trajectory.
    // Computed in Python as scene_interaction_density and kept there.
};

/// Non-owning view into per-timestep trajectory data.
///
/// All spans must have the same length. The `valid` span must point to a contiguous
/// sequence of bool-sized values (e.g. std::vector<uint8_t> or a C array of bool).
/// Do NOT pass std::vector<bool>::data() — std::vector<bool> is a bitfield and is
/// not contiguous in memory; construct a std::vector<uint8_t> instead.
struct AgentTrajectory {
    std::span<const double> positions_x;
    std::span<const double> positions_y;
    std::span<const double> velocities_vx;
    std::span<const double> velocities_vy;
    std::span<const double> headings;
    std::span<const bool>   valid;
    double dt;  // time delta in seconds (0.1 s for Waymo Motion dataset)
};

class KinematicsEngine {
public:
    /// Compute full kinematic feature set for one agent trajectory.
    /// Returns zeroed struct if no valid timesteps exist.
    [[nodiscard]] static KinematicFeatures compute(const AgentTrajectory& traj);

    /// Compute minimum positive time-to-collision between two agents.
    /// Returns 999.0 when no positive TTC is found (agents not converging).
    [[nodiscard]] static double compute_ttc(
        const AgentTrajectory& ego,
        const AgentTrajectory& other
    );
};
