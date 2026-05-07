# WaymoCoverageAnalyzer — Feature extraction unit tests
# Purpose: Verify C++ engine import, feature correctness on synthetic scenarios
# Author: <your-name>

"""Tests for waymo_coverage.features: C++ engine bridge and feature correctness."""

import numpy as np
import pytest

from waymo_coverage.features import ScenarioFeatureVector, extract_features
from waymo_coverage.parser import AgentState, ScenarioData

_DT = 0.1
_N_STEPS = 91
_TOL = 1e-5


def test_cpp_engine_importable() -> None:
    """waymo_kinematics C++ module must be importable after build."""
    from waymo_coverage import waymo_kinematics  # noqa: F401 — import-only test

    assert hasattr(waymo_kinematics, "KinematicsEngine")
    assert hasattr(waymo_kinematics, "KinematicFeatures")


def test_mean_speed_constant_velocity(straight_scenario: ScenarioData) -> None:
    """Mean speed for a constant-velocity agent must equal the configured speed."""
    fv = extract_features(straight_scenario)
    assert abs(fv.mean_agent_speed - 10.0) < _TOL, (
        f"Expected mean_speed=10.0, got {fv.mean_agent_speed}"
    )


def test_max_speed_constant_velocity(straight_scenario: ScenarioData) -> None:
    """Max speed for a constant-velocity agent must equal the configured speed."""
    fv = extract_features(straight_scenario)
    assert abs(fv.max_agent_speed - 10.0) < _TOL


def test_mean_acceleration_braking_is_signed_negative(braking_scenario: ScenarioData) -> None:
    """Mean acceleration for a braking agent must be negative (deceleration)."""
    fv = extract_features(braking_scenario)
    expected_decel = 10.0 / ((_N_STEPS - 1) * _DT)
    # mean_acceleration is signed: braking → negative value.
    assert fv.mean_agent_acceleration < 0.0, (
        f"Expected negative mean_acceleration for braking, got {fv.mean_agent_acceleration:.4f}"
    )
    assert abs(fv.mean_agent_acceleration - (-expected_decel)) < 0.05, (
        f"Expected mean_acceleration ≈ -{expected_decel:.4f}, got {fv.mean_agent_acceleration:.4f}"
    )


def test_ttc_collision_course(pedestrian_crossing_scenario: ScenarioData) -> None:
    """TTC must be finite (< 999) for agents on converging paths."""
    fv = extract_features(pedestrian_crossing_scenario)
    assert fv.min_ttc < 999.0, (
        f"Expected finite TTC for crossing agents, got {fv.min_ttc}"
    )


def test_all_invalid_trajectory_no_crash() -> None:
    """Feature extraction on an all-invalid agent must not crash and return zero speed."""
    invalid_agent = AgentState(
        positions_x=[0.0] * 10,
        positions_y=[0.0] * 10,
        velocities_vx=[5.0] * 10,
        velocities_vy=[0.0] * 10,
        headings=[0.0] * 10,
        valid=[False] * 10,
        object_type=1,
    )
    scenario = ScenarioData(
        scenario_id="invalid_001",
        timestamps=[idx * _DT for idx in range(10)],
        agents=[invalid_agent],
        sdc_track_index=0,
    )
    fv = extract_features(scenario)
    assert fv.mean_agent_speed == 0.0
    assert fv.max_agent_speed == 0.0


def test_feature_vector_length(straight_scenario: ScenarioData) -> None:
    """Feature vector must have a fixed length matching the defined feature count."""
    fv = extract_features(straight_scenario)
    assert len(fv.feature_vector) == 15  # 15 defined features


def test_lateral_acceleration_zero_straight_line(straight_scenario: ScenarioData) -> None:
    """Lateral acceleration must be near zero for a constant straight-line trajectory."""
    fv = extract_features(straight_scenario)
    assert fv.mean_lateral_acceleration < 1e-6, (
        f"Expected ~0 lateral acceleration on straight line, got {fv.mean_lateral_acceleration}"
    )
    assert fv.max_lateral_acceleration < 1e-6


def test_feature_vector_all_finite(multi_agent_scenario: ScenarioData) -> None:
    """All feature values must be finite (no NaN or Inf)."""
    fv = extract_features(multi_agent_scenario)
    for value in fv.feature_vector:
        assert np.isfinite(value), f"Non-finite feature value: {value}"


def test_num_agents_counts_correctly(multi_agent_scenario: ScenarioData) -> None:
    """num_agents must equal the total agent count in the scenario."""
    fv = extract_features(multi_agent_scenario)
    assert fv.num_agents == len(multi_agent_scenario.agents)


def test_cpp_engine_direct_call() -> None:
    """Direct call to KinematicsEngine.compute must return correct mean_speed."""
    from waymo_coverage import waymo_kinematics

    n_steps = 91
    speed = 5.0
    px  = np.linspace(0.0, speed * (n_steps - 1) * _DT, n_steps)
    py  = np.zeros(n_steps)
    vx  = np.full(n_steps, speed)
    vy  = np.zeros(n_steps)
    hdg = np.zeros(n_steps)
    valid = np.ones(n_steps, dtype=bool)

    result = waymo_kinematics.KinematicsEngine.compute(px, py, vx, vy, hdg, valid, _DT)
    assert abs(result.mean_speed - speed) < _TOL
