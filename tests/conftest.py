# WaymoCoverageAnalyzer — pytest fixtures
# Purpose: Provide synthetic ScenarioData fixtures that require no real tfrecord files
# Author: <your-name>

"""Shared pytest fixtures: synthetic ScenarioData for all test modules."""

import math

import pytest

from waymo_coverage.parser import AgentState, ScenarioData

# Waymo Motion dataset timestep and standard scenario length.
_DT = 0.1
_N_STEPS = 91
_TIMESTAMPS = [round(idx * _DT, 1) for idx in range(_N_STEPS)]

# Waymo ObjectType constants.
_TYPE_VEHICLE    = 1
_TYPE_PEDESTRIAN = 2
_TYPE_CYCLIST    = 3


def _make_agent(
    *,
    speed: float = 10.0,
    decel: float = 0.0,
    start_x: float = 0.0,
    start_y: float = 0.0,
    heading: float = 0.0,
    n_steps: int = _N_STEPS,
    object_type: int = _TYPE_VEHICLE,
    all_invalid: bool = False,
) -> AgentState:
    """Build a synthetic AgentState with configurable kinematics.

    Args:
        speed: Initial speed in m/s (along x-axis, heading direction).
        decel: Constant deceleration magnitude (m/s^2). Speed clamps to zero.
        start_x: Initial x position.
        start_y: Initial y position.
        heading: Constant heading in radians.
        n_steps: Number of timesteps.
        object_type: Waymo Track.ObjectType enum value.
        all_invalid: If True, all valid flags are False.

    Returns:
        AgentState with analytically derived positions and velocities.
    """
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)

    positions_x: list[float] = []
    positions_y: list[float] = []
    velocities_vx: list[float] = []
    velocities_vy: list[float] = []
    headings: list[float] = []
    valid: list[bool] = []

    current_x = start_x
    current_y = start_y

    for step_idx in range(n_steps):
        current_speed = max(0.0, speed - decel * step_idx * _DT)
        positions_x.append(current_x)
        positions_y.append(current_y)
        velocities_vx.append(current_speed * cos_h)
        velocities_vy.append(current_speed * sin_h)
        headings.append(heading)
        valid.append(not all_invalid)
        current_x += current_speed * cos_h * _DT
        current_y += current_speed * sin_h * _DT

    return AgentState(
        positions_x=positions_x,
        positions_y=positions_y,
        velocities_vx=velocities_vx,
        velocities_vy=velocities_vy,
        headings=headings,
        valid=valid,
        object_type=object_type,
    )


@pytest.fixture
def straight_scenario() -> ScenarioData:
    """Single vehicle at constant 10 m/s along the x-axis for 91 timesteps."""
    vehicle = _make_agent(speed=10.0, decel=0.0, object_type=_TYPE_VEHICLE)
    return ScenarioData(
        scenario_id="straight_001",
        timestamps=_TIMESTAMPS,
        agents=[vehicle],
        sdc_track_index=0,
    )


@pytest.fixture
def braking_scenario() -> ScenarioData:
    """Single vehicle decelerating from 10 m/s to 0 over 91 steps."""
    decel_rate = 10.0 / ((_N_STEPS - 1) * _DT)
    vehicle = _make_agent(speed=10.0, decel=decel_rate, object_type=_TYPE_VEHICLE)
    return ScenarioData(
        scenario_id="braking_001",
        timestamps=_TIMESTAMPS,
        agents=[vehicle],
        sdc_track_index=0,
    )


@pytest.fixture
def pedestrian_crossing_scenario() -> ScenarioData:
    """Vehicle moving in +x, pedestrian moving in +y — paths cross near origin."""
    vehicle = _make_agent(
        speed=8.0, decel=0.0, start_x=-20.0, start_y=0.0,
        heading=0.0, object_type=_TYPE_VEHICLE,
    )
    pedestrian = _make_agent(
        speed=1.5, decel=0.0, start_x=0.0, start_y=-5.0,
        heading=math.pi / 2.0, object_type=_TYPE_PEDESTRIAN,
    )
    return ScenarioData(
        scenario_id="ped_crossing_001",
        timestamps=_TIMESTAMPS,
        agents=[vehicle, pedestrian],
        sdc_track_index=0,
    )


@pytest.fixture
def multi_agent_scenario() -> ScenarioData:
    """Five vehicles with varied speeds and headings."""
    agents = [
        _make_agent(speed=5.0,  heading=0.0,              object_type=_TYPE_VEHICLE),
        _make_agent(speed=12.0, heading=math.pi / 6.0,    object_type=_TYPE_VEHICLE),
        _make_agent(speed=8.0,  heading=math.pi / 3.0,    object_type=_TYPE_VEHICLE),
        _make_agent(speed=3.0,  heading=math.pi,           object_type=_TYPE_VEHICLE, start_x=100.0),
        _make_agent(speed=15.0, heading=-math.pi / 4.0,   object_type=_TYPE_CYCLIST),
    ]
    return ScenarioData(
        scenario_id="multi_agent_001",
        timestamps=_TIMESTAMPS,
        agents=agents,
        sdc_track_index=0,
    )
