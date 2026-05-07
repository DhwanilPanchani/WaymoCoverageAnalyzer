# WaymoCoverageAnalyzer — Parser unit tests
# Purpose: Test ScenarioData and AgentState model construction and validation
# Author: <your-name>

"""Tests for waymo_coverage.parser data models and fixture integrity."""

import pytest

from waymo_coverage.parser import AgentState, ScenarioData


def test_straight_scenario_shape(straight_scenario: ScenarioData) -> None:
    """Straight scenario must have exactly one agent and 91 timesteps."""
    assert len(straight_scenario.agents) == 1
    assert len(straight_scenario.timestamps) == 91


def test_agent_state_all_fields_same_length(straight_scenario: ScenarioData) -> None:
    """All trajectory arrays in an AgentState must have equal length."""
    agent = straight_scenario.agents[0]
    lengths = {
        len(agent.positions_x),
        len(agent.positions_y),
        len(agent.velocities_vx),
        len(agent.velocities_vy),
        len(agent.headings),
        len(agent.valid),
    }
    assert len(lengths) == 1, f"Inconsistent array lengths: {lengths}"


def test_all_valid_flags_true(straight_scenario: ScenarioData) -> None:
    """Straight scenario should have all valid flags set to True."""
    agent = straight_scenario.agents[0]
    assert all(agent.valid)


def test_braking_scenario_has_vehicle(braking_scenario: ScenarioData) -> None:
    """Braking scenario must contain exactly one vehicle agent."""
    assert len(braking_scenario.agents) == 1
    assert braking_scenario.agents[0].object_type == 1  # TYPE_VEHICLE


def test_pedestrian_crossing_has_two_agents(pedestrian_crossing_scenario: ScenarioData) -> None:
    """Pedestrian crossing scenario must have exactly two agents."""
    assert len(pedestrian_crossing_scenario.agents) == 2


def test_multi_agent_scenario_count(multi_agent_scenario: ScenarioData) -> None:
    """Multi-agent scenario must have five agents."""
    assert len(multi_agent_scenario.agents) == 5


def test_scenario_data_is_frozen(straight_scenario: ScenarioData) -> None:
    """ScenarioData is a frozen Pydantic model and must reject attribute mutation."""
    with pytest.raises(Exception):
        straight_scenario.scenario_id = "mutated"  # type: ignore[misc]


def test_agent_state_is_frozen(straight_scenario: ScenarioData) -> None:
    """AgentState is a frozen Pydantic model and must reject attribute mutation."""
    agent = straight_scenario.agents[0]
    with pytest.raises(Exception):
        agent.object_type = 99  # type: ignore[misc]


def test_sdc_track_index_in_range(multi_agent_scenario: ScenarioData) -> None:
    """sdc_track_index must refer to a valid agent index."""
    assert 0 <= multi_agent_scenario.sdc_track_index < len(multi_agent_scenario.agents)
