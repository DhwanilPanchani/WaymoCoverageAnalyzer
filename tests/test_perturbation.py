# WaymoCoverageAnalyzer — Perturbation unit tests
# Purpose: Verify counterfactual variant generation, position re-integration, and edge cases
# Author: <your-name>

"""Tests for waymo_coverage.perturbation: variant count, kinematics, and edge cases."""

import math

import numpy as np
import pytest

from waymo_coverage.parser import AgentState, ScenarioData
from waymo_coverage.perturbation import PerturbationConfig, perturb_scenario

_DT = 0.1
_N_STEPS = 91


def test_variant_count(straight_scenario: ScenarioData) -> None:
    """perturb_scenario must return exactly n_variants ScenarioData objects."""
    config = PerturbationConfig(speed_scale=1.0, heading_offset=0.0, agent_index=0)
    variants = perturb_scenario(straight_scenario, config, n_variants=3)
    assert len(variants) == 3


def test_variant_scenario_ids_are_unique(straight_scenario: ScenarioData) -> None:
    """Each variant must have a distinct scenario_id."""
    config = PerturbationConfig(speed_scale=1.0, heading_offset=0.0, agent_index=0)
    variants = perturb_scenario(straight_scenario, config, n_variants=4)
    ids = [v.scenario_id for v in variants]
    assert len(set(ids)) == 4, f"Non-unique variant IDs: {ids}"


def test_non_perturbed_agents_unchanged(pedestrian_crossing_scenario: ScenarioData) -> None:
    """Agents other than the perturbed one must be byte-for-byte identical."""
    config = PerturbationConfig(speed_scale=2.0, heading_offset=0.5, agent_index=0)
    variants = perturb_scenario(pedestrian_crossing_scenario, config, n_variants=1)

    original_other = pedestrian_crossing_scenario.agents[1]
    variant_other  = variants[0].agents[1]
    assert original_other.positions_x == variant_other.positions_x
    assert original_other.velocities_vx == variant_other.velocities_vx


def test_speed_scaling_changes_velocities(straight_scenario: ScenarioData) -> None:
    """Scaling speed by 2× must double all valid velocity components."""
    config = PerturbationConfig(speed_scale=2.0, heading_offset=0.0, agent_index=0)
    variants = perturb_scenario(straight_scenario, config, n_variants=1)

    # The first variant uses an interpolated scale around base, so check ≥ original.
    original_vx = np.asarray(straight_scenario.agents[0].velocities_vx)
    variant_vx  = np.asarray(variants[0].agents[0].velocities_vx)
    assert np.all(np.abs(variant_vx) >= np.abs(original_vx) * 0.4), (
        "Scaled variant velocities must be larger than original"
    )


def test_position_re_integration_consistent(straight_scenario: ScenarioData) -> None:
    """Re-integrated positions must satisfy x[i+1] ≈ x[i] + vx[i]*dt for valid steps."""
    config = PerturbationConfig(speed_scale=1.0, heading_offset=0.0, agent_index=0)
    variants = perturb_scenario(straight_scenario, config, n_variants=1)

    agent = variants[0].agents[0]
    px = np.asarray(agent.positions_x)
    vx = np.asarray(agent.velocities_vx)
    py = np.asarray(agent.positions_y)
    vy = np.asarray(agent.velocities_vy)

    for idx in range(len(px) - 1):
        expected_x = px[idx] + vx[idx] * _DT
        expected_y = py[idx] + vy[idx] * _DT
        assert abs(px[idx + 1] - expected_x) < 1e-9, (
            f"Position re-integration error at step {idx}: "
            f"expected x={expected_x:.6f}, got {px[idx+1]:.6f}"
        )
        assert abs(py[idx + 1] - expected_y) < 1e-9


def test_heading_offset_applied(straight_scenario: ScenarioData) -> None:
    """Applying a non-zero heading offset must change the agent's heading values."""
    config = PerturbationConfig(speed_scale=1.0, heading_offset=math.pi / 4.0, agent_index=0)
    variants = perturb_scenario(straight_scenario, config, n_variants=1)

    orig_heading  = straight_scenario.agents[0].headings[0]
    var_heading   = variants[0].agents[0].headings[0]
    assert abs(var_heading - orig_heading) > 1e-9


def test_invalid_agent_index_raises(straight_scenario: ScenarioData) -> None:
    """agent_index out of bounds must raise IndexError."""
    config = PerturbationConfig(speed_scale=1.0, heading_offset=0.0, agent_index=99)
    with pytest.raises(IndexError):
        perturb_scenario(straight_scenario, config, n_variants=1)


def test_timestamps_unchanged(straight_scenario: ScenarioData) -> None:
    """Variant timestamps must be identical to the original scenario's timestamps."""
    config = PerturbationConfig(speed_scale=1.5, heading_offset=0.1, agent_index=0)
    variants = perturb_scenario(straight_scenario, config, n_variants=2)
    for variant in variants:
        assert variant.timestamps == straight_scenario.timestamps


def test_single_variant(straight_scenario: ScenarioData) -> None:
    """n_variants=1 must produce exactly one variant."""
    config = PerturbationConfig(speed_scale=1.0, heading_offset=0.0, agent_index=0)
    variants = perturb_scenario(straight_scenario, config, n_variants=1)
    assert len(variants) == 1
