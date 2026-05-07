# WaymoCoverageAnalyzer — Counterfactual trajectory perturbation
# Purpose: Generate kinematic variants of a scenario via velocity scaling and heading rotation
# Author: <your-name>

"""Generate counterfactual scenario variants by perturbing one agent's kinematics."""

import math
import uuid

import numpy as np
from pydantic import BaseModel, ConfigDict

from waymo_coverage.parser import AgentState, ScenarioData

# Waymo Motion dataset timestep in seconds.
_DT = 0.1


class PerturbationConfig(BaseModel):
    """Configuration for one perturbation variant."""

    model_config = ConfigDict(frozen=True)

    speed_scale: float = 1.0      # multiply all velocities by this factor
    heading_offset: float = 0.0   # add this offset (radians) to all headings
    agent_index: int = 0          # which agent to perturb


def _perturb_agent(agent: AgentState, config: PerturbationConfig) -> AgentState:
    """Apply speed scaling and heading rotation to one agent, then re-integrate positions.

    The new trajectory is computed by:
    1. Scaling velocities: vx *= speed_scale, vy *= speed_scale
    2. Rotating the velocity vector by heading_offset (rigid 2D rotation)
    3. Re-integrating positions from the first valid position using Euler forward steps
    4. Adding heading_offset to all heading values

    Args:
        agent: Original agent state.
        config: Perturbation parameters.

    Returns:
        New AgentState with perturbed kinematics.
    """
    vx_orig = np.asarray(agent.velocities_vx, dtype=np.float64)
    vy_orig = np.asarray(agent.velocities_vy, dtype=np.float64)
    headings_orig = np.asarray(agent.headings, dtype=np.float64)
    px_orig = np.asarray(agent.positions_x, dtype=np.float64)
    py_orig = np.asarray(agent.positions_y, dtype=np.float64)
    valid = np.asarray(agent.valid, dtype=bool)

    # Scale velocities.
    vx_new = vx_orig * config.speed_scale
    vy_new = vy_orig * config.speed_scale

    # Rotate velocity vectors by heading_offset.
    cos_offset = math.cos(config.heading_offset)
    sin_offset = math.sin(config.heading_offset)
    vx_rot = vx_new * cos_offset - vy_new * sin_offset
    vy_rot = vx_new * sin_offset + vy_new * cos_offset

    # Rotate headings.
    new_headings = headings_orig + config.heading_offset

    # Re-integrate positions using Euler forward integration from the first valid position.
    n_steps = len(px_orig)
    new_px = px_orig.copy()
    new_py = py_orig.copy()

    # Find the first valid timestep to anchor the integration.
    first_valid = -1
    for idx in range(n_steps):
        if valid[idx]:
            first_valid = idx
            break

    if first_valid >= 0:
        # Positions at invalid timesteps before first_valid remain as-is.
        for idx in range(first_valid, n_steps - 1):
            if valid[idx]:
                new_px[idx + 1] = new_px[idx] + vx_rot[idx] * _DT
                new_py[idx + 1] = new_py[idx] + vy_rot[idx] * _DT

    return AgentState(
        positions_x=new_px.tolist(),
        positions_y=new_py.tolist(),
        velocities_vx=vx_rot.tolist(),
        velocities_vy=vy_rot.tolist(),
        headings=new_headings.tolist(),
        valid=agent.valid,
        object_type=agent.object_type,
    )


def _make_variant_configs(
    base_config: PerturbationConfig,
    n_variants: int,
) -> list[PerturbationConfig]:
    """Generate a sequence of perturbation configs that interpolate around base_config.

    Produces *n_variants* configs by linearly spacing speed_scale around
    base_config.speed_scale and alternating heading_offset signs.

    Args:
        base_config: The anchor perturbation configuration.
        n_variants: How many variants to generate.

    Returns:
        List of PerturbationConfig instances.
    """
    configs: list[PerturbationConfig] = []
    for variant_idx in range(n_variants):
        # Interpolate speed_scale between 0.5× and 1.5× of the base scale.
        alpha = variant_idx / max(1, n_variants - 1)
        scale = base_config.speed_scale * (0.5 + alpha)

        # Alternate heading offsets: 0, +offset, -offset, +2×offset, …
        if variant_idx % 2 == 0:
            heading = base_config.heading_offset * (variant_idx // 2 + 1)
        else:
            heading = -base_config.heading_offset * (variant_idx // 2 + 1)

        configs.append(PerturbationConfig(
            speed_scale=scale,
            heading_offset=heading,
            agent_index=base_config.agent_index,
        ))
    return configs


def perturb_scenario(
    scenario: ScenarioData,
    config: PerturbationConfig,
    n_variants: int = 3,
) -> list[ScenarioData]:
    """Generate kinematic variants of a scenario using constant-velocity integration.

    For each variant:
    1. Scale velocities by speed_scale.
    2. Rotate heading by heading_offset.
    3. Re-integrate positions: x[i+1] = x[i] + vx[i]*dt, y[i+1] = y[i] + vy[i]*dt.
    4. Keep all other agents unchanged.

    Args:
        scenario: Original parsed scenario.
        config: Base perturbation configuration.  Each variant is derived by
            interpolating around this config (speed and heading are varied).
        n_variants: Number of variant ScenarioData objects to produce (default 3).

    Returns:
        List of *n_variants* ScenarioData objects with new scenario_ids.

    Raises:
        IndexError: If config.agent_index is out of bounds for scenario.agents.
    """
    if config.agent_index >= len(scenario.agents):
        raise IndexError(
            f"agent_index {config.agent_index} out of range "
            f"for scenario with {len(scenario.agents)} agents"
        )

    variant_configs = _make_variant_configs(config, n_variants)
    variants: list[ScenarioData] = []

    for variant_config in variant_configs:
        new_agents: list[AgentState] = []
        for agent_idx, agent in enumerate(scenario.agents):
            if agent_idx == variant_config.agent_index:
                new_agents.append(_perturb_agent(agent, variant_config))
            else:
                new_agents.append(agent)

        variant_id = f"{scenario.scenario_id}_perturb_{uuid.uuid4().hex[:8]}"
        variants.append(ScenarioData(
            scenario_id=variant_id,
            timestamps=scenario.timestamps,
            agents=new_agents,
            sdc_track_index=scenario.sdc_track_index,
        ))

    return variants
