# WaymoCoverageAnalyzer — Kinematic feature extraction
# Purpose: Call the C++ engine per agent and aggregate to scenario-level feature vectors
# Author: <your-name>

"""Extract kinematic features from ScenarioData objects using the C++ engine.

Also provides a pure-NumPy baseline implementation for benchmarking.
"""

import numpy as np
from pydantic import BaseModel, ConfigDict

from waymo_coverage import waymo_kinematics  # compiled C++ pybind11 module
from waymo_coverage.parser import AgentState, ScenarioData

# Waymo Motion dataset object type constants (Track.ObjectType enum).
_OBJECT_TYPE_VEHICLE    = 1
_OBJECT_TYPE_PEDESTRIAN = 2
_OBJECT_TYPE_CYCLIST    = 3

# Interaction density radius in metres.
_INTERACTION_RADIUS_M = 10.0

# Waymo Motion dataset timestep.
_DT = 0.1


class ScenarioFeatureVector(BaseModel):
    """All scenario-level features used as input to clustering."""

    model_config = ConfigDict(frozen=True)

    scenario_id: str
    num_agents: int
    num_vehicles: int
    num_pedestrians: int
    num_cyclists: int
    mean_agent_speed: float
    max_agent_speed: float
    mean_agent_acceleration: float
    max_agent_jerk: float
    mean_curvature: float
    mean_lateral_acceleration: float   # centripetal: speed² × curvature; 0 on straight roads
    max_lateral_acceleration: float    # peak cornering g-force across all agents
    min_ttc: float
    scene_interaction_density: float
    heading_variance: float
    path_length_variance: float
    feature_vector: list[float]  # flattened numeric features for clustering


def _agent_arrays(agent: AgentState) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Convert AgentState lists to contiguous numpy arrays.

    Args:
        agent: Parsed agent state.

    Returns:
        Tuple of (px, py, vx, vy, headings, valid) as numpy arrays.
    """
    return (
        np.asarray(agent.positions_x,  dtype=np.float64),
        np.asarray(agent.positions_y,  dtype=np.float64),
        np.asarray(agent.velocities_vx, dtype=np.float64),
        np.asarray(agent.velocities_vy, dtype=np.float64),
        np.asarray(agent.headings,      dtype=np.float64),
        np.asarray(agent.valid,         dtype=bool),
    )


def _compute_interaction_density(scenario: ScenarioData) -> float:
    """Compute average number of agents within *_INTERACTION_RADIUS_M* over time.

    Args:
        scenario: Parsed scenario data.

    Returns:
        Mean interaction density across all timesteps and agents.
    """
    num_timesteps = len(scenario.timestamps)
    if len(scenario.agents) < 2 or num_timesteps == 0:
        return 0.0

    density_sum = 0.0
    count = 0

    for timestep in range(num_timesteps):
        valid_positions: list[tuple[float, float]] = []
        for agent in scenario.agents:
            if timestep < len(agent.valid) and agent.valid[timestep]:
                valid_positions.append((
                    agent.positions_x[timestep],
                    agent.positions_y[timestep],
                ))

        if len(valid_positions) < 2:
            continue

        positions_array = np.array(valid_positions)
        for agent_pos in positions_array:
            diffs = positions_array - agent_pos
            distances = np.sqrt((diffs ** 2).sum(axis=1))
            # Exclude self (distance == 0).
            neighbors = int(np.sum(distances < _INTERACTION_RADIUS_M)) - 1
            density_sum += neighbors
            count += 1

    return density_sum / count if count > 0 else 0.0


def extract_features(scenario: ScenarioData) -> ScenarioFeatureVector:
    """Call the C++ engine per agent and aggregate to scenario-level features.

    Args:
        scenario: Parsed scenario data from parser.load_scenarios().

    Returns:
        ScenarioFeatureVector with all numeric features for clustering.
    """
    num_vehicles    = sum(1 for a in scenario.agents if a.object_type == _OBJECT_TYPE_VEHICLE)
    num_pedestrians = sum(1 for a in scenario.agents if a.object_type == _OBJECT_TYPE_PEDESTRIAN)
    num_cyclists    = sum(1 for a in scenario.agents if a.object_type == _OBJECT_TYPE_CYCLIST)

    all_features: list[waymo_kinematics.KinematicFeatures] = []
    for agent in scenario.agents:
        px, py, vx, vy, hdg, valid = _agent_arrays(agent)
        feat = waymo_kinematics.KinematicsEngine.compute(px, py, vx, vy, hdg, valid, _DT)
        all_features.append(feat)

    # Scenario-level aggregations across all agents.
    mean_speeds    = [f.mean_speed                  for f in all_features]
    max_speeds     = [f.max_speed                   for f in all_features]
    mean_accels    = [f.mean_acceleration            for f in all_features]  # signed
    max_jerks      = [f.max_jerk_magnitude          for f in all_features]  # unsigned magnitude
    mean_curvs     = [f.mean_curvature              for f in all_features]
    mean_lat_accels = [f.mean_lateral_acceleration  for f in all_features]
    max_lat_accels  = [f.max_lateral_acceleration   for f in all_features]
    path_lengths   = [f.path_length                 for f in all_features]
    # Recover circular variance from lane_change_score = circ_var * mean_speed.
    heading_vars   = [f.lane_change_score / (f.mean_speed + 1e-9) for f in all_features]

    # TTC across all unique agent pairs.
    min_ttc = 999.0
    agents = scenario.agents
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            ego_px, ego_py, ego_vx, ego_vy, ego_h, ego_v = _agent_arrays(agents[i])
            oth_px, oth_py, oth_vx, oth_vy, oth_h, oth_v = _agent_arrays(agents[j])
            ttc = waymo_kinematics.KinematicsEngine.compute_ttc(
                ego_px, ego_py, ego_vx, ego_vy, ego_h, ego_v,
                oth_px, oth_py, oth_vx, oth_vy, oth_h, oth_v,
                _DT,
            )
            min_ttc = min(min_ttc, ttc)

    interaction_density = _compute_interaction_density(scenario)

    mean_speed_val    = float(np.mean(mean_speeds))     if mean_speeds     else 0.0
    max_speed_val     = float(np.max(max_speeds))       if max_speeds      else 0.0
    mean_accel_val    = float(np.mean(mean_accels))     if mean_accels     else 0.0
    max_jerk_val      = float(np.max(max_jerks))        if max_jerks       else 0.0
    mean_curv_val     = float(np.mean(mean_curvs))      if mean_curvs      else 0.0
    mean_lat_acc_val  = float(np.mean(mean_lat_accels)) if mean_lat_accels else 0.0
    max_lat_acc_val   = float(np.max(max_lat_accels))   if max_lat_accels  else 0.0
    path_len_var      = float(np.var(path_lengths))     if path_lengths    else 0.0
    heading_var_val   = float(np.mean(heading_vars))    if heading_vars    else 0.0

    feature_vector = [
        float(len(scenario.agents)),
        float(num_vehicles),
        float(num_pedestrians),
        float(num_cyclists),
        mean_speed_val,
        max_speed_val,
        mean_accel_val,
        max_jerk_val,
        mean_curv_val,
        mean_lat_acc_val,
        max_lat_acc_val,
        min(min_ttc, 999.0),
        interaction_density,
        heading_var_val,
        path_len_var,
    ]

    return ScenarioFeatureVector(
        scenario_id=scenario.scenario_id,
        num_agents=len(scenario.agents),
        num_vehicles=num_vehicles,
        num_pedestrians=num_pedestrians,
        num_cyclists=num_cyclists,
        mean_agent_speed=mean_speed_val,
        max_agent_speed=max_speed_val,
        mean_agent_acceleration=mean_accel_val,
        max_agent_jerk=max_jerk_val,
        mean_curvature=mean_curv_val,
        mean_lateral_acceleration=mean_lat_acc_val,
        max_lateral_acceleration=max_lat_acc_val,
        min_ttc=min(min_ttc, 999.0),
        scene_interaction_density=interaction_density,
        heading_variance=heading_var_val,
        path_length_variance=path_len_var,
        feature_vector=feature_vector,
    )


def extract_features_numpy_baseline(scenario: ScenarioData) -> ScenarioFeatureVector:
    """Pure NumPy re-implementation of the same kinematic features as the C++ engine.

    Used only for benchmarking — NOT used in the main pipeline.

    Args:
        scenario: Parsed scenario data.

    Returns:
        ScenarioFeatureVector computed entirely in NumPy.
    """
    num_vehicles    = sum(1 for a in scenario.agents if a.object_type == _OBJECT_TYPE_VEHICLE)
    num_pedestrians = sum(1 for a in scenario.agents if a.object_type == _OBJECT_TYPE_PEDESTRIAN)
    num_cyclists    = sum(1 for a in scenario.agents if a.object_type == _OBJECT_TYPE_CYCLIST)

    all_mean_speeds:    list[float] = []
    all_max_speeds:     list[float] = []
    all_mean_accels:    list[float] = []
    all_max_jerks:      list[float] = []
    all_mean_curvs:     list[float] = []
    all_mean_lat_accels: list[float] = []
    all_max_lat_accels:  list[float] = []
    all_path_lens:      list[float] = []
    all_heading_vars:   list[float] = []

    for agent in scenario.agents:
        px, py, vx, vy, hdg, valid = _agent_arrays(agent)
        valid_mask = valid.astype(bool)

        if valid_mask.sum() < 2:
            all_mean_speeds.append(0.0)
            all_max_speeds.append(0.0)
            all_mean_accels.append(0.0)
            all_max_jerks.append(0.0)
            all_mean_curvs.append(0.0)
            all_mean_lat_accels.append(0.0)
            all_max_lat_accels.append(0.0)
            all_path_lens.append(0.0)
            all_heading_vars.append(0.0)
            continue

        valid_vx  = vx[valid_mask]
        valid_vy  = vy[valid_mask]
        valid_px  = px[valid_mask]
        valid_py  = py[valid_mask]
        valid_hdg = hdg[valid_mask]

        speeds = np.sqrt(valid_vx ** 2 + valid_vy ** 2)
        # Longitudinal acceleration (signed, rate of speed-magnitude change).
        accels = np.diff(speeds) / _DT
        jerks  = np.diff(accels) / _DT if len(accels) > 1 else np.array([0.0])

        # Curvature and lateral acceleration from the 2D cross product.
        # cross_z = vx*ay - vy*ax  (z-component of v × a)
        # curvature[i]   = |cross_z| / speed^3
        # lat_accel[i]   = speed^2 * curvature = |cross_z| / speed
        accel_vx   = np.diff(valid_vx) / _DT
        accel_vy   = np.diff(valid_vy) / _DT
        cross_z    = valid_vx[:-1] * accel_vy - valid_vy[:-1] * accel_vx
        abs_cross  = np.abs(cross_z)
        speed_mid  = np.maximum(speeds[:-1], 1e-9)
        curvatures = abs_cross / np.maximum(speed_mid ** 3, 1e-9)
        lat_accels = abs_cross / speed_mid  # = speed^2 * curvature

        dx = np.diff(valid_px)
        dy = np.diff(valid_py)
        path_length = float(np.sum(np.sqrt(dx ** 2 + dy ** 2)))

        # Circular variance: 1 - |mean(exp(i*theta))| — wrap-safe, range [0, 1].
        circ_var = float(1.0 - np.abs(np.mean(np.exp(1j * valid_hdg))))

        all_mean_speeds.append(float(np.mean(speeds)))
        all_max_speeds.append(float(np.max(speeds)))
        all_mean_accels.append(float(np.mean(accels)))                                   # signed
        all_max_jerks.append(float(np.max(np.abs(jerks))) if len(jerks) > 0 else 0.0)   # magnitude
        all_mean_curvs.append(float(np.mean(curvatures)) if len(curvatures) > 0 else 0.0)
        all_mean_lat_accels.append(float(np.mean(lat_accels)) if len(lat_accels) > 0 else 0.0)
        all_max_lat_accels.append(float(np.max(lat_accels))   if len(lat_accels) > 0 else 0.0)
        all_path_lens.append(path_length)
        all_heading_vars.append(circ_var)

    # TTC — pure NumPy.
    min_ttc = 999.0
    agents = scenario.agents
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            ego_px, ego_py, ego_vx, ego_vy, _, ego_v = _agent_arrays(agents[i])
            oth_px, oth_py, oth_vx, oth_vy, _, oth_v = _agent_arrays(agents[j])
            both_valid = ego_v & oth_v
            if not np.any(both_valid):
                continue
            rel_pos_x = oth_px[both_valid] - ego_px[both_valid]
            rel_pos_y = oth_py[both_valid] - ego_py[both_valid]
            rel_vel_x = oth_vx[both_valid] - ego_vx[both_valid]
            rel_vel_y = oth_vy[both_valid] - ego_vy[both_valid]
            denom = rel_vel_x ** 2 + rel_vel_y ** 2
            mask  = denom > 1e-9
            if not np.any(mask):
                continue
            ttc_vals = -(rel_pos_x[mask] * rel_vel_x[mask] + rel_pos_y[mask] * rel_vel_y[mask]) / denom[mask]
            positive_ttc = ttc_vals[ttc_vals > 0.0]
            if len(positive_ttc) > 0:
                min_ttc = min(min_ttc, float(np.min(positive_ttc)))

    interaction_density = _compute_interaction_density(scenario)

    mean_speed_val    = float(np.mean(all_mean_speeds))     if all_mean_speeds     else 0.0
    max_speed_val     = float(np.max(all_max_speeds))       if all_max_speeds      else 0.0
    mean_accel_val    = float(np.mean(all_mean_accels))     if all_mean_accels     else 0.0
    max_jerk_val      = float(np.max(all_max_jerks))        if all_max_jerks       else 0.0
    mean_curv_val     = float(np.mean(all_mean_curvs))      if all_mean_curvs      else 0.0
    mean_lat_acc_val  = float(np.mean(all_mean_lat_accels)) if all_mean_lat_accels else 0.0
    max_lat_acc_val   = float(np.max(all_max_lat_accels))   if all_max_lat_accels  else 0.0
    path_len_var      = float(np.var(all_path_lens))        if all_path_lens       else 0.0
    heading_var_val   = float(np.mean(all_heading_vars))    if all_heading_vars    else 0.0

    feature_vector = [
        float(len(scenario.agents)),
        float(num_vehicles),
        float(num_pedestrians),
        float(num_cyclists),
        mean_speed_val,
        max_speed_val,
        mean_accel_val,
        max_jerk_val,
        mean_curv_val,
        mean_lat_acc_val,
        max_lat_acc_val,
        min(min_ttc, 999.0),
        interaction_density,
        heading_var_val,
        path_len_var,
    ]

    return ScenarioFeatureVector(
        scenario_id=scenario.scenario_id,
        num_agents=len(scenario.agents),
        num_vehicles=num_vehicles,
        num_pedestrians=num_pedestrians,
        num_cyclists=num_cyclists,
        mean_agent_speed=mean_speed_val,
        max_agent_speed=max_speed_val,
        mean_agent_acceleration=mean_accel_val,
        max_agent_jerk=max_jerk_val,
        mean_curvature=mean_curv_val,
        mean_lateral_acceleration=mean_lat_acc_val,
        max_lateral_acceleration=max_lat_acc_val,
        min_ttc=min(min_ttc, 999.0),
        scene_interaction_density=interaction_density,
        heading_variance=heading_var_val,
        path_length_variance=path_len_var,
        feature_vector=feature_vector,
    )
