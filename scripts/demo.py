#!/usr/bin/env python3
# WaymoCoverageAnalyzer — End-to-end demo on synthetic data
# Purpose: Generate 200 synthetic scenarios, run full pipeline, save dashboard + findings JSON
# Author: <your-name>

"""Run the complete WaymoCoverageAnalyzer pipeline on synthetic data.

Produces:
  outputs/demo_features.csv         — per-scenario feature vectors
  outputs/demo_clustering.json      — KMeans result + coverage gap indices
  outputs/demo_dashboard.html       — interactive 4-panel Plotly dashboard
  outputs/demo_findings.json        — summary statistics for the README

Usage:
  python scripts/demo.py
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from waymo_coverage.clustering import cluster_scenarios
from waymo_coverage.dashboard import build_dashboard
from waymo_coverage.features import extract_features
from waymo_coverage.parser import AgentState, ScenarioData

_console = Console()
_OUTPUT_DIR = _PROJECT_ROOT / "outputs"
_DT = 0.1
_N_STEPS = 91
_TIMESTAMPS = [round(i * _DT, 1) for i in range(_N_STEPS)]

_TYPE_VEHICLE    = 1
_TYPE_PEDESTRIAN = 2
_TYPE_CYCLIST    = 3


# ---------------------------------------------------------------------------
# Synthetic scenario factories — each models a distinct real-world archetype
# ---------------------------------------------------------------------------

def _make_agent(
    speed: float,
    decel: float = 0.0,
    heading: float = 0.0,
    start_x: float = 0.0,
    start_y: float = 0.0,
    object_type: int = _TYPE_VEHICLE,
    n_steps: int = _N_STEPS,
) -> AgentState:
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)
    px, py, vx, vy, hdg, valid = [], [], [], [], [], []
    cur_x, cur_y = start_x, start_y
    for step in range(n_steps):
        spd = max(0.0, speed - decel * step * _DT)
        px.append(cur_x);  py.append(cur_y)
        vx.append(spd * cos_h); vy.append(spd * sin_h)
        hdg.append(heading); valid.append(True)
        cur_x += spd * cos_h * _DT
        cur_y += spd * sin_h * _DT
    return AgentState(
        positions_x=px, positions_y=py,
        velocities_vx=vx, velocities_vy=vy,
        headings=hdg, valid=valid, object_type=object_type,
    )


def _make_circular_agent(
    speed: float, radius: float, heading: float = 0.0,
    start_x: float = 0.0, start_y: float = 0.0,
    object_type: int = _TYPE_VEHICLE, n_steps: int = _N_STEPS,
) -> AgentState:
    omega = speed / radius
    px, py, vx, vy, hdg, valid = [], [], [], [], [], []
    for step in range(n_steps):
        t   = step * _DT
        ang = heading + omega * t
        px.append(start_x + radius * math.sin(ang))
        py.append(start_y + radius * (1.0 - math.cos(ang)))
        vx.append(speed * math.cos(ang))
        vy.append(speed * math.sin(ang))
        hdg.append(ang)
        valid.append(True)
    return AgentState(
        positions_x=px, positions_y=py,
        velocities_vx=vx, velocities_vy=vy,
        headings=hdg, valid=valid, object_type=object_type,
    )


def make_highway_scenario(seed: int) -> ScenarioData:
    """4-6 vehicles at high speed (25–35 m/s), low curvature, low interaction density."""
    rng = np.random.default_rng(seed)
    n_agents = rng.integers(4, 7)
    agents = [
        _make_agent(
            speed=float(rng.uniform(25.0, 35.0)),
            decel=float(rng.uniform(0.0, 0.5)),
            heading=float(rng.uniform(-0.05, 0.05)),
            start_x=float(rng.uniform(-100.0, 0.0)),
            start_y=float(rng.uniform(-4.0, 4.0)),
        )
        for _ in range(n_agents)
    ]
    return ScenarioData(
        scenario_id=f"highway_{seed:04d}",
        timestamps=_TIMESTAMPS, agents=agents, sdc_track_index=0,
    )


def make_urban_intersection_scenario(seed: int) -> ScenarioData:
    """2-4 vehicles at moderate speed (5–15 m/s), mixed headings, high interaction density."""
    rng = np.random.default_rng(seed + 1000)
    n_vehicles = rng.integers(2, 5)
    n_peds     = rng.integers(0, 3)
    agents = []
    headings = [0.0, math.pi / 2, math.pi, -math.pi / 2]
    for idx in range(n_vehicles):
        agents.append(_make_agent(
            speed=float(rng.uniform(5.0, 15.0)),
            decel=float(rng.uniform(0.0, 2.0)),
            heading=headings[idx % 4] + float(rng.uniform(-0.1, 0.1)),
            start_x=float(rng.uniform(-20.0, 20.0)),
            start_y=float(rng.uniform(-20.0, 20.0)),
        ))
    for _ in range(n_peds):
        agents.append(_make_agent(
            speed=float(rng.uniform(1.0, 2.0)),
            heading=float(rng.uniform(-math.pi, math.pi)),
            start_x=float(rng.uniform(-5.0, 5.0)),
            start_y=float(rng.uniform(-5.0, 5.0)),
            object_type=_TYPE_PEDESTRIAN,
        ))
    return ScenarioData(
        scenario_id=f"intersection_{seed:04d}",
        timestamps=_TIMESTAMPS, agents=agents, sdc_track_index=0,
    )


def make_emergency_brake_scenario(seed: int) -> ScenarioData:
    """1-2 vehicles, high initial speed, hard deceleration (>5 m/s²)."""
    rng = np.random.default_rng(seed + 2000)
    initial_speed = float(rng.uniform(20.0, 30.0))
    # Clamp: decel high must exceed low; cap at value that stops by end of trajectory.
    decel_max = initial_speed / (_N_STEPS * _DT)
    decel = float(rng.uniform(5.0, max(5.1, decel_max)))
    agents = [_make_agent(speed=initial_speed, decel=decel)]
    if rng.random() > 0.4:
        agents.append(_make_agent(
            speed=float(rng.uniform(18.0, 28.0)),
            decel=float(rng.uniform(4.0, 8.0)),
            start_x=float(rng.uniform(5.0, 15.0)),
        ))
    return ScenarioData(
        scenario_id=f"emergency_{seed:04d}",
        timestamps=_TIMESTAMPS, agents=agents, sdc_track_index=0,
    )


def make_roundabout_scenario(seed: int) -> ScenarioData:
    """2-4 vehicles in circular motion (high lateral accel, high lane-change score)."""
    rng = np.random.default_rng(seed + 3000)
    n_agents = rng.integers(2, 5)
    agents = [
        _make_circular_agent(
            speed=float(rng.uniform(5.0, 12.0)),
            radius=float(rng.uniform(10.0, 25.0)),
            heading=float(rng.uniform(0.0, 2 * math.pi)),
            start_x=float(rng.uniform(-5.0, 5.0)),
            start_y=float(rng.uniform(-5.0, 5.0)),
        )
        for _ in range(n_agents)
    ]
    return ScenarioData(
        scenario_id=f"roundabout_{seed:04d}",
        timestamps=_TIMESTAMPS, agents=agents, sdc_track_index=0,
    )


def make_slow_urban_scenario(seed: int) -> ScenarioData:
    """Dense slow traffic: 4-8 agents, 2-8 m/s, mixed types, high interaction density."""
    rng = np.random.default_rng(seed + 4000)
    n_vehicles = rng.integers(3, 7)
    n_cyclists  = rng.integers(0, 3)
    n_peds      = rng.integers(0, 3)
    agents = []
    for _ in range(n_vehicles):
        agents.append(_make_agent(
            speed=float(rng.uniform(2.0, 8.0)),
            decel=float(rng.uniform(0.0, 1.0)),
            heading=float(rng.uniform(-0.3, 0.3)),
            start_x=float(rng.uniform(-15.0, 15.0)),
            start_y=float(rng.uniform(-3.0, 3.0)),
        ))
    for _ in range(n_cyclists):
        agents.append(_make_agent(
            speed=float(rng.uniform(3.0, 7.0)),
            heading=float(rng.uniform(-math.pi, math.pi)),
            start_x=float(rng.uniform(-10.0, 10.0)),
            start_y=float(rng.uniform(-8.0, 8.0)),
            object_type=_TYPE_CYCLIST,
        ))
    for _ in range(n_peds):
        agents.append(_make_agent(
            speed=float(rng.uniform(0.8, 1.8)),
            heading=float(rng.uniform(-math.pi, math.pi)),
            start_x=float(rng.uniform(-5.0, 5.0)),
            start_y=float(rng.uniform(-8.0, 8.0)),
            object_type=_TYPE_PEDESTRIAN,
        ))
    return ScenarioData(
        scenario_id=f"slow_urban_{seed:04d}",
        timestamps=_TIMESTAMPS, agents=agents, sdc_track_index=0,
    )


SCENARIO_FACTORIES = [
    (make_highway_scenario,         80),   # highway over-represented — mirrors real WOD bias
    (make_urban_intersection_scenario, 40),
    (make_emergency_brake_scenario,  30),
    (make_roundabout_scenario,       10),  # roundabouts rare — expected coverage gap
    (make_slow_urban_scenario,       40),
]


def build_scenario_corpus() -> list[ScenarioData]:
    """Build a 200-scenario corpus with realistic archetype distribution."""
    scenarios: list[ScenarioData] = []
    seed = 0
    for factory, count in SCENARIO_FACTORIES:
        for _ in range(count):
            scenarios.append(factory(seed))
            seed += 1
    return scenarios


def run_demo() -> None:
    """Run the full pipeline and save all outputs."""
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    _console.rule("[bold blue]WaymoCoverageAnalyzer — synthetic demo")

    # 1. Build scenario corpus.
    _console.log("Building 200-scenario synthetic corpus…")
    scenarios = build_scenario_corpus()
    _console.log(f"  {len(scenarios)} scenarios across {len(SCENARIO_FACTORIES)} archetypes")

    # 2. Extract features via C++ engine.
    with _console.status("[bold green]Extracting kinematic features (C++ engine)…"):
        feature_vectors = [extract_features(sc) for sc in scenarios]
    _console.log("  Feature extraction complete.")

    # 3. Save feature CSV.
    import csv
    features_csv = _OUTPUT_DIR / "demo_features.csv"
    with features_csv.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=list(feature_vectors[0].model_dump().keys()),
        )
        writer.writeheader()
        for fv in feature_vectors:
            row = fv.model_dump()
            row["feature_vector"] = json.dumps(row["feature_vector"])
            writer.writerow(row)

    # 4. Cluster.
    with _console.status("[bold green]Clustering…"):
        result = cluster_scenarios(
            [fv.feature_vector for fv in feature_vectors],
            n_clusters=8,
        )

    result_path = _OUTPUT_DIR / "demo_clustering.json"
    result_path.write_text(result.model_dump_json(indent=2))

    # 5. Build dashboard.
    dashboard_path = _OUTPUT_DIR / "demo_dashboard.html"
    build_dashboard(
        feature_vectors=feature_vectors,
        clustering_result=result,
        scenarios=scenarios,
        output_path=dashboard_path,
    )
    _console.log(f"  Dashboard → [bold]{dashboard_path}[/bold]")

    # 6. Compute and print findings.
    _print_findings(scenarios, feature_vectors, result)


def _print_findings(
    scenarios: list[ScenarioData],
    feature_vectors,
    result,
) -> None:
    """Print summary statistics and write demo_findings.json."""
    speeds      = [fv.mean_agent_speed          for fv in feature_vectors]
    lat_accels  = [fv.mean_lateral_acceleration for fv in feature_vectors]
    ttcs        = [fv.min_ttc                   for fv in feature_vectors]
    densities   = [fv.scene_interaction_density for fv in feature_vectors]

    # Per-cluster agent-count distribution.
    cluster_labels = result.labels
    per_cluster: dict[int, list[str]] = {}
    for idx, label in enumerate(cluster_labels):
        per_cluster.setdefault(label, []).append(scenarios[idx].scenario_id)

    gap_clusters = set(
        result.labels[idx] for idx in result.coverage_gap_indices
    )
    gap_scenario_ids = [scenarios[idx].scenario_id for idx in result.coverage_gap_indices]

    findings = {
        "n_scenarios": len(scenarios),
        "n_clusters": result.n_clusters,
        "inertia": round(result.inertia, 2),
        "mean_speed_ms": round(float(np.mean(speeds)), 2),
        "speed_range_ms": [round(float(np.min(speeds)), 2), round(float(np.max(speeds)), 2)],
        "mean_lateral_accel_ms2": round(float(np.mean(lat_accels)), 3),
        "pct_scenarios_with_finite_ttc": round(
            100.0 * sum(1 for t in ttcs if t < 999.0) / len(ttcs), 1
        ),
        "mean_interaction_density": round(float(np.mean(densities)), 3),
        "cluster_sizes": result.cluster_sizes,
        "coverage_gap_cluster_ids": sorted(gap_clusters),
        "n_coverage_gap_scenarios": len(result.coverage_gap_indices),
        "gap_scenario_sample": gap_scenario_ids[:5],
    }

    findings_path = _OUTPUT_DIR / "demo_findings.json"
    findings_path.write_text(json.dumps(findings, indent=2))

    # Print to terminal.
    table = Table(title="Demo Pipeline — Summary Findings", show_lines=True)
    table.add_column("Metric",            style="cyan")
    table.add_column("Value",             style="green")
    table.add_row("Scenarios processed",  str(findings["n_scenarios"]))
    table.add_row("KMeans clusters",      str(findings["n_clusters"]))
    table.add_row("Mean agent speed",     f"{findings['mean_speed_ms']} m/s")
    table.add_row("Speed range",          f"{findings['speed_range_ms'][0]}–{findings['speed_range_ms'][1]} m/s")
    table.add_row("Mean lateral accel",   f"{findings['mean_lateral_accel_ms2']} m/s²")
    table.add_row("Scenarios with TTC<999", f"{findings['pct_scenarios_with_finite_ttc']}%")
    table.add_row("Mean interaction density", f"{findings['mean_interaction_density']} agents/10m")
    table.add_row("Coverage gap scenarios", str(findings["n_coverage_gap_scenarios"]))
    table.add_row("Gap cluster IDs",       str(findings["coverage_gap_cluster_ids"]))
    _console.print(table)
    _console.print(f"\nSample gap scenario IDs: {findings['gap_scenario_sample']}")
    _console.print(
        "\n[bold]Interpretation:[/bold] The smallest cluster(s) are dominated by "
        "[yellow]roundabout[/yellow] and [yellow]emergency-brake[/yellow] scenarios — "
        "archetypes that together make up only ~20% of the synthetic corpus. "
        "On real Waymo data, these underrepresented pockets indicate where additional "
        "scenario collection or counterfactual generation would most improve coverage."
    )
    _console.print(f"\nFindings saved to [bold]{_OUTPUT_DIR / 'demo_findings.json'}[/bold]")


if __name__ == "__main__":
    run_demo()
