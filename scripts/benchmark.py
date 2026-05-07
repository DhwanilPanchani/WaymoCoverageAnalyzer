#!/usr/bin/env python3
# WaymoCoverageAnalyzer — Benchmark script
# Purpose: Compare C++ engine vs NumPy baseline for 10, 50, 100 synthetic scenarios
# Author: <your-name>

"""Benchmark the C++ kinematic engine against the pure-NumPy baseline implementation."""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

# Allow running from the project root without installation.
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from waymo_coverage.features import extract_features, extract_features_numpy_baseline
from waymo_coverage.parser import AgentState, ScenarioData

_console = Console()

_DT = 0.1
_N_STEPS = 91
_TIMESTAMPS = [round(idx * _DT, 1) for idx in range(_N_STEPS)]
_SCENARIO_SIZES = [10, 50, 100]


def _make_synthetic_scenario(scenario_index: int, n_agents: int = 5) -> ScenarioData:
    """Build one synthetic scenario with *n_agents* vehicles moving in varied directions.

    Args:
        scenario_index: Unique integer used to vary kinematics across scenarios.
        n_agents: Number of agents per scenario.

    Returns:
        Fully populated ScenarioData.
    """
    agents: list[AgentState] = []
    rng = np.random.default_rng(seed=scenario_index)

    for agent_idx in range(n_agents):
        speed   = float(rng.uniform(2.0, 20.0))
        heading = float(rng.uniform(-math.pi, math.pi))
        cos_h   = math.cos(heading)
        sin_h   = math.sin(heading)
        start_x = float(rng.uniform(-50.0, 50.0))
        start_y = float(rng.uniform(-50.0, 50.0))
        decel   = float(rng.uniform(0.0, speed / ((_N_STEPS - 1) * _DT)))

        positions_x: list[float] = []
        positions_y: list[float] = []
        velocities_vx: list[float] = []
        velocities_vy: list[float] = []
        headings: list[float] = []
        valid: list[bool] = []

        current_x = start_x
        current_y = start_y

        for step in range(_N_STEPS):
            current_speed = max(0.0, speed - decel * step * _DT)
            positions_x.append(current_x)
            positions_y.append(current_y)
            velocities_vx.append(current_speed * cos_h)
            velocities_vy.append(current_speed * sin_h)
            headings.append(heading)
            valid.append(True)
            current_x += current_speed * cos_h * _DT
            current_y += current_speed * sin_h * _DT

        agents.append(AgentState(
            positions_x=positions_x,
            positions_y=positions_y,
            velocities_vx=velocities_vx,
            velocities_vy=velocities_vy,
            headings=headings,
            valid=valid,
            object_type=1,
        ))

    return ScenarioData(
        scenario_id=f"bench_{scenario_index:04d}",
        timestamps=_TIMESTAMPS,
        agents=agents,
        sdc_track_index=0,
    )


def _time_extraction(
    scenarios: list[ScenarioData],
    use_cpp: bool,
) -> float:
    """Run feature extraction on all scenarios and return elapsed wall-clock time.

    Args:
        scenarios: List of synthetic scenarios.
        use_cpp: If True, use the C++ engine; otherwise use NumPy baseline.

    Returns:
        Elapsed time in seconds.
    """
    extractor = extract_features if use_cpp else extract_features_numpy_baseline
    start = time.perf_counter()
    for scenario in scenarios:
        extractor(scenario)
    return time.perf_counter() - start


def run_benchmark(output_path: Path) -> None:
    """Run the benchmark and print results as a rich table, saving JSON to output_path.

    Args:
        output_path: Path to write the JSON benchmark results.
    """
    _console.rule("[bold blue]WaymoCoverageAnalyzer — Benchmark")

    results: dict[str, dict[str, float]] = {}
    rows: list[tuple[str, str, str, str]] = []

    for n_scenarios in _SCENARIO_SIZES:
        _console.log(f"Generating {n_scenarios} synthetic scenarios…")
        scenarios = [_make_synthetic_scenario(idx) for idx in range(n_scenarios)]

        # Warm-up pass (not timed) to avoid cold-start bias.
        extract_features(scenarios[0])
        extract_features_numpy_baseline(scenarios[0])

        cpp_time   = _time_extraction(scenarios, use_cpp=True)
        numpy_time = _time_extraction(scenarios, use_cpp=False)
        speedup    = numpy_time / cpp_time if cpp_time > 0 else float("inf")

        results[str(n_scenarios)] = {
            "n_scenarios":  n_scenarios,
            "cpp_seconds":  round(cpp_time, 4),
            "numpy_seconds": round(numpy_time, 4),
            "speedup":      round(speedup, 2),
        }
        rows.append((
            str(n_scenarios),
            f"{cpp_time:.4f}",
            f"{numpy_time:.4f}",
            f"{speedup:.2f}×",
        ))

    # Print rich table.
    table = Table(title="Kinematic Feature Extraction Benchmark", show_lines=True)
    table.add_column("Scenarios",    style="cyan",   justify="right")
    table.add_column("C++ Engine (s)",  style="green",  justify="right")
    table.add_column("NumPy Baseline (s)", style="yellow", justify="right")
    table.add_column("Speedup",      style="bold magenta", justify="right")
    for row in rows:
        table.add_row(*row)
    _console.print(table)

    # Save JSON.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    _console.print(f"\nResults saved to [bold]{output_path}[/bold]")


if __name__ == "__main__":
    run_benchmark(Path("outputs/benchmark.json"))
