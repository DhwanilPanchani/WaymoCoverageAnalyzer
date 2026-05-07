# WaymoCoverageAnalyzer — Typer CLI entry point
# Purpose: Expose analyze, cluster, perturb, and serve commands for the pipeline
# Author: <your-name>

"""Command-line interface for the WaymoCoverageAnalyzer pipeline."""

import csv
import json
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from waymo_coverage.clustering import cluster_scenarios
from waymo_coverage.dashboard import build_dashboard
from waymo_coverage.features import ScenarioFeatureVector, extract_features
from waymo_coverage.parser import ScenarioData, load_scenarios
from waymo_coverage.perturbation import PerturbationConfig, perturb_scenario

app = typer.Typer(
    name="waymo-coverage",
    help="Analyze Waymo Open Dataset scenarios for kinematic coverage gaps.",
    no_args_is_help=True,
)
_console = Console()


def _save_features_csv(
    feature_vectors: list[ScenarioFeatureVector],
    output_path: Path,
) -> None:
    """Write feature vectors to a CSV file.

    Args:
        feature_vectors: List of extracted feature vectors.
        output_path: Destination CSV path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    field_names = list(ScenarioFeatureVector.model_fields.keys())

    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=field_names)
        writer.writeheader()
        for fv in feature_vectors:
            row = fv.model_dump()
            row["feature_vector"] = json.dumps(row["feature_vector"])
            writer.writerow(row)


def _load_features_csv(csv_path: Path) -> list[ScenarioFeatureVector]:
    """Load feature vectors from a CSV written by _save_features_csv.

    Args:
        csv_path: Path to the features CSV.

    Returns:
        List of ScenarioFeatureVector.
    """
    feature_vectors: list[ScenarioFeatureVector] = []
    with csv_path.open(newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            row["feature_vector"] = json.loads(row["feature_vector"])
            # Cast numeric fields.
            for numeric_field in [
                "num_agents", "num_vehicles", "num_pedestrians", "num_cyclists"
            ]:
                row[numeric_field] = int(row[numeric_field])
            for float_field in [
                "mean_agent_speed", "max_agent_speed", "mean_agent_acceleration",
                "max_agent_jerk", "mean_curvature", "min_ttc",
                "scene_interaction_density", "heading_variance", "path_length_variance",
            ]:
                row[float_field] = float(row[float_field])
            feature_vectors.append(ScenarioFeatureVector(**row))
    return feature_vectors


@app.command()
def analyze(
    data_path: Path = typer.Argument(..., help="Path to a Waymo .tfrecord file."),
    max_scenarios: int = typer.Option(100, help="Maximum number of scenarios to process."),
    output_dir: Path = typer.Option(Path("outputs"), help="Directory for output files."),
) -> None:
    """Parse scenarios from a .tfrecord, extract C++ kinematic features, and save a CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    features_csv = output_dir / "features.csv"

    _console.rule("[bold blue]WaymoCoverageAnalyzer — analyze")
    _console.print(f"Input  : {data_path}")
    _console.print(f"Output : {features_csv}")

    start_time = time.perf_counter()
    scenarios: list[ScenarioData] = load_scenarios(data_path, max_scenarios=max_scenarios)

    if not scenarios:
        _console.print("[red]No scenarios loaded. Check the input file path.[/red]")
        raise typer.Exit(code=1)

    feature_vectors: list[ScenarioFeatureVector] = []
    with _console.status("[bold green]Extracting kinematic features via C++ engine…"):
        for scenario in scenarios:
            feature_vectors.append(extract_features(scenario))

    elapsed = time.perf_counter() - start_time

    _save_features_csv(feature_vectors, features_csv)

    summary_table = Table(title="Feature Extraction Summary", show_lines=True)
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")
    summary_table.add_row("Scenarios processed", str(len(feature_vectors)))
    summary_table.add_row("Wall-clock time (s)", f"{elapsed:.3f}")
    summary_table.add_row("Features CSV", str(features_csv))
    _console.print(summary_table)


@app.command()
def cluster(
    features_csv: Path = typer.Argument(..., help="Path to features CSV from 'analyze'."),
    n_clusters: int = typer.Option(8, help="Number of KMeans clusters."),
    output_dir: Path = typer.Option(Path("outputs"), help="Directory for output files."),
) -> None:
    """Load a features CSV, run KMeans clustering, and save results + Plotly dashboard."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results_json  = output_dir / "clustering_results.json"
    dashboard_html = output_dir / "dashboard.html"

    _console.rule("[bold blue]WaymoCoverageAnalyzer — cluster")

    feature_vectors = _load_features_csv(features_csv)
    if not feature_vectors:
        _console.print("[red]No feature vectors loaded from CSV.[/red]")
        raise typer.Exit(code=1)

    with _console.status("[bold green]Running KMeans clustering…"):
        clustering_result = cluster_scenarios(
            [fv.feature_vector for fv in feature_vectors],
            n_clusters=n_clusters,
        )

    # Persist clustering result.
    results_json.write_text(clustering_result.model_dump_json(indent=2))

    _console.print(f"[green]Clustering complete.[/green] Inertia: {clustering_result.inertia:.2f}")
    _console.print(f"Coverage gap scenarios: {len(clustering_result.coverage_gap_indices)}")

    _console.print("Dashboard requires scenario objects for the perturbation panel.")
    _console.print(
        "[yellow]Hint:[/yellow] run 'perturb' command to generate variants, "
        "or the dashboard will use placeholder trajectories."
    )

    # Build a minimal dashboard without real scenario objects (trajectory panel will be empty).
    _build_dashboard_from_features(
        feature_vectors=feature_vectors,
        clustering_result=clustering_result,
        dashboard_html=dashboard_html,
    )
    _console.print(f"Dashboard saved to [bold]{dashboard_html}[/bold]")


def _build_dashboard_from_features(
    feature_vectors: list[ScenarioFeatureVector],
    clustering_result,
    dashboard_html: Path,
) -> None:
    """Build a dashboard using synthetic single-point scenarios for the perturbation panel.

    This lets 'cluster' work standalone without access to the original .tfrecord.

    Args:
        feature_vectors: Extracted feature vectors.
        clustering_result: Output of cluster_scenarios().
        dashboard_html: Destination HTML path.
    """
    from waymo_coverage.parser import AgentState

    synthetic_agents = [
        AgentState(
            positions_x=[0.0, 1.0],
            positions_y=[0.0, 0.0],
            velocities_vx=[10.0, 10.0],
            velocities_vy=[0.0, 0.0],
            headings=[0.0, 0.0],
            valid=[True, True],
            object_type=1,
        )
    ]
    synthetic_scenario = ScenarioData(
        scenario_id="placeholder",
        timestamps=[0.0, 0.1],
        agents=synthetic_agents,
        sdc_track_index=0,
    )
    build_dashboard(
        feature_vectors=feature_vectors,
        clustering_result=clustering_result,
        scenarios=[synthetic_scenario],
        output_path=dashboard_html,
    )


@app.command()
def perturb(
    data_path: Path = typer.Argument(..., help="Path to a Waymo .tfrecord file."),
    scenario_id: str = typer.Option(..., help="scenario_id to perturb."),
    n_variants: int = typer.Option(3, help="Number of counterfactual variants."),
    output_dir: Path = typer.Option(Path("outputs"), help="Directory for output files."),
) -> None:
    """Load one scenario by ID, generate counterfactual variants, and save a dashboard."""
    output_dir.mkdir(parents=True, exist_ok=True)

    _console.rule("[bold blue]WaymoCoverageAnalyzer — perturb")

    scenarios = load_scenarios(data_path, max_scenarios=500)
    target: ScenarioData | None = next(
        (sc for sc in scenarios if sc.scenario_id == scenario_id), None
    )

    if target is None:
        _console.print(f"[red]Scenario '{scenario_id}' not found in {data_path}[/red]")
        raise typer.Exit(code=1)

    config = PerturbationConfig(speed_scale=1.0, heading_offset=0.15, agent_index=0)
    variants = perturb_scenario(target, config, n_variants=n_variants)

    # Extract features for all variants to produce a mini dashboard.
    all_scenarios = [target, *variants]
    all_features  = [extract_features(sc) for sc in all_scenarios]
    clustering_result = cluster_scenarios(
        [fv.feature_vector for fv in all_features],
        n_clusters=min(4, len(all_features)),
    )

    dashboard_html = output_dir / f"perturb_{scenario_id[:16]}.html"
    build_dashboard(
        feature_vectors=all_features,
        clustering_result=clustering_result,
        scenarios=all_scenarios,
        output_path=dashboard_html,
    )
    _console.print(f"Perturbation dashboard saved to [bold]{dashboard_html}[/bold]")


@app.command()
def serve(
    dashboard_path: Path = typer.Option(
        Path("outputs/dashboard.html"),
        help="Path to the dashboard HTML file to serve.",
    ),
    port: int = typer.Option(8050, help="Local port to serve on."),
) -> None:
    """Serve the Plotly dashboard HTML over a simple HTTP server."""
    import http.server
    import socketserver

    if not dashboard_path.exists():
        _console.print(f"[red]Dashboard not found at {dashboard_path}[/red]")
        _console.print("Run 'waymo-coverage cluster' first to generate a dashboard.")
        raise typer.Exit(code=1)

    serve_dir = dashboard_path.parent

    class _Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(serve_dir), **kwargs)

        def log_message(self, fmt: str, *args) -> None:  # suppress default logging
            _console.log(fmt % args)

    _console.print(f"Serving [bold]{dashboard_path.name}[/bold] at http://localhost:{port}/")
    _console.print("Press Ctrl+C to stop.")
    with socketserver.TCPServer(("", port), _Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            _console.print("\nServer stopped.")


if __name__ == "__main__":
    app()
