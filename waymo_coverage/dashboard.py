# WaymoCoverageAnalyzer — Interactive Plotly dashboard
# Purpose: Build and save a 4-panel HTML dashboard showing clustering results and perturbations
# Author: <your-name>

"""Build and save an interactive Plotly HTML dashboard with 4 analysis panels."""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from waymo_coverage.clustering import ClusteringResult
from waymo_coverage.features import ScenarioFeatureVector
from waymo_coverage.parser import ScenarioData
from waymo_coverage.perturbation import PerturbationConfig, perturb_scenario

# Feature names matching the order in ScenarioFeatureVector.feature_vector.
_FEATURE_NAMES = [
    "num_agents",
    "num_vehicles",
    "num_pedestrians",
    "num_cyclists",
    "mean_speed",
    "max_speed",
    "mean_acceleration",
    "max_jerk",
    "mean_curvature",
    "mean_lateral_accel",
    "max_lateral_accel",
    "min_ttc",
    "interaction_density",
    "heading_variance",
    "path_length_variance",
]

# Amber color for coverage gap highlighting.
_GAP_COLOR = "rgba(255, 165, 0, 0.85)"
_NORMAL_COLOR = "rgba(99, 110, 250, 0.75)"


def _pca_scatter_panel(
    feature_vectors: list[ScenarioFeatureVector],
    clustering_result: ClusteringResult,
) -> go.Figure:
    """Build a scatter panel of scenarios in PCA space, colored by cluster label.

    Args:
        feature_vectors: Ordered list matching clustering_result.labels.
        clustering_result: Output of cluster_scenarios().

    Returns:
        Plotly Scatter trace dict for embedding in a subplot.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    matrix = np.array([fv.feature_vector for fv in feature_vectors], dtype=np.float64)
    scaled = StandardScaler().fit_transform(matrix)
    n_components = min(2, scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    coords = pca.fit_transform(scaled)
    if coords.shape[1] < 2:
        coords = np.hstack([coords, np.zeros((len(coords), 1))])

    gap_set = set(clustering_result.coverage_gap_indices)
    colors = [
        _GAP_COLOR if idx in gap_set else _NORMAL_COLOR
        for idx in range(len(feature_vectors))
    ]
    hover_texts = [
        f"Scenario: {fv.scenario_id}<br>Cluster: {clustering_result.labels[idx]}<br>"
        f"Speed: {fv.mean_agent_speed:.2f} m/s<br>TTC: {fv.min_ttc:.1f} s"
        for idx, fv in enumerate(feature_vectors)
    ]

    return go.Scatter(
        x=coords[:, 0].tolist(),
        y=coords[:, 1].tolist(),
        mode="markers",
        marker=dict(color=colors, size=8, opacity=0.8),
        text=hover_texts,
        hoverinfo="text",
        name="Scenarios",
    )


def _cluster_bar_panel(clustering_result: ClusteringResult) -> go.Bar:
    """Build a bar chart of cluster sizes, highlighting coverage gap clusters in amber.

    Args:
        clustering_result: Output of cluster_scenarios().

    Returns:
        Plotly Bar trace.
    """
    sorted_clusters = sorted(
        range(clustering_result.n_clusters),
        key=lambda c: clustering_result.cluster_sizes[c],
        reverse=True,
    )
    sizes = [clustering_result.cluster_sizes[c] for c in sorted_clusters]

    gap_cluster_set: set[int] = set()
    gap_scenario_labels = set(
        clustering_result.labels[idx]
        for idx in clustering_result.coverage_gap_indices
    )
    gap_cluster_set = gap_scenario_labels

    bar_colors = [
        _GAP_COLOR if sorted_clusters[i] in gap_cluster_set else _NORMAL_COLOR
        for i in range(len(sorted_clusters))
    ]

    return go.Bar(
        x=[f"Cluster {c}" for c in sorted_clusters],
        y=sizes,
        marker_color=bar_colors,
        name="Cluster Size",
        hovertemplate="Cluster %{x}<br>Size: %{y}<extra></extra>",
    )


def _feature_heatmap_panel(
    feature_vectors: list[ScenarioFeatureVector],
    clustering_result: ClusteringResult,
) -> go.Heatmap:
    """Build a feature heatmap with scenarios sorted by cluster label.

    Args:
        feature_vectors: Ordered list matching clustering_result.labels.
        clustering_result: Output of cluster_scenarios().

    Returns:
        Plotly Heatmap trace.
    """
    # Sort scenarios by cluster label for visual grouping.
    sorted_indices = sorted(
        range(len(feature_vectors)),
        key=lambda idx: clustering_result.labels[idx],
    )
    matrix = np.array(
        [feature_vectors[idx].feature_vector for idx in sorted_indices],
        dtype=np.float64,
    )
    # Z-score normalize each column for uniform color scale.
    col_std = np.std(matrix, axis=0)
    col_std[col_std < 1e-9] = 1.0
    matrix_norm = (matrix - np.mean(matrix, axis=0)) / col_std

    scenario_labels = [
        f"S{idx} (C{clustering_result.labels[sorted_indices[idx]]})"
        for idx in range(len(sorted_indices))
    ]

    return go.Heatmap(
        z=matrix_norm.tolist(),
        x=_FEATURE_NAMES[:matrix_norm.shape[1]],
        y=scenario_labels,
        colorscale="RdBu_r",
        zmid=0.0,
        colorbar=dict(title="Z-score"),
        hovertemplate="Feature: %{x}<br>Scenario: %{y}<br>Value: %{z:.2f}<extra></extra>",
    )


def _perturbation_panel(
    scenario: ScenarioData,
    agent_index: int = 0,
    n_variants: int = 3,
) -> list[go.Scatter]:
    """Build trajectory overlay traces for original + perturbed scenario variants.

    Args:
        scenario: The original scenario.
        agent_index: Which agent to perturb.
        n_variants: Number of counterfactual variants to show.

    Returns:
        List of Plotly Scatter traces (one original + n_variants perturbed).
    """
    config = PerturbationConfig(
        speed_scale=1.0,
        heading_offset=0.1,
        agent_index=agent_index,
    )
    variants = perturb_scenario(scenario, config, n_variants=n_variants)

    original_agent = scenario.agents[agent_index]
    valid_mask = np.asarray(original_agent.valid, dtype=bool)
    orig_x = np.asarray(original_agent.positions_x)[valid_mask].tolist()
    orig_y = np.asarray(original_agent.positions_y)[valid_mask].tolist()

    traces: list[go.Scatter] = [
        go.Scatter(
            x=orig_x,
            y=orig_y,
            mode="lines+markers",
            line=dict(color="blue", width=2),
            marker=dict(size=4),
            name="Original",
        )
    ]

    variant_colors = ["red", "green", "purple", "orange", "cyan"]
    for variant_idx, variant in enumerate(variants):
        variant_agent = variant.agents[agent_index]
        v_mask = np.asarray(variant_agent.valid, dtype=bool)
        var_x = np.asarray(variant_agent.positions_x)[v_mask].tolist()
        var_y = np.asarray(variant_agent.positions_y)[v_mask].tolist()
        color = variant_colors[variant_idx % len(variant_colors)]
        traces.append(go.Scatter(
            x=var_x,
            y=var_y,
            mode="lines",
            line=dict(color=color, width=1.5, dash="dash"),
            name=f"Variant {variant_idx + 1}",
        ))

    return traces


def build_dashboard(
    feature_vectors: list[ScenarioFeatureVector],
    clustering_result: ClusteringResult,
    scenarios: list[ScenarioData],
    output_path: Path,
    selected_scenario_index: int = 0,
) -> None:
    """Build and save a 4-panel interactive Plotly HTML dashboard.

    Panels:
    1. PCA scatter: scenarios coloured by cluster, gaps highlighted in amber.
    2. Cluster size bar chart, coverage gaps in amber.
    3. Feature heatmap: scenarios × features, Z-score normalised.
    4. Perturbation preview: original vs. variant trajectories for one scenario.

    Args:
        feature_vectors: Extracted scenario feature vectors.
        clustering_result: Output of cluster_scenarios().
        scenarios: Original parsed scenarios (needed for perturbation panel).
        output_path: Destination for the HTML file.
        selected_scenario_index: Which scenario to use in the perturbation panel.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Scenario Coverage — PCA Space",
            "Cluster Sizes (amber = coverage gap)",
            "Feature Heatmap (Z-score normalised)",
            "Perturbation Preview — Original vs. Variants",
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    # Panel 1 — PCA scatter.
    scatter_trace = _pca_scatter_panel(feature_vectors, clustering_result)
    fig.add_trace(scatter_trace, row=1, col=1)

    # Panel 2 — cluster bar chart.
    bar_trace = _cluster_bar_panel(clustering_result)
    fig.add_trace(bar_trace, row=1, col=2)

    # Panel 3 — feature heatmap.
    heatmap_trace = _feature_heatmap_panel(feature_vectors, clustering_result)
    fig.add_trace(heatmap_trace, row=2, col=1)

    # Panel 4 — perturbation preview.
    scenario_idx = min(selected_scenario_index, len(scenarios) - 1)
    perturb_traces = _perturbation_panel(scenarios[scenario_idx])
    for trace in perturb_traces:
        fig.add_trace(trace, row=2, col=2)

    fig.update_layout(
        title_text="WaymoCoverageAnalyzer — Dataset Coverage Dashboard",
        title_font_size=18,
        height=900,
        width=1400,
        showlegend=True,
        template="plotly_white",
    )
    fig.update_xaxes(title_text="PC 1", row=1, col=1)
    fig.update_yaxes(title_text="PC 2", row=1, col=1)
    fig.update_xaxes(title_text="Cluster", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_xaxes(title_text="X (m)", row=2, col=2)
    fig.update_yaxes(title_text="Y (m)", row=2, col=2)

    fig.write_html(str(output_path))
