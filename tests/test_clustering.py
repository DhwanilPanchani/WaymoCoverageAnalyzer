# WaymoCoverageAnalyzer — Clustering unit tests
# Purpose: Verify KMeans clustering output shape, coverage gap detection, and edge cases
# Author: <your-name>

"""Tests for waymo_coverage.clustering: ClusteringResult shape and coverage gap logic."""

import pytest

from waymo_coverage.clustering import ClusteringResult, cluster_scenarios


def _make_feature_vectors(n_scenarios: int, n_features: int = 13) -> list[list[float]]:
    """Build dummy feature vectors with distinct per-scenario values.

    Args:
        n_scenarios: Number of feature vectors to generate.
        n_features: Number of features per vector.

    Returns:
        List of feature vectors.
    """
    return [
        [float(scenario_idx * n_features + feature_idx) for feature_idx in range(n_features)]
        for scenario_idx in range(n_scenarios)
    ]


def test_cluster_result_label_count() -> None:
    """Number of labels must equal number of input feature vectors."""
    vectors = _make_feature_vectors(20)
    result = cluster_scenarios(vectors, n_clusters=4)
    assert len(result.labels) == 20


def test_cluster_result_n_clusters_clamped() -> None:
    """n_clusters must be clamped to the number of available samples."""
    vectors = _make_feature_vectors(3)
    result = cluster_scenarios(vectors, n_clusters=8)
    assert result.n_clusters <= 3


def test_cluster_sizes_sum_to_total() -> None:
    """Sum of all cluster_sizes must equal total number of scenarios."""
    vectors = _make_feature_vectors(30)
    result = cluster_scenarios(vectors, n_clusters=5)
    assert sum(result.cluster_sizes) == 30


def test_cluster_centers_pca_shape() -> None:
    """cluster_centers_pca must have shape (n_clusters, 2)."""
    n_clusters = 4
    vectors = _make_feature_vectors(20)
    result = cluster_scenarios(vectors, n_clusters=n_clusters)
    assert len(result.cluster_centers_pca) == result.n_clusters
    for center in result.cluster_centers_pca:
        assert len(center) == 2


def test_coverage_gap_indices_are_valid() -> None:
    """All coverage_gap_indices must be valid scenario indices."""
    n_scenarios = 20
    vectors = _make_feature_vectors(n_scenarios)
    result = cluster_scenarios(vectors, n_clusters=4)
    for idx in result.coverage_gap_indices:
        assert 0 <= idx < n_scenarios


def test_coverage_gap_indices_nonempty() -> None:
    """At least one coverage gap index must be identified for a non-trivial dataset."""
    vectors = _make_feature_vectors(20)
    result = cluster_scenarios(vectors, n_clusters=4)
    assert len(result.coverage_gap_indices) > 0


def test_inertia_is_positive() -> None:
    """KMeans inertia must be non-negative."""
    vectors = _make_feature_vectors(15)
    result = cluster_scenarios(vectors, n_clusters=3)
    assert result.inertia >= 0.0


def test_empty_feature_vectors_raises() -> None:
    """cluster_scenarios must raise ValueError on empty input."""
    with pytest.raises(ValueError, match="must not be empty"):
        cluster_scenarios([])


def test_labels_in_valid_range() -> None:
    """All cluster labels must be in [0, n_clusters)."""
    vectors = _make_feature_vectors(25)
    result = cluster_scenarios(vectors, n_clusters=5)
    for label in result.labels:
        assert 0 <= label < result.n_clusters


def test_single_scenario() -> None:
    """Single-scenario input must produce a valid result with one cluster."""
    vectors = _make_feature_vectors(1)
    result = cluster_scenarios(vectors, n_clusters=8)
    assert result.n_clusters == 1
    assert result.labels == [0]
