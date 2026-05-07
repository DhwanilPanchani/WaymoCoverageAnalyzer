# WaymoCoverageAnalyzer — Scenario clustering
# Purpose: Standardize feature vectors, run KMeans, project to 2D PCA, identify coverage gaps
# Author: <your-name>

"""Cluster scenario feature vectors and identify underrepresented coverage regions."""

import numpy as np
from pydantic import BaseModel, ConfigDict
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class ClusteringResult(BaseModel):
    """Output of the KMeans clustering step."""

    model_config = ConfigDict(frozen=True)

    n_clusters: int
    labels: list[int]
    cluster_sizes: list[int]
    cluster_centers_pca: list[list[float]]  # shape (n_clusters, 2)
    inertia: float
    coverage_gap_indices: list[int]  # scenario indices in the smallest clusters


# Fraction of clusters considered as coverage gaps (bottom quartile by size).
_GAP_FRACTION = 0.25


def cluster_scenarios(
    feature_vectors: list[list[float]],
    n_clusters: int = 8,
    random_state: int = 42,
) -> ClusteringResult:
    """Standardize feature vectors, run KMeans, project to 2D PCA for visualization.

    Coverage gaps are defined as scenarios that fall in the smallest clusters
    (bottom *_GAP_FRACTION* fraction of clusters, ranked by size).

    Args:
        feature_vectors: List of per-scenario numeric feature lists. All must
            have the same length.
        n_clusters: Number of KMeans clusters (default 8).
        random_state: Random seed for reproducibility (default 42).

    Returns:
        ClusteringResult with labels, PCA projections, and coverage gap indices.

    Raises:
        ValueError: If feature_vectors is empty or all vectors have zero length.
    """
    if not feature_vectors:
        raise ValueError("feature_vectors must not be empty")

    feature_matrix = np.array(feature_vectors, dtype=np.float64)
    if feature_matrix.ndim != 2 or feature_matrix.shape[1] == 0:
        raise ValueError(f"Expected 2-D feature matrix, got shape {feature_matrix.shape}")

    # Clamp n_clusters to the number of available samples.
    effective_clusters = min(n_clusters, len(feature_vectors))

    # --- Standardize ---
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_matrix)

    # --- KMeans ---
    kmeans = KMeans(
        n_clusters=effective_clusters,
        random_state=random_state,
        n_init="auto",
    )
    labels_array = kmeans.fit_predict(scaled)

    # --- PCA to 2D for scatter visualization ---
    n_components = min(2, scaled.shape[0], scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=random_state)
    pca.fit(scaled)
    centers_2d = pca.transform(kmeans.cluster_centers_[:, :scaled.shape[1]])
    # Zero-pad to 2D if only one component available.
    if centers_2d.shape[1] < 2:
        centers_2d = np.hstack([centers_2d, np.zeros((effective_clusters, 1))])

    # --- Cluster sizes ---
    cluster_sizes = [
        int(np.sum(labels_array == cluster_id))
        for cluster_id in range(effective_clusters)
    ]

    # --- Coverage gap indices ---
    n_gap_clusters = max(1, int(np.ceil(effective_clusters * _GAP_FRACTION)))
    gap_cluster_ids = set(
        sorted(range(effective_clusters), key=lambda c: cluster_sizes[c])[:n_gap_clusters]
    )
    coverage_gap_indices = [
        int(idx)
        for idx, label in enumerate(labels_array)
        if int(label) in gap_cluster_ids
    ]

    return ClusteringResult(
        n_clusters=effective_clusters,
        labels=[int(label) for label in labels_array],
        cluster_sizes=cluster_sizes,
        cluster_centers_pca=centers_2d.tolist(),
        inertia=float(kmeans.inertia_),
        coverage_gap_indices=coverage_gap_indices,
    )
