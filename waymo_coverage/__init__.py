# WaymoCoverageAnalyzer — Python package initializer
# Purpose: Expose top-level public API for the waymo_coverage package
# Author: <your-name>

"""Waymo Coverage Analyzer.

A pipeline that ingests Waymo Open Dataset Motion scenarios, extracts
kinematic features via a high-performance C++20 engine (pybind11), clusters
scenarios to surface coverage gaps, and renders an interactive Plotly dashboard.
"""

__version__ = "0.1.0"
