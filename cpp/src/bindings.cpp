// WaymoCoverageAnalyzer — pybind11 Python bindings for the C++ kinematics engine
// Purpose: Expose KinematicFeatures, KinematicsEngine.compute, and compute_ttc to Python
// Author: <your-name>

#include "kinematics.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace {

/// Build an AgentTrajectory from numpy arrays without copying data.
/// Arrays must be 1-D, C-contiguous, and all the same length.
AgentTrajectory make_trajectory(
    py::array_t<double> positions_x,
    py::array_t<double> positions_y,
    py::array_t<double> velocities_vx,
    py::array_t<double> velocities_vy,
    py::array_t<double> headings,
    py::array_t<bool>   valid,
    double dt
) {
    auto buf_x   = positions_x.request();
    auto buf_y   = positions_y.request();
    auto buf_vx  = velocities_vx.request();
    auto buf_vy  = velocities_vy.request();
    auto buf_h   = headings.request();
    auto buf_v   = valid.request();

    const std::size_t n = static_cast<std::size_t>(buf_x.size);

    AgentTrajectory traj{
        .positions_x    = std::span<const double>(static_cast<const double*>(buf_x.ptr),   n),
        .positions_y    = std::span<const double>(static_cast<const double*>(buf_y.ptr),   n),
        .velocities_vx  = std::span<const double>(static_cast<const double*>(buf_vx.ptr),  n),
        .velocities_vy  = std::span<const double>(static_cast<const double*>(buf_vy.ptr),  n),
        .headings       = std::span<const double>(static_cast<const double*>(buf_h.ptr),   n),
        .valid          = std::span<const bool>  (static_cast<const bool*>  (buf_v.ptr),   n),
        .dt             = dt
    };
    return traj;
}

} // anonymous namespace

PYBIND11_MODULE(waymo_kinematics, module) {
    module.doc() = "C++20 kinematic feature engine for Waymo scenario analysis.";

    py::class_<KinematicFeatures>(module, "KinematicFeatures",
        "All scalar kinematic features extracted from one agent trajectory.")
        .def(py::init<>())
        .def_readwrite("mean_speed",          &KinematicFeatures::mean_speed)
        .def_readwrite("max_speed",           &KinematicFeatures::max_speed)
        .def_readwrite("mean_acceleration",           &KinematicFeatures::mean_acceleration)
        .def_readwrite("max_acceleration_magnitude", &KinematicFeatures::max_acceleration_magnitude)
        .def_readwrite("mean_jerk",                  &KinematicFeatures::mean_jerk)
        .def_readwrite("max_jerk_magnitude",         &KinematicFeatures::max_jerk_magnitude)
        .def_readwrite("mean_curvature",             &KinematicFeatures::mean_curvature)
        .def_readwrite("max_curvature",              &KinematicFeatures::max_curvature)
        .def_readwrite("mean_lateral_acceleration",  &KinematicFeatures::mean_lateral_acceleration)
        .def_readwrite("max_lateral_acceleration",   &KinematicFeatures::max_lateral_acceleration)
        .def_readwrite("path_length",                &KinematicFeatures::path_length)
        .def_readwrite("displacement",               &KinematicFeatures::displacement)
        .def_readwrite("lane_change_score",          &KinematicFeatures::lane_change_score)
        .def("__repr__", [](const KinematicFeatures& f) {
            return "<KinematicFeatures mean_speed=" + std::to_string(f.mean_speed) +
                   " max_speed=" + std::to_string(f.max_speed) + ">";
        });

    py::class_<KinematicsEngine>(module, "KinematicsEngine",
        "Static methods for computing kinematic features and TTC.")
        .def_static(
            "compute",
            [](py::array_t<double> px, py::array_t<double> py_,
               py::array_t<double> vx, py::array_t<double> vy,
               py::array_t<double> h,  py::array_t<bool>   valid,
               double dt) -> KinematicFeatures {
                auto traj = make_trajectory(px, py_, vx, vy, h, valid, dt);
                return KinematicsEngine::compute(traj);
            },
            py::arg("positions_x"),
            py::arg("positions_y"),
            py::arg("velocities_vx"),
            py::arg("velocities_vy"),
            py::arg("headings"),
            py::arg("valid"),
            py::arg("dt") = 0.1,
            "Compute kinematic features for one agent trajectory.\n\n"
            "All array arguments must be 1-D numpy arrays of equal length.\n"
            "valid is a boolean array; timesteps where valid[i]=False are skipped.\n"
            "dt is the timestep in seconds (default 0.1 for Waymo Motion dataset)."
        )
        .def_static(
            "compute_ttc",
            [](py::array_t<double> ego_px,  py::array_t<double> ego_py,
               py::array_t<double> ego_vx,  py::array_t<double> ego_vy,
               py::array_t<double> ego_h,   py::array_t<bool>   ego_valid,
               py::array_t<double> oth_px,  py::array_t<double> oth_py,
               py::array_t<double> oth_vx,  py::array_t<double> oth_vy,
               py::array_t<double> oth_h,   py::array_t<bool>   oth_valid,
               double dt) -> double {
                auto ego   = make_trajectory(ego_px, ego_py, ego_vx, ego_vy, ego_h, ego_valid, dt);
                auto other = make_trajectory(oth_px, oth_py, oth_vx, oth_vy, oth_h, oth_valid, dt);
                return KinematicsEngine::compute_ttc(ego, other);
            },
            py::arg("ego_positions_x"),
            py::arg("ego_positions_y"),
            py::arg("ego_velocities_vx"),
            py::arg("ego_velocities_vy"),
            py::arg("ego_headings"),
            py::arg("ego_valid"),
            py::arg("other_positions_x"),
            py::arg("other_positions_y"),
            py::arg("other_velocities_vx"),
            py::arg("other_velocities_vy"),
            py::arg("other_headings"),
            py::arg("other_valid"),
            py::arg("dt") = 0.1,
            "Compute minimum positive time-to-collision between two agents.\n\n"
            "Returns 999.0 if no positive TTC is found (agents not converging)."
        );
}
