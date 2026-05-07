# WaymoCoverageAnalyzer — TFRecord scenario parser
# Purpose: Read Waymo Open Dataset .tfrecord files and produce typed ScenarioData objects
# Author: <your-name>

"""Parse Waymo Open Dataset Motion .tfrecord files into typed Python data models."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

from pydantic import BaseModel, ConfigDict, ValidationError
from rich.console import Console

if TYPE_CHECKING:
    import tensorflow as tf
    from waymo_open_dataset.protos import scenario_pb2

# ---------------------------------------------------------------------------
# Proto backend selection (lazy, resolved once at first parse call)
# ---------------------------------------------------------------------------
# Preferred: official WOD SDK (Linux x86 / pip).
# Fallback:  compiled stub in waymo_coverage/scenario_pb2.py — works on all
#            platforms including macOS ARM where the SDK has no wheels.
def _load_scenario_pb2():
    """Return the scenario_pb2 module, preferring the WOD SDK if available."""
    try:
        from waymo_open_dataset.protos import scenario_pb2 as _pb2
        return _pb2
    except ImportError:
        from waymo_coverage import scenario_pb2 as _pb2  # compiled stub
        return _pb2

_console = Console(stderr=True)


@dataclass
class ParseStats:
    """Counters collected during a single load_scenarios call."""
    attempted: int = 0
    succeeded: int = 0
    proto_errors: int = 0       # malformed / truncated protobuf bytes
    validation_errors: int = 0  # pydantic rejected the parsed data
    other_errors: int = 0

    @property
    def failed(self) -> int:
        return self.proto_errors + self.validation_errors + self.other_errors

    def log(self) -> None:
        if self.failed == 0:
            _console.log(
                f"Parsed [green]{self.succeeded}[/green] / {self.attempted} records "
                f"([green]0 errors[/green])."
            )
        else:
            _console.log(
                f"Parsed [green]{self.succeeded}[/green] / {self.attempted} records — "
                f"[yellow]{self.proto_errors}[/yellow] proto errors, "
                f"[yellow]{self.validation_errors}[/yellow] validation errors, "
                f"[yellow]{self.other_errors}[/yellow] other errors."
            )


class AgentState(BaseModel):
    """Per-agent trajectory data extracted from one Waymo scenario."""

    model_config = ConfigDict(frozen=True)

    positions_x: list[float]
    positions_y: list[float]
    velocities_vx: list[float]
    velocities_vy: list[float]
    headings: list[float]
    valid: list[bool]
    object_type: int  # scenario_pb2.Track.ObjectType enum value


class ScenarioData(BaseModel):
    """All data needed for kinematic analysis of one Waymo scenario."""

    model_config = ConfigDict(frozen=True)

    scenario_id: str
    timestamps: list[float]
    agents: list[AgentState]
    sdc_track_index: int


def _parse_track(track: "scenario_pb2.Track") -> AgentState:
    """Convert a single proto Track into an AgentState.

    Args:
        track: A Waymo Open Dataset Track protobuf message.

    Returns:
        AgentState populated from the track's ObjectState sequence.
    """
    positions_x: list[float] = []
    positions_y: list[float] = []
    velocities_vx: list[float] = []
    velocities_vy: list[float] = []
    headings: list[float] = []
    valid: list[bool] = []

    for state in track.states:
        positions_x.append(float(state.center_x))
        positions_y.append(float(state.center_y))
        velocities_vx.append(float(state.velocity_x))
        velocities_vy.append(float(state.velocity_y))
        headings.append(float(state.heading))
        valid.append(bool(state.valid))

    return AgentState(
        positions_x=positions_x,
        positions_y=positions_y,
        velocities_vx=velocities_vx,
        velocities_vy=velocities_vy,
        headings=headings,
        valid=valid,
        object_type=int(track.object_type),
    )


def _unwrap_tf_example(raw_bytes: bytes) -> bytes:
    """Extract the raw Scenario proto bytes from a tf.train.Example wrapper.

    The tf_example/ dataset format stores each Scenario proto as a bytes value
    under the 'scenario' key of a tf.train.Example.  The scenario/ format stores
    raw Scenario proto bytes directly — this function detects which format is in
    use and unwraps only when necessary.

    Args:
        raw_bytes: Raw bytes from a TFRecord entry (either format).

    Returns:
        Bytes of the serialised Scenario proto, ready for ParseFromString.
    """
    import tensorflow as tf  # lazy import — tf is only needed here

    example = tf.train.Example()
    try:
        example.ParseFromString(raw_bytes)
    except Exception:
        # Bytes are already a raw Scenario proto (scenario/ format).
        return raw_bytes

    scenario_bytes_list = example.features.feature.get("scenario")
    if scenario_bytes_list and scenario_bytes_list.bytes_list.value:
        # tf_example/ format: unwrap the inner Scenario bytes.
        return scenario_bytes_list.bytes_list.value[0]

    # No 'scenario' key — assume raw Scenario proto.
    return raw_bytes


def _parse_scenario(raw_bytes: bytes) -> ScenarioData:
    """Deserialize one TFRecord bytes entry into a ScenarioData.

    Handles both Waymo TFRecord formats automatically:
    - ``scenario/``    — raw serialised Scenario proto per record
    - ``tf_example/``  — tf.train.Example wrapping a serialised Scenario proto

    Args:
        raw_bytes: Raw bytes from a TFRecord entry (either format).

    Returns:
        ScenarioData for the parsed scenario.
    """
    scenario_pb2 = _load_scenario_pb2()

    scenario_bytes = _unwrap_tf_example(raw_bytes)
    proto = scenario_pb2.Scenario()
    proto.ParseFromString(scenario_bytes)

    timestamps = [float(ts) for ts in proto.timestamps_seconds]
    agents = [_parse_track(track) for track in proto.tracks]

    return ScenarioData(
        scenario_id=proto.scenario_id,
        timestamps=timestamps,
        agents=agents,
        sdc_track_index=int(proto.sdc_track_index),
    )


def _iter_raw_records(tfrecord_path: Path) -> Iterator[bytes]:
    """Yield raw serialized bytes from a TFRecord file.

    Args:
        tfrecord_path: Path to a Waymo .tfrecord file.

    Yields:
        Raw bytes for each record.
    """
    import tensorflow as tf  # lazy: only needed for TFRecord parsing

    dataset = tf.data.TFRecordDataset(str(tfrecord_path), compression_type="")
    for raw_record in dataset:
        yield raw_record.numpy()


def load_scenarios(
    tfrecord_path: Path,
    max_scenarios: int = 100,
) -> list[ScenarioData]:
    """Parse a .tfrecord file and return a list of ScenarioData objects.

    Reads up to *max_scenarios* scenario protos from the given TFRecord.
    Skips malformed records individually — a single corrupt entry does not abort
    the pipeline.  Parse statistics are logged at INFO level when the call returns.

    Args:
        tfrecord_path: Path to a Waymo Open Dataset Motion .tfrecord file.
        max_scenarios: Maximum number of scenarios to parse (default 100).

    Returns:
        List of ScenarioData, one per successfully parsed record.

    Raises:
        FileNotFoundError: If *tfrecord_path* does not exist.
        RuntimeError: If every record in the file fails to parse (nothing usable
            was produced), indicating a likely file-format or SDK mismatch.
    """
    if not tfrecord_path.exists():
        raise FileNotFoundError(f"TFRecord file not found: {tfrecord_path}")

    scenarios: list[ScenarioData] = []
    stats = ParseStats()
    _console.log(f"Loading scenarios from [bold]{tfrecord_path}[/bold] (max={max_scenarios})")

    for raw_bytes in _iter_raw_records(tfrecord_path):
        if len(scenarios) >= max_scenarios:
            break
        stats.attempted += 1
        try:
            scenario = _parse_scenario(raw_bytes)
            scenarios.append(scenario)
            stats.succeeded += 1
        except ValidationError as exc:
            # Parsed proto values violate the pydantic model — log the first error only.
            stats.validation_errors += 1
            first = exc.errors()[0]
            _console.log(
                f"[yellow]Warning:[/yellow] record {stats.attempted} failed validation "
                f"({first['loc']}: {first['msg']}) — skipped."
            )
        except Exception as exc:  # catches DecodeError and anything TF raises
            # Treat as a proto-level decode failure.
            stats.proto_errors += 1
            _console.log(
                f"[yellow]Warning:[/yellow] record {stats.attempted} parse error "
                f"({type(exc).__name__}: {exc}) — skipped."
            )

    stats.log()

    if stats.succeeded == 0 and stats.attempted > 0:
        raise RuntimeError(
            f"All {stats.attempted} records in {tfrecord_path} failed to parse. "
            "Check that the file is a valid Waymo Open Dataset Motion TFRecord and "
            "that the waymo-open-dataset SDK version matches the file format."
        )

    return scenarios
