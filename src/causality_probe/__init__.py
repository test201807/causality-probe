"""causality-probe: detect look-ahead bias in backtest features by truncation equivalence."""
from .controls import CONTROLS, run_positive_controls
from .probe import ProbeReport, probe, probe_points
from .scanner import scan_code

__all__ = ["probe", "ProbeReport", "probe_points", "run_positive_controls", "CONTROLS", "scan_code"]
__version__ = "0.1.0"
