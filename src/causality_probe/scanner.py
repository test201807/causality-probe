"""Static scanner: flags code patterns that often introduce look-ahead.

The scanner is the map, the probe is the proof. A flagged line can be perfectly correct (for
example a negative shift used only to build the target label); only `probe()` decides.
"""
from __future__ import annotations

import re
from pathlib import Path

# (key, risk, regex, meaning, what to check)
PATTERNS = [
    ("resample_unsealed", "HIGH", r"\.resample\s*\(",
     "downsampling labels each period at its START",
     "stamp the aggregate at its CLOSE (index + period) before merging into a faster timeframe"),
    ("reindex_ffill", "HIGH", r"reindex\s*\([^)]*method\s*=\s*[\"'](ffill|pad)[\"']",
     "forward propagation of a series from another frequency",
     "if it comes from an unsealed resample it leaks; prefer merge_asof(direction='backward') on close times"),
    ("global_normalisation", "HIGH", r"\.mean\(\)\s*\)\s*/\s*\w+\.std\(\)|-\s*\w+\.mean\(\)\s*\)?\s*/",
     "normalising with the mean/std of the whole series", "use rolling or expanding windows"),
    ("global_quantile", "HIGH", r"\.quantile\s*\(", "threshold computed on the full history",
     "use rolling(n).quantile(); critical if it defines a regime"),
    ("backward_fill", "HIGH", r"\.bfill\s*\(|method\s*=\s*[\"'](bfill|backfill)[\"']",
     "backward fill uses future values by definition", "justify every occurrence"),
    ("centered_window", "HIGH", r"center\s*=\s*True", "half of a centered window lies in the future",
     "use trailing windows except for plots"),
    ("interpolate", "MEDIUM", r"\.interpolate\s*\(", "interpolation uses later values by default",
     "set limit_direction='forward'"),
    ("negative_shift", "MEDIUM", r"\.shift\s*\(\s*-\s*\d+", "brings future data to the current row",
     "legitimate for the target label, a leak if it ends up among input features"),
    ("sort_by_outcome", "MEDIUM", r"sort_values\s*\([^)]*(pnl|profit|return|sharpe)",
     "ordering by outcome inside the pipeline",
     "selection of assets, periods or parameters by outcome is design look-ahead no probe can catch"),
]
IGNORE = {".git", "__pycache__", ".venv", "venv", "env", "site-packages", "node_modules", "build", "dist"}
RANK = {"HIGH": 0, "MEDIUM": 1}


def scan_code(root: str | Path, include_tests: bool = False) -> list[dict]:
    root = Path(root).resolve()
    files = [root] if root.is_file() else sorted(root.rglob("*.py"))
    findings = []
    for f in files:
        rel = f.name if root.is_file() else str(f.relative_to(root))
        if any(p in IGNORE for p in Path(rel).parts):
            continue
        if not include_tests and re.search(r"(^|[/\\])(test_|tests?[/\\])", rel):
            continue
        try:
            lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for i, line in enumerate(lines, 1):
            code = line.split("#")[0]
            if not code.strip():
                continue
            for key, risk, pat, meaning, check in PATTERNS:
                if re.search(pat, code):
                    findings.append({"file": rel.replace("\\", "/"), "line": i, "risk": risk, "pattern": key,
                                     "code": line.strip()[:150], "meaning": meaning, "check": check})
    findings.sort(key=lambda x: (RANK[x["risk"]], x["file"], x["line"]))
    return findings
