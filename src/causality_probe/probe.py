"""Truncation-equivalence probe for look-ahead bias.

A feature is causal at time t only if it depends on data available up to t. Therefore

    build(full_history).loc[t]  ==  build(history_up_to_t).iloc[-1]

must hold for every column and every t. Any difference is look-ahead by definition, without
reading a single line of the feature code.

The probe evaluates random points plus calendar boundaries (day, week, month), because that is
where resampling bugs live. A probe that finds nothing proves nothing unless it is shown to
catch known leaks, so `probe()` runs the positive controls of `causality_probe.controls` by
default and refuses to give a clean verdict if any control is missed.

Independence from the future is a NECESSARY condition, not a sufficient one: a clean column can
still be wrong or useless. This closes one door, but it is a door many backtests walk through.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

import numpy as np
import pandas as pd

Builder = Callable[[pd.DataFrame], pd.DataFrame]
TOL = 1e-9

VERDICT_CLEAN = "NO_LEAK"
VERDICT_LEAK = "LEAK_DETECTED"
VERDICT_INVALID = "INVALID_CONTROL_FAILED"


@dataclass
class ProbeReport:
    verdict: str
    rows: int
    points: int
    columns: int
    leaking_columns: list[str]
    leak_rate_pct: dict[str, float]
    examples: dict[str, dict]
    controls: dict[str, dict] = field(default_factory=dict)
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def clean(self) -> bool:
        return self.verdict == VERDICT_CLEAN

    def summary(self) -> str:
        lines = [f"verdict          : {self.verdict}",
                 f"rows / points    : {self.rows} / {self.points}",
                 f"columns tested   : {self.columns}"]
        if self.controls:
            ok = sum(c["ok"] for c in self.controls.values())
            lines.append(f"positive controls: {ok}/{len(self.controls)} behaved as expected")
        for c in self.leaking_columns:
            e = self.examples.get(c, {})
            lines.append(f"  LEAK {c}: {self.leak_rate_pct[c]:.2f}% of points "
                         f"(e.g. row {e.get('row')}: full={e.get('with_future')} vs "
                         f"truncated={e.get('without_future')})")
        return "\n".join(lines)


def _differs(x, y, tol: float) -> bool:
    xn, yn = pd.isna(x), pd.isna(y)
    if xn and yn:
        return False
    if xn != yn:
        return True
    try:
        return abs(float(x) - float(y)) > tol
    except (TypeError, ValueError):
        return x != y


def _num(v):
    try:
        return None if pd.isna(v) else float(v)
    except (TypeError, ValueError):
        return str(v)


def probe_points(index: pd.Index, n: int = 120, warmup: int | None = None, seed: int = 11) -> list[int]:
    """Random positions after warm-up plus the two bars around every day/week/month boundary."""
    size = len(index)
    warmup = size // 2 if warmup is None else warmup
    if size - warmup < 10:
        raise ValueError("not enough rows after warm-up to probe")
    rng = np.random.default_rng(seed)
    pts = set(rng.integers(warmup, size - 1, size=min(n, size - 1 - warmup)).tolist())
    if isinstance(index, pd.DatetimeIndex):
        idx = index.tz_convert(None) if index.tz is not None else index
        for period in (idx.normalize(), idx.to_period("W"), idx.to_period("M")):
            values = np.asarray(period)
            change = np.nonzero(values[1:] != values[:-1])[0]
            change = change[change >= warmup]
            if len(change) > 40:
                change = change[np.linspace(0, len(change) - 1, 40).astype(int)]
            for k in change:
                for j in (int(k), int(k) + 1):
                    if warmup <= j < size - 1:
                        pts.add(j)
    return sorted(pts)


def scan(build: Builder, data: pd.DataFrame, points: list[int], tol: float = TOL):
    """Compare build(data) with build(data[:t+1]) at each point. Returns counts and examples."""
    full = build(data)
    cols = list(full.columns)
    leaks = {c: 0 for c in cols}
    compared = {c: 0 for c in cols}
    examples: dict[str, dict] = {}
    for t in points:
        trunc = build(data.iloc[: t + 1])
        a, b = full.iloc[t], trunc.iloc[-1]
        for c in cols:
            x, y = a.get(c), b.get(c)
            if pd.isna(x) and pd.isna(y):
                continue
            compared[c] += 1
            if _differs(x, y, tol):
                leaks[c] += 1
                examples.setdefault(c, {"row": int(t), "with_future": _num(x), "without_future": _num(y)})
    return leaks, compared, examples, cols


def probe(build: Builder, data: pd.DataFrame, n_points: int = 120, warmup: int | None = None,
          seed: int = 11, tol: float = TOL, run_controls: bool = True) -> ProbeReport:
    """Run the truncation probe on `build` (DataFrame -> DataFrame with the same index).

    With `run_controls=True` (default) the probe first proves it can detect the known leaks in
    `causality_probe.controls` and keep the sealed versions clean; otherwise the verdict is
    INVALID_CONTROL_FAILED and the findings must not be trusted.
    """
    if len(data) < 200:
        raise ValueError("data needs at least 200 rows so that indicators warm up")
    pts = probe_points(data.index, n_points, warmup, seed)
    leaks, compared, examples, cols = scan(build, data, pts, tol)
    rate = {c: round(100.0 * leaks[c] / max(compared[c], 1), 2) for c in cols}
    leaking = sorted(c for c in cols if leaks[c] > 0)
    report = ProbeReport(verdict=VERDICT_LEAK if leaking else VERDICT_CLEAN, rows=len(data),
                         points=len(pts), columns=len(cols), leaking_columns=leaking,
                         leak_rate_pct={c: rate[c] for c in leaking}, examples=examples)
    if run_controls:
        from .controls import run_positive_controls
        report.controls = run_positive_controls(tol=tol)
        if not all(c["ok"] for c in report.controls.values()):
            report.verdict = VERDICT_INVALID
    return report
