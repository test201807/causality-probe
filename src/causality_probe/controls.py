"""Positive and negative controls: known leaks the probe MUST detect, sealed twins it must NOT flag.

Each control is a feature builder applied to synthetic data on which the leak cannot hide. The
intraday controls use a 15-minute series whose price rises steadily within each day, so the
daily close is always the daily high and using it in the morning is unmistakable. (On daily
bars `resample('1D').last()` is the identity and the bug cannot show up: a control that cannot
fail proves nothing.)

Leaks covered
  shift_minus_1        next bar's return used as a feature
  final_statistic      last value of the sample broadcast to every row
  global_zscore        z-score with mean/std of the whole sample
  global_quantile      threshold from a quantile of the whole sample
  bfill                backward fill (uses later values by construction)
  centered_rolling     rolling window with center=True
  daily_resample_ffill daily close labelled at the START of the day and forward-filled into
                       intraday bars: at 03:00 you already know the 23:45 close
  misaligned_htf       higher-timeframe bars on another clock (e.g. a broker server offset),
                       labelled at bar start and merged into the engine's bars before they close
Sealed (clean) twins
  daily_resample_sealed  the same daily aggregate made available only on day D+1
  misaligned_htf_sealed  the same higher-timeframe bars stamped at their CLOSE time
  rolling_mean           a plain trailing rolling mean
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .probe import scan, probe_points


def synthetic_m15(days: int = 30) -> pd.DataFrame:
    """15-minute bars where price climbs from 100 to 200 inside every day."""
    idx = pd.date_range("2020-01-01", periods=days * 96, freq="15min")
    day = np.arange(len(idx)) // 96
    pos = np.arange(len(idx)) % 96
    close = 100.0 + 100.0 * pos / 95.0 + day * 0.01
    return pd.DataFrame({"open": close, "high": close * 1.0001, "low": close * 0.9999, "close": close},
                        index=idx)


def synthetic_random_walk(n: int = 2000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="h")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    return pd.DataFrame({"open": close, "high": close * 1.001, "low": close * 0.999, "close": close},
                        index=idx)


def _frame(b: pd.DataFrame, values) -> pd.DataFrame:
    return pd.DataFrame({"feature": np.asarray(values, dtype=float)}, index=b.index)


def shift_minus_1(b):
    return _frame(b, np.log(b["close"]).diff().shift(-1))


def final_statistic(b):
    return _frame(b, np.full(len(b), float(b["close"].iloc[-1])))


def global_zscore(b):
    c = b["close"]
    return _frame(b, (c - c.mean()) / c.std())


def global_quantile(b):
    c = b["close"]
    return _frame(b, c / c.quantile(0.9))


def bfill(b):
    s = b["close"].copy()
    s.iloc[::7] = np.nan
    return _frame(b, s.bfill())


def centered_rolling(b):
    return _frame(b, b["close"].rolling(25, center=True).mean())


def rolling_mean(b):
    return _frame(b, b["close"].rolling(25).mean())


def daily_resample_ffill(b):
    daily = b["close"].resample("1D").last()
    return _frame(b, daily.reindex(b.index, method="ffill"))


def daily_resample_sealed(b):
    daily = b["close"].resample("1D").last()
    daily.index = daily.index + pd.Timedelta(days=1)
    return _frame(b, daily.reindex(b.index, method="ffill"))


def _htf(b, sealed: bool):
    # 4-hour bars on a clock shifted by 1 hour (as with a broker server time offset)
    htf = b["close"].resample("4h", offset="1h").last()
    if sealed:
        htf.index = htf.index + pd.Timedelta(hours=4)
    merged = pd.merge_asof(pd.DataFrame(index=b.index), htf.rename("htf").to_frame(),
                           left_index=True, right_index=True, direction="backward")
    return _frame(b, merged["htf"])


def misaligned_htf(b):
    return _htf(b, sealed=False)


def misaligned_htf_sealed(b):
    return _htf(b, sealed=True)


# name -> (builder, must_leak, dataset)
CONTROLS = {
    "shift_minus_1": (shift_minus_1, True, "rw"),
    "final_statistic": (final_statistic, True, "rw"),
    "global_zscore": (global_zscore, True, "rw"),
    "global_quantile": (global_quantile, True, "rw"),
    "bfill": (bfill, True, "rw"),
    "centered_rolling": (centered_rolling, True, "rw"),
    "daily_resample_ffill": (daily_resample_ffill, True, "m15"),
    "misaligned_htf": (misaligned_htf, True, "m15"),
    "daily_resample_sealed": (daily_resample_sealed, False, "m15"),
    "misaligned_htf_sealed": (misaligned_htf_sealed, False, "m15"),
    "rolling_mean": (rolling_mean, False, "rw"),
}


def run_positive_controls(tol: float = 1e-9) -> dict[str, dict]:
    data = {"rw": synthetic_random_walk(), "m15": synthetic_m15()}
    points = {k: probe_points(v.index, n=40, warmup=300, seed=3) for k, v in data.items()}
    out = {}
    for name, (fn, must_leak, ds) in CONTROLS.items():
        leaks, _, _, _ = scan(fn, data[ds], points[ds], tol)
        detected = leaks.get("feature", 0) > 0
        out[name] = {"expected": "LEAK" if must_leak else "CLEAN", "points_differing": int(leaks["feature"]),
                     "ok": detected == must_leak}
    return out
