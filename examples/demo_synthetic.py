"""Minimal demo: a clean feature set passes, a feature set with a classic resample bug fails.

Run:  python examples/demo_synthetic.py
"""
import numpy as np
import pandas as pd

from causality_probe import probe
from causality_probe.controls import synthetic_m15


def features_clean(b):
    c = b["close"]
    daily = c.resample("1D").last()
    daily.index = daily.index + pd.Timedelta(days=1)          # sealed: yesterday's close
    return pd.DataFrame({"ret_4": np.log(c).diff(4),
                         "prev_day_close": daily.reindex(b.index, method="ffill")}, index=b.index)


def features_buggy(b):
    c = b["close"]
    daily = c.resample("1D").last()                            # labelled at the START of the day
    return pd.DataFrame({"ret_4": np.log(c).diff(4),
                         "day_close": daily.reindex(b.index, method="ffill")}, index=b.index)


if __name__ == "__main__":
    data = synthetic_m15(days=40)
    for name, fn in (("clean", features_clean), ("buggy", features_buggy)):
        print(f"--- {name}")
        print(probe(fn, data, warmup=400).summary())
