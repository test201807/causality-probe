"""Probe a feature set on real public data: BTCUSDT 15-minute klines from data.binance.vision.

Downloads one month of public klines (no account or API key needed), then runs the probe on a
clean feature set and on the same set with an unsealed higher-timeframe merge.

Run:  python examples/demo_binance_public.py 2024-01
"""
import io
import sys
import urllib.request
import zipfile

import pandas as pd

from causality_probe import probe
from causality_probe.controls import misaligned_htf

URL = "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/15m/BTCUSDT-15m-{m}.zip"
COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "qv", "n", "tb", "tq", "ig"]


def load(month: str) -> pd.DataFrame:
    raw = urllib.request.urlopen(URL.format(m=month), timeout=60).read()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        df = pd.read_csv(z.open(z.namelist()[0]), header=None, names=COLS)
    unit = "us" if df["open_time"].iloc[0] > 1e14 else "ms"
    df.index = pd.to_datetime(df["open_time"], unit=unit)
    return df[["open", "high", "low", "close", "volume"]].astype(float)


def features(b):
    c = b["close"]
    return pd.DataFrame({"ret_1": c.pct_change(), "sma_96": c.rolling(96).mean()}, index=b.index)


def features_with_bug(b):
    out = features(b)
    out["h4_close"] = misaligned_htf(b)["feature"]
    return out


if __name__ == "__main__":
    data = load(sys.argv[1] if len(sys.argv) > 1 else "2024-01")
    print(f"{len(data)} bars {data.index.min()} -> {data.index.max()}")
    for name, fn in (("clean", features), ("with unsealed 4h merge", features_with_bug)):
        print(f"--- {name}")
        print(probe(fn, data).summary())
