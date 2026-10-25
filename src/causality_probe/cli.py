"""Command line: `causality-probe probe ...`, `causality-probe scan ...`, `causality-probe controls`."""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import asdict

import pandas as pd

from .controls import run_positive_controls
from .probe import VERDICT_CLEAN, probe
from .scanner import scan_code


def _load(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    for c in ("time", "timestamp", "date", "datetime", "open_time"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
            df = df.set_index(c)
            break
    return df.sort_index()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="causality-probe")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe", help="run the truncation probe on a feature builder")
    p.add_argument("--module", required=True, help="importable module containing the builder")
    p.add_argument("--function", required=True, help="builder: DataFrame -> DataFrame (same index)")
    p.add_argument("--data", required=True, help="CSV or Parquet price panel")
    p.add_argument("--points", type=int, default=120)
    p.add_argument("--json", default=None)
    s = sub.add_parser("scan", help="static scan of Python code for look-ahead patterns")
    s.add_argument("path")
    s.add_argument("--json", default=None)
    sub.add_parser("controls", help="run only the positive/negative controls")
    a = ap.parse_args(argv)

    if a.cmd == "controls":
        res = run_positive_controls()
        for k, v in res.items():
            print(f"{k:24s} expected={v['expected']:5s} differing_points={v['points_differing']:3d} "
                  f"{'OK' if v['ok'] else 'PROBE FAILURE'}")
        return 0 if all(v["ok"] for v in res.values()) else 1
    if a.cmd == "scan":
        h = scan_code(a.path)
        for x in h:
            print(f"[{x['risk']}] {x['file']}:{x['line']} {x['pattern']}: {x['code']}\n    check: {x['check']}")
        print(f"{len(h)} findings")
        if a.json:
            json.dump(h, open(a.json, "w", encoding="utf-8"), indent=1)
        return 1 if any(x["risk"] == "HIGH" for x in h) else 0
    sys.path.insert(0, ".")
    fn = getattr(importlib.import_module(a.module), a.function)
    rep = probe(fn, _load(a.data), n_points=a.points)
    print(rep.summary())
    if a.json:
        json.dump(asdict(rep), open(a.json, "w", encoding="utf-8"), indent=1, default=str)
    return 0 if rep.verdict == VERDICT_CLEAN else 1


if __name__ == "__main__":
    raise SystemExit(main())
