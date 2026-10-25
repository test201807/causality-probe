import numpy as np
import pandas as pd
import pytest

from causality_probe import CONTROLS, probe, run_positive_controls, scan_code
from causality_probe.controls import synthetic_m15, synthetic_random_walk


def clean_features(b):
    c = b["close"]
    return pd.DataFrame({"ret_1": np.log(c).diff(), "sma_20": c.rolling(20).mean(),
                         "z_100": (c - c.rolling(100).mean()) / c.rolling(100).std()}, index=b.index)


def leaky_features(b):
    out = clean_features(b)
    out["next_ret"] = np.log(b["close"]).diff().shift(-1)
    return out


def test_controls_all_behave():
    res = run_positive_controls()
    assert set(res) == set(CONTROLS)
    failed = {k: v for k, v in res.items() if not v["ok"]}
    assert not failed, failed


def test_clean_builder_passes():
    rep = probe(clean_features, synthetic_random_walk())
    assert rep.verdict == "NO_LEAK", rep.summary()
    assert rep.leaking_columns == []


def test_leaky_column_is_identified():
    rep = probe(leaky_features, synthetic_random_walk(), run_controls=False)
    assert rep.verdict == "LEAK_DETECTED"
    assert rep.leaking_columns == ["next_ret"]
    assert rep.leak_rate_pct["next_ret"] == 100.0


@pytest.mark.parametrize("name", [k for k, v in CONTROLS.items() if v[1]])
def test_each_known_leak_is_flagged(name):
    fn, _, ds = CONTROLS[name]
    data = synthetic_m15() if ds == "m15" else synthetic_random_walk()
    rep = probe(fn, data, warmup=300, run_controls=False)
    assert rep.verdict == "LEAK_DETECTED", name


def test_daily_resample_bug_detected_only_intraday():
    # on daily bars resample('1D').last() is the identity: the control would be blind there
    fn = CONTROLS["daily_resample_ffill"][0]
    daily = synthetic_random_walk(800).resample("1D").last().dropna()
    daily = pd.concat([daily] * 1)
    if len(daily) >= 200:
        assert probe(fn, daily, run_controls=False).verdict == "NO_LEAK"
    assert probe(fn, synthetic_m15(), warmup=300, run_controls=False).verdict == "LEAK_DETECTED"


def test_too_short_data_rejected():
    with pytest.raises(ValueError):
        probe(clean_features, synthetic_random_walk(100))


def test_scanner_flags_patterns(tmp_path):
    f = tmp_path / "feat.py"
    f.write_text("x = s.resample('1D').last()\ny = s.bfill()\nz = s.rolling(5).mean()\n", encoding="utf-8")
    keys = {h["pattern"] for h in scan_code(tmp_path)}
    assert {"resample_unsealed", "backward_fill"} <= keys
    assert all(h["line"] != 3 for h in scan_code(tmp_path))
