# causality-probe

**Detect look-ahead bias in backtest features, with positive controls.**
[Versión en español](README.es.md)

A feature is causal at time *t* only if it depends on data available up to *t*:

```
build(full_history).loc[t]  ==  build(history_up_to_t).iloc[-1]
```

`causality-probe` checks this equivalence column by column at random points and at every day,
week and month boundary. Boundaries are where resampling bugs live. You do not need to read the
feature code: any difference is look-ahead by definition.

A probe that finds nothing proves nothing unless it can catch known leaks. Before giving a clean
verdict, `probe()` therefore runs 11 controls:
- **8 known leaks it must detect;**
- **3 sealed twins it must leave clean.**

If any control misbehaves, the verdict is `INVALID_CONTROL_FAILED`.

## Why

In the author's own research, a mean-reversion rule passed walk-forward, red-team review and a
sealed holdout of 2,768 trades: +0.672 R per trade, profit factor 2.23, p ≈ 0.002. It was traded
on an evaluation account.

The edge was manufactured by one line. A daily close was computed with `resample('1D').last()`,
which labels each day at its **start**, and was then forward-filled into 15-minute bars. At 03:00
the model already knew the 23:45 close. Once the leak was fixed, the same rule lost −4,380 R over
22 years.

Every validation step had consumed the same contaminated feature. A truncation probe catches this
in seconds. See [docs/mechanisms.md](docs/mechanisms.md).

## Install

```bash
pip install git+https://github.com/<user>/causality-probe
```

## Use

```python
import numpy as np
import pandas as pd
from causality_probe import probe

def features(bars: pd.DataFrame) -> pd.DataFrame:      # same index in, same index out
    c = bars["close"]
    return pd.DataFrame({"ret_1": np.log(c).diff(), "sma_20": c.rolling(20).mean()}, index=bars.index)

report = probe(features, bars)        # bars: OHLC DataFrame with a DatetimeIndex, >= 200 rows
print(report.summary())
assert report.clean
```

Command line:

```bash
causality-probe controls                                   # show the 11 controls
causality-probe probe --module my_pkg.features --function build --data prices.csv
causality-probe scan path/to/code                          # static map of risky patterns
```

The static scanner is the **map** and the probe is the **proof**. A flagged line can be correct:
for example, a negative shift used only to build the target label.

## Controls

| Control | Must | What it reproduces |
|---|---|---|
| `shift_minus_1` | leak | next bar's return used as a feature |
| `final_statistic` | leak | last value of the sample broadcast to every row |
| `global_zscore` | leak | normalisation with full-sample mean and standard deviation |
| `global_quantile` | leak | threshold from a full-sample quantile |
| `bfill` | leak | backward fill |
| `centered_rolling` | leak | rolling window with `center=True` |
| `daily_resample_ffill` | leak | daily aggregate labelled at the day start and forward-filled intraday |
| `misaligned_htf` | leak | higher-timeframe bars on another clock (for example a broker server offset), merged before they close |
| `daily_resample_sealed` | clean | the same daily aggregate, available only on day D+1 |
| `misaligned_htf_sealed` | clean | the same higher-timeframe bars stamped at their close |
| `rolling_mean` | clean | a plain trailing mean |

The intraday controls run on synthetic 15-minute bars. Price rises inside each day, so the daily
close is the daily high and cannot hide. On daily bars, `resample('1D').last()` is the identity:
a control there could never fail.

## What it does not prove

Independence from the future is necessary, not sufficient. The probe does not detect:
- **Design look-ahead**: choosing assets, periods or parameters by their outcome.
- **Survivorship bias.**
- **Execution artefacts**, such as spreads at the rollover hour that bid-only candles hide. That one needs tick-level bid/ask checks.

## Tests

```bash
pip install -e ".[test]" && pytest
```

## Cite

See [CITATION.cff](CITATION.cff). Each release is archived on Zenodo with a DOI.

## License

MIT. Educational and research software; not investment advice.
