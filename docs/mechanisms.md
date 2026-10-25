# Three look-ahead mechanisms that survive ordinary validation

Ordinary validation (walk-forward, hold-out, cost stress, multi-engine parity) cannot see
these mechanisms. Every stage consumes the same contaminated feature or the same contaminated
price.

## 1. Downsampling labelled at the period start

```python
daily = close.resample("1D").last()                  # value of the whole day, labelled at 00:00
feature = daily.reindex(intraday.index, method="ffill")
```

**What goes wrong.** At 03:00 the feature already holds the 23:45 close. A `shift(1)` on the
intraday frame does not fix it: it moves the value by one intraday bar, not by one day.

**Fix.** Stamp the aggregate at its close before bringing it down to the faster timeframe:

```python
daily.index = daily.index + pd.Timedelta(days=1)
```

Controls: `daily_resample_ffill` (leak) and `daily_resample_sealed` (clean).

## 2. Higher-timeframe bars on a different clock

**What goes wrong.** Broker servers often run at a fixed offset from the research engine's clock.
Suppose an H4 bar on the server closes after the engine's daily anchor but before the engine's
next bar closes. That bar gets assigned to the earlier day, even though it becomes known only
afterwards.

**Real case.** In the author's audit of a multi-asset event-detector pipeline:
- 116 of 145 instruments were affected;
- 772,890 of 25.7 million events were assigned to a day whose anchor had already passed.

**Fix.** Stamp every higher-timeframe bar at its close time and merge with
`merge_asof(direction="backward")`.

Controls: `misaligned_htf` (leak) and `misaligned_htf_sealed` (clean).

## 3. Bid-only candles at the rollover hour (not detectable by this probe)

**What goes wrong.** Many retail CFD feeds store bid candles. In the minute after the daily
rollover, spreads widen sharply and then close. A backtest that enters on bid prices in that
minute "earns" the spread closing.

**Real case.** In the author's tick-level check:

| Price used | Result per trade |
|---|---|
| Bid backtest | +0.209 R |
| Mid price | +0.019 R |
| Real entry at the ask | −0.228 R |

**Why the probe misses it.** It is not a feature leak but an execution artefact. Only tick-level
bid/ask data for the entry minute reveals it.

## What none of this catches

- Selecting assets, periods or parameters by outcome.
- Survivorship bias.
- Overfitting through multiple testing: count every trial and correct for it, for example with
  Holm or Bonferroni over the full research load.
