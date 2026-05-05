"""
S&P 500 — Random Walk Benchmark with 70% Confidence Gate
============================================================
The random walk model predicts zero return at every horizon:
  E[r_{t+h}] = 0

This is the standard benchmark in financial forecasting — if a model
can't beat the random walk, it has no predictive value.

Same pipeline as Transformer, SVR, and ARMA(5,4):
  - Data     : yfinance ^GSPC, 2017-01-01 to 2025-12-31
  - Targets  : Target_hd = (Close.shift(-h)/Close - 1) clipped ±0.5
  - Gate     : top 70% by |pred| / rolling_std(20)
  - Metrics  : RMSE, DA, DA-gated, Spearman ρ
"""

import os, warnings
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
TICKER       = "^GSPC"
TRAIN_START  = "2017-01-01"
TRAIN_END    = "2023-12-31"
TEST_START   = "2024-01-01"
TEST_END     = "2025-12-31"

FORECAST_HORIZONS = [1, 5, 10]
KEEP_TOP_PCT = 0.70
GATE_ROLL    = 20

print("=" * 72)
print("  S&P 500 — Random Walk Benchmark")
print("=" * 72)


# ─────────────────────────────────────────────────────────────
# 1. DATA
# ─────────────────────────────────────────────────────────────
print(f"\n[1/3] Downloading {TICKER} from yfinance...")
raw = yf.download(TICKER, start=TRAIN_START, end=TEST_END,
                  progress=False, auto_adjust=False)
if isinstance(raw.columns, pd.MultiIndex):
    raw.columns = raw.columns.get_level_values(0)
if len(raw) == 0:
    raise RuntimeError(f"yfinance returned no data for {TICKER}.")
print(f"  Downloaded {len(raw)} rows: "
      f"{raw.index[0].date()} → {raw.index[-1].date()}")

df = raw[["Close"]].copy()
df["Return_1d"] = df["Close"].pct_change().clip(-0.5, 0.5)
for h in FORECAST_HORIZONS:
    df[f"Target_{h}d"] = (df["Close"].shift(-h) / df["Close"] - 1).clip(-0.5, 0.5)

df = df.dropna(subset=["Return_1d"]).copy()

train_df = df.loc[:TRAIN_END].copy()
test_df  = df.loc[TEST_START:].copy()
n_train  = len(train_df)
n_test   = len(test_df)
print(f"  Train: {n_train} days  |  Test: {n_test} days")

test_dates = test_df.index


# ─────────────────────────────────────────────────────────────
# 2. RANDOM WALK — predict zero for every horizon
# ─────────────────────────────────────────────────────────────
def run_random_walk(h):
    n_valid = len(test_df) - h
    actuals = df[f"Target_{h}d"].values[n_train:n_train + n_valid]
    preds   = np.zeros(n_valid)  # random walk: E[return] = 0
    return preds, actuals


# ─────────────────────────────────────────────────────────────
# 3. METRICS
# ─────────────────────────────────────────────────────────────
def adaptive_gate(preds, keep_top_pct=KEEP_TOP_PCT, roll=GATE_ROLL):
    s  = pd.Series(preds)
    rs = s.rolling(roll, min_periods=1).std().fillna(s.std()).values
    rs = np.where(rs < 1e-8, 1e-8, rs)
    conf = np.abs(preds) / rs
    threshold = np.percentile(conf, (1 - keep_top_pct) * 100)
    active = conf >= threshold
    gated = np.where(active, preds, 0.0)
    return gated, active


def compute_metrics(preds, actuals):
    n = min(len(preds), len(actuals))
    preds, actuals = preds[:n], actuals[:n]
    rmse = np.sqrt(mean_squared_error(actuals, preds))

    # DA: for a zero-prediction model, sign(0) = 0 which won't match
    # sign(actual) for any nonzero actual. So we compute DA as the
    # fraction of days the market went up (since the RW prediction of
    # "no change" is effectively neutral). By convention, DA for a
    # zero-prediction benchmark = fraction of days actual > 0, because
    # a naive investor holding the market gets those days "right".
    # However, to be consistent with np.sign comparison used in other
    # models: np.sign(0) = 0, which matches neither +1 nor -1.
    # We report both for transparency.
    da_strict = np.mean(np.sign(preds) == np.sign(actuals)) * 100
    pct_up    = np.mean(actuals > 0) * 100

    # Spearman: all predictions identical → undefined
    try:
        rho, _ = spearmanr(preds, actuals)
        if np.isnan(rho):
            rho = 0.0
    except Exception:
        rho = 0.0

    # Gate: all predictions are zero, so gate is meaningless
    # Report DA-gated as N/A
    da_gated = np.nan
    coverage = 0.0

    return dict(
        n=n, rmse=rmse, da=da_strict, pct_up=pct_up,
        da_gated=da_gated, coverage=coverage, spearman=rho,
        preds=preds, actuals=actuals,
    )


# ─────────────────────────────────────────────────────────────
# 4. RUN + REPORT
# ─────────────────────────────────────────────────────────────
print("\n[2/3] Running random walk benchmark...")
results = {}
for h in FORECAST_HORIZONS:
    p, a = run_random_walk(h)
    results[h] = compute_metrics(p, a)

print("\n[3/3] Results")
print("=" * 84)
print(f"  {'Horizon':<10}{'N':>6}{'RMSE':>12}{'DA':>10}{'% Up':>10}"
      f"{'DA-Gated':>12}{'Spearman':>12}")
print("  " + "-" * 82)
for h in FORECAST_HORIZONS:
    r = results[h]
    dag_str = f"{r['da_gated']:.1f}%" if not np.isnan(r['da_gated']) else "N/A"
    print(f"  {h}-step{'':<4}{r['n']:>6d}{r['rmse']:>12.5f}"
          f"{r['da']:>9.1f}%{r['pct_up']:>9.1f}%"
          f"{dag_str:>12s}{r['spearman']:>12.3f}")
print("  " + "-" * 82)
print(f"\n  Notes:")
print(f"    Random walk predicts 0 return at every horizon.")
print(f"    DA (strict) = np.sign(0) vs np.sign(actual) → always 0%")
print(f"      because sign(0)=0 matches neither positive nor negative days.")
print(f"    % Up = fraction of test days with positive actual returns.")
print(f"      This is the DA any model achieves by always predicting 'up'.")
print(f"    RMSE = std(actuals), since predictions are all zero.")
print(f"    DA-Gated = N/A (all predictions identical, gate has nothing to filter).")
print(f"    Spearman = 0 (no rank variation in predictions).")
print(f"\n  The RMSE column is the key benchmark: any model with RMSE")
print(f"  below these values is adding predictive value beyond the random walk.")

print("\n[Done]")

# ─────────────────────────────────────────────────────────────
# SAVE PREDICTIONS FOR SIGNIFICANCE TESTING
# ─────────────────────────────────────────────────────────────
import os
out_dir = "predictions"
os.makedirs(out_dir, exist_ok=True)
for h in FORECAST_HORIZONS:
    r = results[h]
    pd.DataFrame({"pred": r["preds"], "actual": r["actuals"]}).to_csv(
        os.path.join(out_dir, f"rw_fc{h}.csv"), index=False)
    print(f"  Saved {out_dir}/rw_fc{h}.csv")