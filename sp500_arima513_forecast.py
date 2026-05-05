"""
S&P 500 — ARIMA(5,1,3) Walk-Forward Forecast
=====================================================================
Single-model version: fits ARIMA(5,1,3), runs walk-forward dynamic
forecasting at horizons 1, 5, 10, computes metrics, and saves
predictions to ./predictions/arima_fc{h}.csv for use by the
significance testing script.

  - Data  : yfinance ^GSPC, 2017-01-01 to 2025-12-31
  - Series: Return_1d = pct_change clipped ±0.5
  - Gate  : top 70% by |pred| / rolling_std(20)
"""

import os, warnings, time
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox

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
ARIMA_ORDER       = (5, 1, 3)

# Diagnostics
LB_LAGS      = 20
SIG_ALPHA    = 0.05

# Walk-forward parameters
REFIT_EVERY  = 20
EXPANDING    = True
ROLLING_SIZE = None

# Confidence gate
KEEP_TOP_PCT = 0.70
GATE_ROLL    = 20

OUT_DIR  = "arima_513_outputs"
PRED_DIR = "predictions"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)

print("=" * 72)
print(f"  S&P 500 — ARIMA{ARIMA_ORDER} Walk-Forward Forecast")
print("=" * 72)


# ─────────────────────────────────────────────────────────────
# 1. DATA
# ─────────────────────────────────────────────────────────────
print(f"\n[1/5] Downloading {TICKER} from yfinance...")
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


# ─────────────────────────────────────────────────────────────
# 2. TRAINING-FIT DIAGNOSTICS
# ─────────────────────────────────────────────────────────────
def fit_and_diagnose(train_returns, order,
                     lb_lags=LB_LAGS, sig_alpha=SIG_ALPHA):
    """Fit ARIMA(order) on train_returns, return diagnostics dict."""
    try:
        m = ARIMA(train_returns, order=order,
                  enforce_stationarity=False,
                  enforce_invertibility=False).fit(
                      method_kwargs={"maxiter": 1000})
    except Exception as e:
        return dict(order=order, aic=np.nan, bic=np.nan,
                    lb_p=np.nan, sig_frac=np.nan, n_coef=0,
                    status=f"fit-fail: {type(e).__name__}")

    try:
        lb = acorr_ljungbox(m.resid, lags=[lb_lags], return_df=True)
        lb_p = float(lb["lb_pvalue"].iloc[0])
    except Exception:
        lb_p = np.nan

    names = list(m.param_names)
    pvals_arr = np.asarray(m.pvalues)
    keep_mask = np.array([nm != "sigma2" for nm in names])
    pvals = pvals_arr[keep_mask]
    n_coef = int(pvals.size)
    frac_sig = float((pvals < sig_alpha).mean()) if n_coef > 0 else 0.0

    return dict(order=order, aic=float(m.aic), bic=float(m.bic),
                lb_p=lb_p, sig_frac=frac_sig, n_coef=n_coef,
                status="fit-ok")


print(f"\n[2/5] Training-fit diagnostics for ARIMA{ARIMA_ORDER}")
train_returns = train_df["Return_1d"].values
diag = fit_and_diagnose(train_returns, ARIMA_ORDER)
if diag["status"] == "fit-ok":
    print(f"       BIC={diag['bic']:.2f}  AIC={diag['aic']:.2f}  "
          f"LB p={diag['lb_p']:.3f}  sig_frac={diag['sig_frac']:.2f}  "
          f"(n_coef={diag['n_coef']})")
else:
    raise RuntimeError(f"Training fit failed: {diag['status']}")


# ─────────────────────────────────────────────────────────────
# 3. WALK-FORWARD DYNAMIC FORECASTS
# ─────────────────────────────────────────────────────────────
def run_arima(h, arima_order):
    full_ret = df["Return_1d"].values
    full_tgt = df[f"Target_{h}d"].values
    n_valid  = len(test_df) - h
    n_refits = (n_valid // REFIT_EVERY) + 1
    print(f"  h={h}: order {arima_order}, {n_valid} forecasts, "
          f"refit every {REFIT_EVERY} days (~{n_refits} refits)")

    t0 = time.time()
    preds, actuals = [], []
    current_fit = None
    current_fit_end = -1
    refit_count = 0

    for i in range(n_valid):
        t = n_train + i

        # Refit if needed
        if (current_fit is None) or (i % REFIT_EVERY == 0):
            if EXPANDING:
                train_slice = full_ret[:t]
            else:
                start_idx = max(0, t - (ROLLING_SIZE or n_train))
                train_slice = full_ret[start_idx:t]

            try:
                current_fit = ARIMA(
                    train_slice, order=arima_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                ).fit(method_kwargs={"maxiter": 1000})
                current_fit_end = t - 1
                refit_count += 1
            except Exception:
                if current_fit is None:
                    raise
        else:
            if t - 1 > current_fit_end:
                current_fit = current_fit.apply(
                    full_ret[:t], refit=False)
                current_fit_end = t - 1

        try:
            fc = current_fit.get_forecast(steps=h).predicted_mean
            fc = np.asarray(fc)
            cum = np.prod(1.0 + fc) - 1.0
        except Exception:
            cum = 0.0

        preds.append(np.clip(cum, -0.5, 0.5))
        actuals.append(full_tgt[t])

    print(f"       {refit_count} refits, time={time.time()-t0:.1f}s")
    return np.array(preds), np.array(actuals)


# ─────────────────────────────────────────────────────────────
# 4. 70% CONFIDENCE GATE + METRICS
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
    rmse   = np.sqrt(mean_squared_error(actuals, preds))
    da     = np.mean(np.sign(preds) == np.sign(actuals)) * 100
    gated, active = adaptive_gate(preds)
    if active.sum() > 0:
        da_g = np.mean(np.sign(gated[active]) == np.sign(actuals[active])) * 100
    else:
        da_g = da
    rho, _ = spearmanr(preds, actuals)
    return dict(
        n=n, rmse=rmse, da=da, da_gated=da_g,
        coverage=active.mean() * 100, spearman=rho,
        preds=preds, actuals=actuals, gated=gated, active=active,
    )


# ─────────────────────────────────────────────────────────────
# 5. RUN + REPORT
# ─────────────────────────────────────────────────────────────
print(f"\n[3/5] Walk-forward dynamic forecasts for ARIMA{ARIMA_ORDER}")
results = {}
for h in FORECAST_HORIZONS:
    p, a = run_arima(h, ARIMA_ORDER)
    results[h] = compute_metrics(p, a)

print(f"\n[4/5] Results for ARIMA{ARIMA_ORDER}")
print("  " + "-" * 82)
print(f"  {'Horizon':<10}{'N':>6}{'RMSE':>12}{'DA':>10}{'DA-gated':>12}"
      f"{'Coverage':>12}{'Spearman':>12}")
print("  " + "-" * 82)
for h in FORECAST_HORIZONS:
    r = results[h]
    print(f"  {h}-step{'':<4}{r['n']:>6d}{r['rmse']:>12.5f}"
          f"{r['da']:>9.1f}%{r['da_gated']:>11.1f}%"
          f"{r['coverage']:>11.0f}%{r['spearman']:>12.3f}")
print("  " + "-" * 82)


# ─────────────────────────────────────────────────────────────
# 6. SAVE FORECAST CSVs (full version with gate info)
# ─────────────────────────────────────────────────────────────
print(f"\n[5/5] Saving forecasts → {OUT_DIR}/")
for h in FORECAST_HORIZONS:
    r = results[h]
    dates = test_df.index[:r["n"]]
    out = pd.DataFrame({
        "date":   dates,
        "pred":   r["preds"],
        "actual": r["actuals"],
        "gated":  r["gated"],
        "active": r["active"].astype(int),
    })
    fname = os.path.join(OUT_DIR, f"arima_513_fc{h}.csv")
    out.to_csv(fname, index=False)
    print(f"  {fname}  ({len(out)} rows)")


# ─────────────────────────────────────────────────────────────
# 7. SAVE PREDICTIONS FOR SIGNIFICANCE TESTING
# ─────────────────────────────────────────────────────────────
print(f"\n  Saving predictions for significance testing → {PRED_DIR}/")
for h in FORECAST_HORIZONS:
    r = results[h]
    pd.DataFrame({"pred": r["preds"], "actual": r["actuals"]}).to_csv(
        os.path.join(PRED_DIR, f"arima_fc{h}.csv"), index=False)
    print(f"  {PRED_DIR}/arima_fc{h}.csv  ({r['n']} rows)")

print(f"\n  Gate: top {int(KEEP_TOP_PCT*100)}% by |pred| / rolling_std({GATE_ROLL})")


# ─────────────────────────────────────────────────────────────
# 8. PLOTS — Predicted vs Actual time series
# ─────────────────────────────────────────────────────────────
print(f"\n  Generating forecast plots → {OUT_DIR}/")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

fig, axes = plt.subplots(len(FORECAST_HORIZONS), 1, figsize=(12, 9), sharex=True)
if len(FORECAST_HORIZONS) == 1:
    axes = [axes]

for ax, h in zip(axes, FORECAST_HORIZONS):
    r = results[h]
    dates = test_df.index[:r["n"]]

    # Actual returns: thin grey line in background
    ax.plot(dates, r["actuals"] * 100, color="grey", linewidth=0.8,
            alpha=0.7, label="Actual return")

    # Predicted returns: solid blue line
    ax.plot(dates, r["preds"] * 100, color="steelblue", linewidth=1.2,
            label=f"ARIMA{ARIMA_ORDER} forecast")

    # Zero line for reference
    ax.axhline(0, color="black", linewidth=0.4, linestyle="--", alpha=0.5)

    # Title and metrics annotation
    ax.set_title(f"{h}-step ahead   "
                 f"(RMSE={r['rmse']:.4f}, DA={r['da']:.1f}%, "
                 f"DA-gated={r['da_gated']:.1f}%, ρ={r['spearman']:.3f})",
                 fontsize=11)
    ax.set_ylabel("Return (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

# Format x-axis dates only on bottom plot
axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha="right")

fig.suptitle(f"ARIMA{ARIMA_ORDER} — Forecast vs Actual Returns",
             fontsize=13, fontweight="bold", y=0.995)
plt.tight_layout()

plot_path = os.path.join(OUT_DIR, "arima_513_forecasts.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  {plot_path}")


print("\n[Done]")
