"""
S&P 500 — ARIMAX Walk-Forward Forecasting (Box-Jenkins Specification)
======================================================================
Direct conversion of the supervisor's ARIMA baseline into ARIMAX, with
proper Box-Jenkins order selection:

  1. Significance: all AR (φ) and MA (θ) coefficients must have p < 0.05
  2. Residual whiteness: Ljung-Box at lag 40 must pass (p > 0.05)
  3. Among models passing both, pick the one with lowest AIC

If no model passes both criteria, the script falls back in this order:
  (a) Allow one insignificant AR or MA coefficient
  (b) Best AIC overall, with a printed warning showing what failed

Pipeline (otherwise identical to supervisor's ARIMA):
  - Data     : yfinance ^GSPC + ^VIX, 2017-01-01 to 2025-12-31
  - Series   : daily simple returns (Return_1d) — endogenous
  - Targets  : Target_hd = (Close.shift(-h) / Close - 1) clipped ±0.5
  - Order    : Box-Jenkins (above), grid (p,q) ∈ 0..5, d=0
  - Walk-fwd : refit every 20 days, expanding window
  - h-step   : forecast(steps=h, exog=X_held_flat), then compound:
               cum = prod(1 + fc) - 1
  - Gate     : top 70% by |pred| / rolling_std(20)
  - Metrics  : RMSE, DA, DA-gated, coverage, Spearman ρ

Exogenous variables (per horizon):
  1-step  : Return_1d, Return_5d, RSI_14, MACD_Hist,
            VIX, VIX_Change, Volume_Ratio, High_Low_Spread
  5/10    : Return_5d, Return_20d, Close_to_MA_20, Close_to_MA_50,
            Volatility_20d, Volatility_ratio, RSI_14, MACD_Hist,
            VIX, VIX_Log
"""

import os, warnings, time
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

# Order selection — Box-Jenkins
PQ_GRID         = [(p, q) for p in range(0, 6) for q in range(0, 6)]  # 36 candidates
D_ORDER         = 0      # returns are already stationary, no differencing
COEF_PVAL_MAX   = 0.05   # AR/MA coefficients must have p < this
LJUNGBOX_LAG    = 40
LJUNGBOX_ALPHA  = 0.05   # Ljung-Box p-value must exceed this

# Walk-forward
ARIMAX_REFIT  = 20
KEEP_TOP_PCT  = 0.70
GATE_ROLL     = 20

# Per-horizon exogenous variables
EXOG_1STEP = [
    "Return_1d", "Return_5d",
    "RSI_14", "MACD_Hist",
    "VIX", "VIX_Change",
    "Volume_Ratio", "High_Low_Spread",
]
EXOG_5STEP = [
    "Return_5d", "Return_20d",
    "Close_to_MA_20", "Close_to_MA_50",
    "Volatility_20d", "Volatility_ratio",
    "RSI_14", "MACD_Hist",
    "VIX", "VIX_Log",
]
EXOG_10STEP = EXOG_5STEP[:]
EXOG_BY_HORIZON = {1: EXOG_1STEP, 5: EXOG_5STEP, 10: EXOG_10STEP}

OUT_DIR = "arimax_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# Plot colours
ACTUAL_COLOR    = "#4C72B0"
PREDICTED_COLOR = "#DD4444"
H_COLORS        = {1: "#4C72B0", 5: "#DD8844", 10: "#27AE60"}

print("=" * 72)
print("  S&P 500 — ARIMAX Walk-Forward (Box-Jenkins Specification)")
print(f"  Order grid: (p,q) in 0..5, d={D_ORDER}")
print(f"  Acceptance: AR/MA p<{COEF_PVAL_MAX} AND Ljung-Box({LJUNGBOX_LAG})>{LJUNGBOX_ALPHA}")
print(f"  Refit every {ARIMAX_REFIT} days, gate top {int(KEEP_TOP_PCT*100)}%")
print("=" * 72)


# ─────────────────────────────────────────────────────────────
# 1. DATA
# ─────────────────────────────────────────────────────────────
print(f"\n[1/5] Downloading data & engineering features...")

def safe_div(a, b, default=0.0):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(b) > 1e-8, a / b, default)


df = yf.download(TICKER, start=TRAIN_START, end=TEST_END,
                 progress=False, auto_adjust=False)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
if len(df) == 0:
    raise RuntimeError(f"yfinance returned no data for {TICKER}.")
print(f"  Downloaded {len(df)} rows")

try:
    vix = yf.download("^VIX", start=TRAIN_START, end=TEST_END, progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    df["VIX"] = vix["Close"]
    print(f"  VIX: {df['VIX'].notna().sum()} rows")
except Exception as e:
    df["VIX"] = 15.0
    print(f"  VIX fallback: constant 15 ({e})")


# ─────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
def add_features(df):
    df = df.copy()

    # Returns
    df["Return_1d"]  = df["Close"].pct_change().clip(-0.5, 0.5)
    df["Return_5d"]  = df["Close"].pct_change(5).clip(-0.5, 0.5)
    df["Return_20d"] = df["Close"].pct_change(20).clip(-0.5, 0.5)

    # Moving averages
    for w in [20, 50]:
        ma = df["Close"].rolling(w).mean()
        df[f"MA_{w}"] = ma
        df[f"Close_to_MA_{w}"] = np.clip(
            safe_div((df["Close"] - ma).values, ma.values) * 100, -50, 50)

    # Volatility
    df["Volatility_20d"] = df["Return_1d"].rolling(20).std() * np.sqrt(252)
    df["Volatility_50d"] = df["Return_1d"].rolling(50).std() * np.sqrt(252)
    df["Volatility_ratio"] = np.clip(
        safe_div(df["Volatility_20d"].values,
                 df["Volatility_50d"].values, 1.0), 0.1, 10)

    # RSI-14
    delta = df["Close"].diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["RSI_14"] = np.clip(
        100 - (100 / (1 + safe_div(gain.values, loss.values, 1.0))), 0, 100)

    # MACD histogram
    e1 = df["Close"].ewm(span=12, adjust=False).mean()
    e2 = df["Close"].ewm(span=26, adjust=False).mean()
    macd        = (e1 - e2).clip(-100, 100)
    macd_signal = macd.ewm(span=9, adjust=False).mean().clip(-100, 100)
    df["MACD_Hist"] = (macd - macd_signal).clip(-50, 50)

    # Volume
    if "Volume" in df.columns and df["Volume"].notna().any():
        vol_ma20 = df["Volume"].rolling(20).mean()
        df["Volume_Ratio"] = np.clip(
            safe_div(df["Volume"].values, vol_ma20.values, 1.0), 0.1, 10)
    else:
        df["Volume_Ratio"] = 1.0

    # High-Low spread
    if "High" in df.columns and "Low" in df.columns:
        df["High_Low_Spread"] = ((df["High"] - df["Low"]) / df["Close"] * 100).clip(0, 20)
    else:
        df["High_Low_Spread"] = 0.0

    # VIX transforms
    df["VIX_Change"] = df["VIX"].pct_change().clip(-0.3, 0.3)
    df["VIX_Log"]    = np.log1p(df["VIX"].clip(lower=5))

    # Targets — cumulative simple returns
    for h in FORECAST_HORIZONS:
        df[f"Target_{h}d"] = (df["Close"].shift(-h) / df["Close"] - 1).clip(-0.5, 0.5)

    return df.replace([np.inf, -np.inf], np.nan).dropna()


df = add_features(df)
print(f"  Rows after feature engineering: {len(df)}")

train_df = df.loc[:TRAIN_END].copy()
test_df  = df.loc[TEST_START:].copy()
n_train  = len(train_df)
n_test   = len(test_df)
print(f"  Train: {n_train} days  |  Test: {n_test} days")

test_dates  = test_df.index
test_prices = test_df["Close"].values


# ─────────────────────────────────────────────────────────────
# 3. BOX-JENKINS ORDER SELECTION
# ─────────────────────────────────────────────────────────────
def fit_arimax(y, X, order):
    """Fit ARIMAX(p, d, q). Returns fitted results or None."""
    try:
        return ARIMA(
            y, exog=X, order=(order[0], D_ORDER, order[1]),
            enforce_stationarity=False, enforce_invertibility=False,
        ).fit(method_kwargs={"maxiter": 300})
    except Exception:
        return None


def diagnose_fit(res, p, q):
    """
    Inspect a fitted ARIMAX. Returns dict with:
      - aic, bic
      - n_ar_insig: count of AR coefficients with p >= COEF_PVAL_MAX
      - n_ma_insig: count of MA coefficients with p >= COEF_PVAL_MAX
      - lb_pvalue : Ljung-Box at LJUNGBOX_LAG
      - all_arma_sig : True iff every AR and MA coef is significant
      - lb_pass : True iff lb_pvalue > LJUNGBOX_ALPHA
    """
    out = {"aic": float(res.aic), "bic": float(res.bic),
           "n_ar_insig": 0, "n_ma_insig": 0,
           "lb_pvalue": 0.0, "all_arma_sig": False, "lb_pass": False}

    # Coefficient p-values  (statsmodels names AR terms 'ar.L1'..'ar.Lp', MA 'ma.L1'..'ma.Lq')
    try:
        pvals = res.pvalues
        for i in range(1, p + 1):
            name = f"ar.L{i}"
            if name in pvals.index and pvals[name] >= COEF_PVAL_MAX:
                out["n_ar_insig"] += 1
        for i in range(1, q + 1):
            name = f"ma.L{i}"
            if name in pvals.index and pvals[name] >= COEF_PVAL_MAX:
                out["n_ma_insig"] += 1
    except Exception:
        # If we can't read pvals, treat as "all insignificant" so this model loses
        out["n_ar_insig"] = p
        out["n_ma_insig"] = q

    out["all_arma_sig"] = (out["n_ar_insig"] == 0 and out["n_ma_insig"] == 0)

    # Ljung-Box
    try:
        lb = acorr_ljungbox(res.resid, lags=[LJUNGBOX_LAG], return_df=True)
        out["lb_pvalue"] = float(lb["lb_pvalue"].iloc[0])
        out["lb_pass"]   = out["lb_pvalue"] > LJUNGBOX_ALPHA
    except Exception:
        pass

    return out


def select_order_box_jenkins(y, X, horizon):
    """
    Search the (p,q) grid. Acceptance criteria:
      Tier A : all AR/MA significant AND Ljung-Box passes
      Tier B : at most ONE AR or MA insignificant AND Ljung-Box passes
      Tier C : best AIC overall (warning printed)
    Among acceptable models in the highest available tier, pick lowest AIC.
    Returns (chosen_order, log_df, chosen_diag, tier).
    """
    print(f"\n  Selecting order for h={horizon}-step...")
    print(f"  Searching {len(PQ_GRID)} candidate (p,q) combinations...")

    log_rows = []
    for (p, q) in PQ_GRID:
        if p == 0 and q == 0:
            # Pure regression on exog with no ARMA structure - keep as candidate
            pass
        res = fit_arimax(y, X, (p, q))
        if res is None:
            continue
        diag = diagnose_fit(res, p, q)
        log_rows.append({
            "p": p, "q": q,
            "aic": diag["aic"], "bic": diag["bic"],
            "n_ar_insig": diag["n_ar_insig"],
            "n_ma_insig": diag["n_ma_insig"],
            "lb_pvalue": diag["lb_pvalue"],
            "all_arma_sig": diag["all_arma_sig"],
            "lb_pass": diag["lb_pass"],
        })

    if not log_rows:
        raise RuntimeError(f"No model could be fit for h={horizon}")

    log_df = pd.DataFrame(log_rows).sort_values("aic").reset_index(drop=True)

    # Tier A: all ARMA significant AND Ljung-Box passes
    tier_a = log_df[(log_df["all_arma_sig"]) & (log_df["lb_pass"])].copy()
    if len(tier_a) > 0:
        best  = tier_a.sort_values("aic").iloc[0]
        order = (int(best["p"]), int(best["q"]))
        print(f"  TIER A: {len(tier_a)} model(s) passed both criteria")
        print(f"  Top 5 by AIC among Tier A:")
        for _, row in tier_a.sort_values("aic").head(5).iterrows():
            print(f"    (p={int(row['p'])}, q={int(row['q'])})  "
                  f"AIC={row['aic']:.2f}  BIC={row['bic']:.2f}  "
                  f"LB_p={row['lb_pvalue']:.3f}")
        print(f"  CHOSEN: ({order[0]},{D_ORDER},{order[1]})  [Tier A]")
        return order, log_df, best.to_dict(), "A"

    # Tier B: relax — allow ONE insignificant ARMA coef, but Ljung-Box still must pass
    tier_b = log_df[(log_df["n_ar_insig"] + log_df["n_ma_insig"] <= 1)
                    & (log_df["lb_pass"])].copy()
    if len(tier_b) > 0:
        best  = tier_b.sort_values("aic").iloc[0]
        order = (int(best["p"]), int(best["q"]))
        print(f"  No model met Tier A (all ARMA significant + LB pass).")
        print(f"  TIER B: {len(tier_b)} model(s) with <=1 insignificant ARMA coef + LB pass")
        print(f"  CHOSEN: ({order[0]},{D_ORDER},{order[1]})  [Tier B]  "
              f"insig_AR={int(best['n_ar_insig'])}  insig_MA={int(best['n_ma_insig'])}")
        return order, log_df, best.to_dict(), "B"

    # Tier C: fall back to best AIC, warn loudly
    best  = log_df.sort_values("aic").iloc[0]
    order = (int(best["p"]), int(best["q"]))
    print(f"  WARNING: no model met Tier A or Tier B criteria.")
    print(f"  Falling back to best AIC: ({order[0]},{D_ORDER},{order[1]})  "
          f"AIC={best['aic']:.2f}  LB_p={best['lb_pvalue']:.3f}  "
          f"insig_AR={int(best['n_ar_insig'])}  insig_MA={int(best['n_ma_insig'])}")
    print(f"  Note this in your write-up: residual autocorrelation may persist.")
    return order, log_df, best.to_dict(), "C"


# ─────────────────────────────────────────────────────────────
# 4. WALK-FORWARD ARIMAX
# ─────────────────────────────────────────────────────────────
def run_arimax(h, order):
    """Walk-forward, refit every ARIMAX_REFIT days using the chosen order."""
    exog_cols = EXOG_BY_HORIZON[h]
    full_ret  = df["Return_1d"].values
    full_exog = df[exog_cols].values
    full_tgt  = df[f"Target_{h}d"].values
    n_test_h  = len(test_df) - h

    print(f"  h={h}: walk-forward with order ({order[0]},{D_ORDER},{order[1]}), "
          f"{len(exog_cols)} exog, {n_test_h} forecasts")

    preds, actuals = [], []
    fitted, refits, fallbacks = None, 0, 0
    t0 = time.time()

    for i in range(n_test_h):
        hist_y = full_ret[:n_train + i]
        hist_X = full_exog[:n_train + i]

        if i % ARIMAX_REFIT == 0:
            fitted_new = fit_arimax(hist_y, hist_X, order)
            if fitted_new is not None:
                fitted = fitted_new
                refits += 1
            else:
                # Last-ditch fallback to a simple order
                fitted_fb = fit_arimax(hist_y, hist_X, (1, 0))
                if fitted_fb is not None:
                    fitted = fitted_fb
                    fallbacks += 1

        try:
            x_now    = full_exog[n_train + i].reshape(1, -1)
            x_future = np.repeat(x_now, h, axis=0)
            fc       = fitted.forecast(steps=h, exog=x_future)
            cum      = np.prod(1 + np.asarray(fc)) - 1
        except Exception:
            cum = 0.0

        preds.append(np.clip(cum, -0.5, 0.5))
        actuals.append(full_tgt[n_train + i])

    elapsed = time.time() - t0
    print(f"       refits={refits}  fallbacks={fallbacks}  "
          f"time={elapsed:.1f}s")
    return np.array(preds), np.array(actuals)


# ─────────────────────────────────────────────────────────────
# 5. CONFIDENCE GATE + METRICS
# ─────────────────────────────────────────────────────────────
def adaptive_gate(preds, keep_top_pct=KEEP_TOP_PCT, roll=GATE_ROLL):
    s  = pd.Series(preds)
    rs = s.rolling(roll, min_periods=1).std().fillna(s.std()).values
    rs = np.where(rs < 1e-8, 1e-8, rs)
    conf      = np.abs(preds) / rs
    threshold = np.percentile(conf, (1 - keep_top_pct) * 100)
    active    = conf >= threshold
    gated     = np.where(active, preds, 0.0)
    return gated, active


def compute_metrics(preds, actuals):
    n = min(len(preds), len(actuals))
    preds, actuals = preds[:n], actuals[:n]
    rmse   = np.sqrt(mean_squared_error(actuals, preds))
    da     = np.mean(np.sign(preds) == np.sign(actuals)) * 100
    gated, active = adaptive_gate(preds)
    da_g   = (np.mean(np.sign(gated[active]) == np.sign(actuals[active])) * 100
              if active.sum() > 0 else da)
    rho, _ = spearmanr(preds, actuals)
    return dict(
        n=n, rmse=rmse, da=da, da_gated=da_g,
        coverage=active.mean() * 100,
        spearman=float(rho) if not np.isnan(rho) else 0.0,
        preds=preds, actuals=actuals, gated=gated, active=active,
    )


# ─────────────────────────────────────────────────────────────
# 6. RUN: ORDER SELECTION + WALK-FORWARD PER HORIZON
# ─────────────────────────────────────────────────────────────
print(f"\n[2/5] Box-Jenkins order selection per horizon...")
chosen_orders   = {}
chosen_diags    = {}
chosen_tiers    = {}
order_logs      = {}

for h in FORECAST_HORIZONS:
    exog_cols = EXOG_BY_HORIZON[h]
    y_tr = df["Return_1d"].values[:n_train]
    X_tr = df[exog_cols].values[:n_train]
    order, log_df, diag, tier = select_order_box_jenkins(y_tr, X_tr, h)
    chosen_orders[h] = order
    chosen_diags[h]  = diag
    chosen_tiers[h]  = tier
    order_logs[h]    = log_df

print(f"\n[3/5] Walk-forward forecasting...")
results = {}
for h in FORECAST_HORIZONS:
    p, a = run_arimax(h, chosen_orders[h])
    results[h] = compute_metrics(p, a)


# ─────────────────────────────────────────────────────────────
# 7. RESULTS REPORT
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 92)
print("  ARIMAX OUT-OF-SAMPLE RESULTS (2024-2025)")
print("=" * 92)
print(f"\n  {'Horizon':<10}{'Order':<12}{'Tier':<6}{'LB_p':<8}"
      f"{'N':>6}{'RMSE':>12}{'DA':>10}{'DA-gated':>12}{'Spearman':>12}")
print("  " + "-" * 88)
for h in FORECAST_HORIZONS:
    r = results[h]
    o = chosen_orders[h]
    d = chosen_diags[h]
    t = chosen_tiers[h]
    print(f"  {h}-step{'':<4}({o[0]},{D_ORDER},{o[1]}){'':<4}{t:<6}"
          f"{d['lb_pvalue']:<8.3f}"
          f"{r['n']:>6d}{r['rmse']:>12.5f}{r['da']:>9.1f}%"
          f"{r['da_gated']:>11.1f}%{r['spearman']:>12.3f}")
print("  " + "-" * 88)
print(f"\n  Tier A: all AR/MA coefficients p<{COEF_PVAL_MAX} AND Ljung-Box({LJUNGBOX_LAG}) p>{LJUNGBOX_ALPHA}")
print(f"  Tier B: at most 1 insignificant AR/MA coef AND Ljung-Box passes")
print(f"  Tier C: fallback to best AIC (residual autocorrelation may persist)")


# ─────────────────────────────────────────────────────────────
# 8. SAVE FORECASTS + ORDER-SEARCH LOGS
# ─────────────────────────────────────────────────────────────
print(f"\n[4/5] Saving CSVs to {OUT_DIR}/...")
for h in FORECAST_HORIZONS:
    r     = results[h]
    dates = test_dates[:r["n"]]
    out = pd.DataFrame({
        "date":   dates,
        "pred":   r["preds"],
        "actual": r["actuals"],
        "gated":  r["gated"],
        "active": r["active"].astype(int),
    })
    fname = os.path.join(OUT_DIR, f"arimax_fc{h}.csv")
    out.to_csv(fname, index=False)
    print(f"  {fname}  ({len(out)} rows)")

    log_fname = os.path.join(OUT_DIR, f"arimax_order_search_h{h}.csv")
    order_logs[h].to_csv(log_fname, index=False)
    print(f"  {log_fname}  ({len(order_logs[h])} candidates)")


# ─────────────────────────────────────────────────────────────
# 9. PLOTS  (matching supervisor's style)
# ─────────────────────────────────────────────────────────────
print(f"\n[5/5] Generating plots in {OUT_DIR}/...")
plt.style.use("seaborn-v0_8-whitegrid")

# Per-horizon 4-panel forecast plots
for h in FORECAST_HORIZONS:
    r       = results[h]
    o       = chosen_orders[h]
    preds   = r["preds"]
    actuals = r["actuals"]
    n       = len(preds)
    dates   = test_dates[:n]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f"ARIMAX({o[0]},{D_ORDER},{o[1]})  |  {h}-Step Ahead Forecast\n"
        f"RMSE: {r['rmse']:.5f}  |  DA: {r['da']:.1f}%  |  "
        f"DA-gated: {r['da_gated']:.1f}%  |  Spearman: {r['spearman']:.3f}",
        fontsize=13, fontweight="bold", y=0.98)

    ax1 = axes[0, 0]
    ax1.plot(dates, actuals, label="Actual", alpha=0.6, lw=0.7, color=ACTUAL_COLOR)
    ax1.plot(dates, preds,   label="Predicted", alpha=0.9, lw=1.0, color=PREDICTED_COLOR)
    ax1.set_title("Full Test Period: Predicted vs Actual Cumulative Returns")
    ax1.set_ylabel("Cumulative Return")
    ax1.legend(loc="lower left"); ax1.tick_params(axis="x", rotation=30)

    ax2 = axes[0, 1]
    zoom = min(100, n)
    ax2.plot(dates[:zoom], actuals[:zoom], label="Actual", alpha=0.6, lw=1.0,
             color=ACTUAL_COLOR, marker="o", markersize=2)
    ax2.plot(dates[:zoom], preds[:zoom], label="Predicted", alpha=0.9, lw=1.2,
             color=PREDICTED_COLOR, marker="s", markersize=2)
    for i in range(zoom):
        if np.sign(preds[i]) == np.sign(actuals[i]):
            ax2.axvspan(dates[i] - pd.Timedelta(hours=12),
                        dates[i] + pd.Timedelta(hours=12),
                        alpha=0.08, color="green")
    ax2.set_title("Zoomed: First 100 Days (green = correct direction)")
    ax2.set_ylabel("Cumulative Return")
    ax2.legend(loc="lower left"); ax2.tick_params(axis="x", rotation=30)

    ax3 = axes[1, 0]
    mn = min(actuals.min(), preds.min())
    mx = max(actuals.max(), preds.max())
    ax3.scatter(actuals, preds, alpha=0.3, s=15, color=ACTUAL_COLOR, edgecolors="none")
    ax3.plot([mn, mx], [mn, mx], "r--", lw=1, alpha=0.7, label="Perfect prediction")
    ax3.axhline(0, color="grey", lw=0.5, alpha=0.5)
    ax3.axvline(0, color="grey", lw=0.5, alpha=0.5)
    ax3.set_title("Scatter: Predicted vs Actual")
    ax3.set_xlabel("Actual Cumulative Return")
    ax3.set_ylabel("Predicted Cumulative Return")
    ax3.legend(loc="upper left", fontsize=9)

    ax4 = axes[1, 1]
    strat_returns = np.where(np.sign(preds) == np.sign(actuals),
                             np.abs(actuals), -np.abs(actuals))
    cum_actual    = np.cumsum(actuals)
    cum_strategy  = np.cumsum(strat_returns)
    ax4.plot(dates, cum_actual,   label="Buy & Hold", alpha=0.7, lw=1.2, color=ACTUAL_COLOR)
    ax4.plot(dates, cum_strategy, label="ARIMAX Strategy", alpha=0.7, lw=1.2,
             color=PREDICTED_COLOR)
    ax4.fill_between(dates, cum_strategy, cum_actual,
                     where=cum_strategy > cum_actual,
                     alpha=0.15, color="green", label="Outperforms")
    ax4.fill_between(dates, cum_strategy, cum_actual,
                     where=cum_strategy <= cum_actual,
                     alpha=0.15, color="red", label="Underperforms")
    ax4.set_title("Cumulative Returns: ARIMAX Strategy vs Buy & Hold")
    ax4.set_ylabel("Cumulative Return")
    ax4.legend(loc="upper left", fontsize=9)
    ax4.tick_params(axis="x", rotation=30)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fname = os.path.join(OUT_DIR, f"arimax_{h}step_forecast.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {fname}")


# Per-horizon price reconstruction plots
for h in FORECAST_HORIZONS:
    r       = results[h]
    o       = chosen_orders[h]
    preds   = r["preds"]
    actuals = r["actuals"]
    n       = len(preds)
    dates   = test_dates[:n]

    actual_prices    = test_prices[:n]
    predicted_prices = np.zeros(n)
    last_train_price = train_df["Close"].iloc[-1]
    for i in range(n):
        prev_price          = last_train_price if i == 0 else actual_prices[i - 1]
        predicted_prices[i] = prev_price * (1 + preds[i])

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(
        f"ARIMAX({o[0]},{D_ORDER},{o[1]})  |  {h}-Step Ahead: Price Reconstruction\n"
        f"RMSE: {r['rmse']:.5f}  |  DA: {r['da']:.1f}%  |  Spearman: {r['spearman']:.3f}",
        fontsize=13, fontweight="bold")

    axes[0].plot(dates, actual_prices,    label="Actual Price",    alpha=0.7,
                 lw=1.2, color=ACTUAL_COLOR)
    axes[0].plot(dates, predicted_prices, label="Predicted Price", alpha=0.7,
                 lw=1.2, color=PREDICTED_COLOR)
    axes[0].set_title("Full Test Period: Predicted vs Actual S&P 500 Price")
    axes[0].set_ylabel("S&P 500 Price ($)")
    axes[0].legend(loc="upper left"); axes[0].tick_params(axis="x", rotation=30)

    zoom = min(100, n)
    axes[1].plot(dates[:zoom], actual_prices[:zoom],
                 label="Actual Price", alpha=0.7, lw=1.2, color=ACTUAL_COLOR,
                 marker="o", markersize=2)
    axes[1].plot(dates[:zoom], predicted_prices[:zoom],
                 label="Predicted Price", alpha=0.7, lw=1.2, color=PREDICTED_COLOR,
                 marker="s", markersize=2)
    for i in range(zoom):
        c = "green" if np.sign(preds[i]) == np.sign(actuals[i]) else "red"
        axes[1].axvspan(dates[i] - pd.Timedelta(hours=12),
                        dates[i] + pd.Timedelta(hours=12), alpha=0.06, color=c)
    axes[1].set_title("Zoomed In: First 100 Days (green = correct direction, red = incorrect)")
    axes[1].set_ylabel("S&P 500 Price ($)")
    axes[1].legend(loc="upper left"); axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    fname = os.path.join(OUT_DIR, f"arimax_{h}step_price.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {fname}")


# Rolling 30-day DA (all + gated)
window    = 30
fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
fig.suptitle(
    f"ARIMAX  |  Rolling {window}-Day Directional Accuracy\n"
    f"Top: all trading days  |  Bottom: active-trade days "
    f"(top {int(KEEP_TOP_PCT*100)}% confidence)",
    fontsize=12, fontweight="bold")

for h in FORECAST_HORIZONS:
    col     = H_COLORS[h]
    r       = results[h]
    preds   = r["preds"]
    actuals = r["actuals"]
    gated   = r["gated"]

    correct = (np.sign(preds) == np.sign(actuals)).astype(float)
    roll    = pd.Series(correct).rolling(window=window).mean() * 100
    axes[0].plot(test_dates[:len(roll)], roll, color=col, lw=1.2,
                 label=f"{h}-step  mean={r['da']:.1f}%")

    roll_g = []
    for j in range(window, len(gated)):
        g_w   = gated[j - window: j]
        a_w   = actuals[j - window: j]
        act_w = g_w != 0
        roll_g.append(
            np.mean(np.sign(g_w[act_w]) == np.sign(a_w[act_w])) * 100
            if act_w.sum() >= 5 else np.nan)
    rg = np.array(roll_g)
    axes[1].plot(test_dates[window: window + len(rg)], rg, color=col, lw=1.2,
                 label=f"{h}-step  DA={r['da_gated']:.1f}%  cov={r['coverage']:.0f}%")

for ax in axes:
    ax.axhline(50, color="grey",  ls="--", lw=1.0, alpha=0.6, label="50% random")
    ax.axhline(60, color="green", ls=":",  lw=0.8, alpha=0.5, label="60% target")
    ax.set_ylabel("Directional Accuracy (%)"); ax.set_ylim(5, 110)
    ax.legend(fontsize=9); ax.tick_params(axis="x", rotation=30); ax.grid(True, alpha=0.3)

axes[0].set_title("All trading days", fontsize=10)
axes[1].set_title(f"Active-trade days only (top {int(KEEP_TOP_PCT*100)}% confidence)",
                  fontsize=10)
plt.tight_layout()
fname = os.path.join(OUT_DIR, "arimax_rolling_da.png")
plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
print(f"  Saved: {fname}")


print("\n[Done]")
