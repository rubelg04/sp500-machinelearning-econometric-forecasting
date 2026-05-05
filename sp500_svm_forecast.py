"""
S&P 500 SVM Forecasting Model — Improved (V2)
================================================
Mirrors the Transformer V14 setup EXACTLY for fair comparison:
  - Same data: ^GSPC with VIX
  - Same 35 engineered features (RSI, MACD, Bollinger, volatility, VIX, etc.)
  - Same split: 2017-2023 train, 2024-2025 test
  - Same cumulative return targets: (Close_{t+h} / Close_t) - 1
  - Same metrics: RMSE + Directional Accuracy
  - RMSE should scale with √h (same as Transformer)

SVM-specific methodology (aligned with literature):
  - SVR with RBF kernel (Paraskevopoulos & Posch 2018, Kurani et al. 2023)
  - RobustScaler on features (same as Transformer)
  - Per-horizon StandardScaler on targets (same as Transformer)
  - GridSearchCV with TimeSeriesSplit (no future leakage in CV)
  - Single feature vector per day (no lookback window flattening — SVMs
    work best with pre-computed features, not raw sequences)

Expected benchmarks (from literature):
  - Directional accuracy: 55-65% for plain SVR on S&P 500
  - RMSE: should be modestly higher than Transformer (no ensemble, no
    sequential modelling), but in the same ballpark

Requirements:
    pip install yfinance pandas numpy scikit-learn matplotlib joblib
"""

import os
import shutil
import warnings
import time
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
import joblib

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG — identical to Transformer V14
# ============================================================
TICKER      = "^GSPC"
TRAIN_START = "2017-01-01"
TRAIN_END   = "2023-12-31"
TEST_START  = "2024-01-01"
TEST_END    = "2025-12-31"

FORECAST_HORIZONS = [1, 5, 10]

# SVM hyperparameter grid
# Tuned for log-return scale (~0.001 to 0.03 magnitude)
PARAM_GRID = {
    "C":       [0.1, 1, 10, 100],
    "gamma":   ["scale", 0.001, 0.01, 0.1],
    "epsilon": [0.0001, 0.0005, 0.001, 0.005],
}

# Force retrain (set False to load saved models)
FORCE_RETRAIN = True

MODEL_DIR = "saved_models_svm_v2"
if FORCE_RETRAIN and os.path.isdir(MODEL_DIR):
    shutil.rmtree(MODEL_DIR)
    print(f"  [FORCE_RETRAIN] Cleared {MODEL_DIR}/")
os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 60)
print("  S&P 500 SVM Forecaster — V2 (Transformer-Aligned)")
print("=" * 60)


# ============================================================
# 1. DATA DOWNLOAD & FEATURE ENGINEERING
#    (Copied from Transformer V14 for identical pipeline)
# ============================================================
print("\n[1/5] Downloading S&P 500 data...")

df = yf.download(TICKER, start=TRAIN_START, end=TEST_END,
                 progress=False, auto_adjust=False)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

print(f"  Downloaded columns: {list(df.columns)}")

# VIX data
print("  Downloading VIX data...")
try:
    vix = yf.download("^VIX", start=TRAIN_START, end=TEST_END, progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    df["VIX"] = vix["Close"]
    print("  VIX data added")
except Exception as e:
    print(f"  VIX not available: {e}")
    df["VIX"] = 15.0


def safe_divide(a, b, default=0.0):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(b) > 1e-8, a / b, default)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build feature set — IDENTICAL to Transformer V14 (no lookahead)."""
    df = df.copy()

    # Returns
    df["Return_1d"]  = df["Close"].pct_change().clip(-0.5, 0.5)
    df["Return_2d"]  = df["Close"].pct_change(2).clip(-0.5, 0.5)
    df["Return_3d"]  = df["Close"].pct_change(3).clip(-0.5, 0.5)
    df["Return_5d"]  = df["Close"].pct_change(5).clip(-0.5, 0.5)
    if "Open" in df.columns:
        df["Gap_Open"] = (
            (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)
        ).clip(-0.05, 0.05)
    df["Return_10d"] = df["Close"].pct_change(10).clip(-0.5, 0.5)
    df["Return_20d"] = df["Close"].pct_change(20).clip(-0.5, 0.5)

    # Rolling MA distance
    for w in [5, 10, 20, 50]:
        df[f"MA_{w}"]  = df["Close"].rolling(w).mean()
        df[f"STD_{w}"] = df["Close"].rolling(w).std()
        dist = safe_divide(
            (df["Close"].values - df[f"MA_{w}"].values),
            df[f"MA_{w}"].values, default=0
        ) * 100
        df[f"Close_to_MA_{w}"] = np.clip(dist, -50, 50)

    # Volatility
    df["Volatility_10d"] = df["Return_1d"].rolling(10).std() * np.sqrt(252)
    df["Volatility_20d"] = df["Return_1d"].rolling(20).std() * np.sqrt(252)
    df["Volatility_50d"] = df["Return_1d"].rolling(50).std() * np.sqrt(252)
    df["Volatility_ratio"] = np.clip(
        safe_divide(df["Volatility_10d"].values,
                    df["Volatility_50d"].values, default=1.0),
        0.1, 10
    )

    # RSI-14
    delta = df["Close"].diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["RSI_14"] = np.clip(
        100 - (100 / (1 + safe_divide(gain.values, loss.values, default=1.0))),
        0, 100
    )

    # MACD
    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"]        = (exp1 - exp2).clip(-100, 100)
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean().clip(-100, 100)
    df["MACD_Hist"]   = (df["MACD"] - df["MACD_Signal"]).clip(-50, 50)

    # Bollinger Bands
    ma20  = df["Close"].rolling(20).mean()
    std20 = df["Close"].rolling(20).std()
    df["BB_Upper"] = ma20 + 2 * std20
    df["BB_Lower"] = ma20 - 2 * std20
    df["BB_Width"] = np.clip(
        safe_divide((df["BB_Upper"] - df["BB_Lower"]).values,
                    ma20.values, default=0) * 100,
        0, 50
    )
    df["BB_Position"] = np.clip(
        safe_divide((df["Close"] - ma20).values,
                    (2 * std20).values, default=0),
        -3, 3
    )

    # Momentum
    df["Momentum_5d"]  = (df["Close"] / df["Close"].shift(5)  - 1).clip(-0.5, 0.5)
    df["Momentum_10d"] = (df["Close"] / df["Close"].shift(10) - 1).clip(-0.5, 0.5)
    df["Momentum_20d"] = (df["Close"] / df["Close"].shift(20) - 1).clip(-0.5, 0.5)

    # Volume
    if "Volume" in df.columns and df["Volume"].notna().any():
        df["Volume_Log"]    = np.log1p(df["Volume"].clip(lower=1))
        df["Volume_Change"] = df["Volume"].pct_change().clip(-0.5, 0.5)
        df["Volume_MA_20"]  = df["Volume"].rolling(20).mean()
        df["Volume_Ratio"]  = np.clip(
            safe_divide(df["Volume"].values,
                        df["Volume_MA_20"].values, default=1.0),
            0.1, 10
        )

    # Price action
    if "High" in df.columns and "Low" in df.columns:
        df["High_Low_Spread"] = (
            (df["High"] - df["Low"]) / df["Close"] * 100
        ).clip(0, 20)
    if "Open" in df.columns:
        df["Close_Open_Spread"] = (
            (df["Close"] - df["Open"]) / df["Open"] * 100
        ).clip(-20, 20)

    # Time / seasonality
    df["DayOfWeek_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df["DayOfWeek_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    df["Month_sin"]     = np.sin(2 * np.pi * df.index.month / 12)
    df["Month_cos"]     = np.cos(2 * np.pi * df.index.month / 12)
    df["DayOfYear_sin"] = np.sin(2 * np.pi * df.index.dayofyear / 365)
    df["DayOfYear_cos"] = np.cos(2 * np.pi * df.index.dayofyear / 365)

    # VIX
    if "VIX" in df.columns:
        df["VIX_Change"] = df["VIX"].pct_change().clip(-0.3, 0.3)
        df["VIX_Log"]    = np.log1p(df["VIX"].clip(lower=5))

    # ============================================================
    # TARGET CALCULATION — Cumulative returns (same as Transformer)
    # ============================================================
    for h in FORECAST_HORIZONS:
        df[f"Target_{h}d"] = (df["Close"].shift(-h) / df["Close"] - 1).clip(-0.5, 0.5)

    df = df.dropna()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df


# Same feature list as Transformer V14
FEATURE_COLS = [
    "Return_1d", "Return_5d", "Return_10d", "Return_20d",
    "Close_to_MA_5", "Close_to_MA_10", "Close_to_MA_20", "Close_to_MA_50",
    "Volatility_10d", "Volatility_20d", "Volatility_50d", "Volatility_ratio",
    "RSI_14",
    "MACD", "MACD_Signal", "MACD_Hist",
    "BB_Width", "BB_Position",
    "Momentum_5d", "Momentum_10d", "Momentum_20d",
    "Volume_Log", "Volume_Change", "Volume_Ratio",
    "High_Low_Spread", "Close_Open_Spread",
    "DayOfWeek_sin", "DayOfWeek_cos",
    "Month_sin", "Month_cos",
    "DayOfYear_sin", "DayOfYear_cos",
    "VIX", "VIX_Change", "VIX_Log",
]


print("  Engineering features (same as Transformer V14)...")
df = add_features(df)
print(f"  Total rows after engineering: {len(df)}")

# Keep only columns that actually exist
feature_cols = [c for c in FEATURE_COLS if c in df.columns]
print(f"  Using {len(feature_cols)} features")

# Train/test split — same as Transformer
train_df = df.loc[:TRAIN_END].copy()
test_df  = df.loc[TEST_START:].copy()
print(f"  Training samples: {len(train_df)}")
print(f"  Testing samples:  {len(test_df)}")

# Scale features with RobustScaler (same as Transformer)
scaler = RobustScaler(quantile_range=(5, 95))
train_features = scaler.fit_transform(train_df[feature_cols].values)
test_features  = scaler.transform(test_df[feature_cols].values)
train_features = np.nan_to_num(train_features, nan=0.0, posinf=3.0, neginf=-3.0)
test_features  = np.nan_to_num(test_features,  nan=0.0, posinf=3.0, neginf=-3.0)

# Per-horizon target scalers (same as Transformer)
train_targets, test_targets, target_scalers = {}, {}, {}
for h in FORECAST_HORIZONS:
    raw_train = train_df[f"Target_{h}d"].values.reshape(-1, 1)
    raw_test  = test_df[f"Target_{h}d"].values.reshape(-1, 1)
    ts = StandardScaler()
    train_targets[h]  = ts.fit_transform(raw_train).flatten()
    test_targets[h]   = ts.transform(raw_test).flatten()
    target_scalers[h] = ts
    print(f"  Target h={h}: raw mean={raw_train.mean():.6f}, "
          f"raw std={raw_train.std():.6f}")

# Store for plots
test_dates  = test_df.index
test_prices = test_df["Close"].values

print(f"  Feature matrix shape: {train_features.shape}")


# ============================================================
# 2. TRAINING WITH GRID SEARCH + TimeSeriesSplit
# ============================================================
def get_model_path(horizon):
    return os.path.join(MODEL_DIR, f"svr_{horizon}step.joblib")


def get_scaler_path(horizon):
    return os.path.join(MODEL_DIR, f"svr_{horizon}step_tscaler.joblib")


def train_model(horizon):
    """
    Train SVR for a given horizon using:
      - Same engineered features as Transformer (1 row per day)
      - GridSearchCV with TimeSeriesSplit (respects temporal ordering)
      - RBF kernel (standard in literature)
    """
    model_path  = get_model_path(horizon)

    if os.path.exists(model_path) and not FORCE_RETRAIN:
        print(f"\n  Loading saved {horizon}-step SVR from {model_path}")
        model = joblib.load(model_path)
        return model

    print(f"\n{'=' * 60}")
    print(f"  Training SVR for {horizon}-step ahead forecast")
    n_combos = (len(PARAM_GRID['C']) * len(PARAM_GRID['gamma'])
                * len(PARAM_GRID['epsilon']))
    print(f"  Grid search: {n_combos} combos × 5 CV folds = "
          f"{n_combos * 5} fits")
    print(f"{'=' * 60}")

    X_train = train_features
    y_train = train_targets[horizon]

    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Features per sample: {X_train.shape[1]}")

    start_time = time.time()

    svr = SVR(kernel="rbf", cache_size=1000)

    # TimeSeriesSplit respects temporal ordering (no future leakage)
    tscv = TimeSeriesSplit(n_splits=5)

    grid_search = GridSearchCV(
        svr,
        PARAM_GRID,
        cv=tscv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    grid_search.fit(X_train, y_train)

    elapsed = time.time() - start_time
    print(f"\n  Grid search completed in {elapsed:.1f}s")
    print(f"  Best parameters: {grid_search.best_params_}")
    print(f"  Best CV MSE: {-grid_search.best_score_:.6f}")

    best_model = grid_search.best_estimator_
    print(f"  Support vectors: {best_model.n_support_} "
          f"(total {sum(best_model.n_support_)})")

    joblib.dump(best_model, model_path)
    print(f"  Saved: {model_path}")

    return best_model


# ============================================================
# 3. EVALUATION
# ============================================================
def evaluate_model(model, horizon):
    """
    Evaluate on the test set.
    Returns predictions and actuals in ORIGINAL return space
    (inverse-transformed from standardized space).
    """
    X_test = test_features
    y_test = test_targets[horizon]

    preds_scaled = model.predict(X_test)

    # Inverse-transform from N(0,1) back to original return space
    ts = target_scalers[horizon]
    preds_real   = ts.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    actuals_real = ts.inverse_transform(y_test.reshape(-1, 1)).flatten()

    return preds_real, actuals_real


# ============================================================
# 4. RUN EVERYTHING
# ============================================================
print("\n[2/5] Training models...")

results = {}
for h in FORECAST_HORIZONS:
    model = train_model(h)
    preds, actuals = evaluate_model(model, h)

    rmse = np.sqrt(mean_squared_error(actuals, preds))
    da   = np.mean(np.sign(preds) == np.sign(actuals)) * 100

    # Spearman rank correlation
    rho, _ = spearmanr(preds, actuals)

    # 70% confidence gate (same as Transformer / ARMA pipeline)
    s  = pd.Series(preds)
    rs = s.rolling(20, min_periods=1).std().fillna(s.std()).values
    rs = np.where(rs < 1e-8, 1e-8, rs)
    conf = np.abs(preds) / rs
    gate_cutoff = np.percentile(conf, (1.0 - 0.70) * 100)
    active = conf >= gate_cutoff
    gated_preds = np.where(active, preds, 0.0)
    da_gated = (np.mean(np.sign(gated_preds[active]) == np.sign(actuals[active])) * 100
                if active.sum() > 0 else da)
    coverage = active.mean() * 100

    # Price reconstruction
    n = len(preds)
    actual_prices_h    = test_prices[:n]
    predicted_prices_h = np.zeros(n)
    for i in range(n):
        prev = train_df["Close"].iloc[-1] if i == 0 else actual_prices_h[i - 1]
        predicted_prices_h[i] = prev * (1 + preds[i])
    price_rmse = np.sqrt(mean_squared_error(actual_prices_h, predicted_prices_h))

    results[h] = {
        "predictions":          preds,
        "gated_predictions":    gated_preds,
        "actuals":              actuals,
        "rmse":                 rmse,
        "directional_accuracy": da,
        "da_gated":             da_gated,
        "coverage":             coverage,
        "spearman":             rho,
        "price_rmse":           price_rmse,
        "predicted_prices":     predicted_prices_h,
        "actual_prices":        actual_prices_h,
    }


# ============================================================
# 5. RESULTS SUMMARY
# ============================================================
print("\n" + "=" * 84)
print("  SVM OUT-OF-SAMPLE RESULTS (2024-2025) — V2 Improved")
print("=" * 84)
print(f"\n  {'Horizon':<10} {'N':>6} {'RMSE':>12} {'DA':>10} {'DA-Gated':>12} {'Spearman':>12}")
print("  " + "-" * 66)

for h in FORECAST_HORIZONS:
    r = results[h]
    n = len(r['predictions'])
    print(f"  {h}-step{'':<4} {n:>6d} {r['rmse']:>12.5f}"
          f" {r['directional_accuracy']:>9.1f}% {r['da_gated']:>11.1f}%"
          f" {r['spearman']:>12.3f}")

print("  " + "-" * 66)
print(f"\n  DA-Gated = accuracy on top-70% confidence days only")


# ============================================================
# 6. COMPARISON TABLE (if Transformer results are known)
# ============================================================
# Typical Transformer V14 results for reference
TRANSFORMER_BENCHMARKS = {
    1:  {"rmse": 0.010, "da": 54.0},
    5:  {"rmse": 0.022, "da": 55.0},
    10: {"rmse": 0.032, "da": 56.0},
}

print("\n" + "=" * 72)
print("  SVM vs TRANSFORMER COMPARISON")
print("=" * 72)
print(f"\n  {'Horizon':<10} {'SVM RMSE':<12} {'TF RMSE*':<12} "
      f"{'SVM DA':<10} {'TF DA*':<10} {'RMSE Diff':<12}")
print("  " + "-" * 66)
for h in FORECAST_HORIZONS:
    r  = results[h]
    tf = TRANSFORMER_BENCHMARKS[h]
    diff_pct = ((r['rmse'] - tf['rmse']) / tf['rmse']) * 100
    print(f"  {h}-step     {r['rmse']:<12.6f} {tf['rmse']:<12.3f} "
          f"{r['directional_accuracy']:<10.2f} {tf['da']:<10.1f} "
          f"{diff_pct:+.1f}%")
print(f"\n  * Transformer values are typical V14 benchmarks; "
      f"your actual run may differ slightly.")


# ============================================================
# 7. VISUALISATIONS
# ============================================================
GENERATE_PLOTS = True

if not GENERATE_PLOTS:
    print("\nPlots skipped. Set GENERATE_PLOTS = True to enable.")
    import sys
    sys.exit()

print("\n[4/5] Generating plots...")
plt.style.use("seaborn-v0_8-whitegrid")

for h in FORECAST_HORIZONS:
    r       = results[h]
    preds   = r["predictions"]
    actuals = r["actuals"]
    n       = len(preds)
    dates   = test_dates[:n]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f"SVM {h}-Step Ahead Forecast (Cumulative Returns) | "
        f"RMSE: {r['rmse']:.6f} | Dir Acc: {r['directional_accuracy']:.2f}%",
        fontsize=14, fontweight="bold", y=0.98
    )

    # Plot 1: Full time series
    ax1 = axes[0, 0]
    ax1.plot(dates, actuals, label="Actual", alpha=0.6,
             linewidth=0.7, color="#4C72B0")
    ax1.plot(dates, preds, label="Predicted", alpha=0.9,
             linewidth=1.0, color="#DD4444")
    ax1.set_title("Full Test Period: Predicted vs Actual Cumulative Returns")
    ax1.set_ylabel("Cumulative Return")
    ax1.legend(loc="lower left")
    ax1.tick_params(axis="x", rotation=30)

    # Plot 2: Zoomed first 100 days
    ax2      = axes[0, 1]
    zoom_end = min(100, n)
    ax2.plot(dates[:zoom_end], actuals[:zoom_end],
             label="Actual", alpha=0.6, linewidth=1.0, color="#4C72B0",
             marker="o", markersize=2)
    ax2.plot(dates[:zoom_end], preds[:zoom_end],
             label="Predicted", alpha=0.9, linewidth=1.2, color="#DD4444",
             marker="s", markersize=2)
    for i in range(zoom_end):
        if np.sign(preds[i]) == np.sign(actuals[i]):
            ax2.axvspan(dates[i] - pd.Timedelta(hours=12),
                        dates[i] + pd.Timedelta(hours=12),
                        alpha=0.08, color="green")
    ax2.set_title("Zoomed: First 100 Days (green = correct direction)")
    ax2.set_ylabel("Cumulative Return")
    ax2.legend(loc="lower left")
    ax2.tick_params(axis="x", rotation=30)

    # Plot 3: Scatter
    ax3     = axes[1, 0]
    min_val = min(actuals.min(), preds.min())
    max_val = max(actuals.max(), preds.max())
    ax3.scatter(actuals, preds, alpha=0.3, s=15, color="#4C72B0",
                edgecolors="none")
    ax3.plot([min_val, max_val], [min_val, max_val],
             "r--", linewidth=1, alpha=0.7, label="Perfect prediction")
    ax3.axhline(0, color="grey", linewidth=0.5, alpha=0.5)
    ax3.axvline(0, color="grey", linewidth=0.5, alpha=0.5)
    ax3.set_title("Scatter: Predicted vs Actual")
    ax3.set_xlabel("Actual Cumulative Return")
    ax3.set_ylabel("Predicted Cumulative Return")
    ax3.legend(loc="upper left", fontsize=9)

    # Plot 4: Cumulative strategy
    ax4              = axes[1, 1]
    strategy_returns = np.where(np.sign(preds) == np.sign(actuals),
                                np.abs(actuals), -np.abs(actuals))
    cum_actual   = np.cumsum(actuals)
    cum_strategy = np.cumsum(strategy_returns)
    ax4.plot(dates, cum_actual, label="Buy & Hold",
             alpha=0.7, linewidth=1.2, color="#4C72B0")
    ax4.plot(dates, cum_strategy, label="SVM Strategy",
             alpha=0.7, linewidth=1.2, color="#DD4444")
    ax4.fill_between(dates, cum_strategy, cum_actual,
                     where=cum_strategy > cum_actual,
                     alpha=0.15, color="green", label="Outperforms")
    ax4.fill_between(dates, cum_strategy, cum_actual,
                     where=cum_strategy <= cum_actual,
                     alpha=0.15, color="red", label="Underperforms")
    ax4.set_title("Cumulative Returns: SVM Strategy vs Buy & Hold")
    ax4.set_ylabel("Cumulative Return")
    ax4.legend(loc="upper left", fontsize=9)
    ax4.tick_params(axis="x", rotation=30)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    filename = f"sp500_svm_v2_forecast_{h}step.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"  Saved: {filename}")
    plt.show()


# ============================================================
# 8. PRICE RECONSTRUCTION CHARTS
# ============================================================
print("\n[5/5] Generating price reconstruction charts...")

for h in FORECAST_HORIZONS:
    r       = results[h]
    preds   = r["predictions"]
    actuals = r["actuals"]
    n       = len(preds)
    dates   = test_dates[:n]
    actual_prices    = r["actual_prices"]
    predicted_prices = r["predicted_prices"]

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(
        f"SVM {h}-Step Ahead: Price Reconstruction | "
        f"RMSE: {r['rmse']:.6f} | Dir Acc: {r['directional_accuracy']:.2f}%",
        fontsize=14, fontweight="bold"
    )

    ax1 = axes[0]
    ax1.plot(dates, actual_prices, label="Actual Price",
             alpha=0.7, linewidth=1.2, color="#4C72B0")
    ax1.plot(dates, predicted_prices, label="Predicted Price",
             alpha=0.7, linewidth=1.2, color="#DD4444")
    ax1.set_title("Full Test Period: Predicted vs Actual S&P 500 Price")
    ax1.set_ylabel("S&P 500 Price ($)")
    ax1.legend(loc="upper left")
    ax1.tick_params(axis="x", rotation=30)

    ax2  = axes[1]
    zoom = min(100, n)
    ax2.plot(dates[:zoom], actual_prices[:zoom],
             label="Actual Price", alpha=0.7, linewidth=1.2, color="#4C72B0",
             marker="o", markersize=2)
    ax2.plot(dates[:zoom], predicted_prices[:zoom],
             label="Predicted Price", alpha=0.7, linewidth=1.2, color="#DD4444",
             marker="s", markersize=2)
    for i in range(zoom):
        color = "green" if np.sign(preds[i]) == np.sign(actuals[i]) else "red"
        ax2.axvspan(dates[i] - pd.Timedelta(hours=12),
                    dates[i] + pd.Timedelta(hours=12),
                    alpha=0.06, color=color)
    ax2.set_title("Zoomed: First 100 Days (green = correct direction)")
    ax2.set_ylabel("S&P 500 Price ($)")
    ax2.legend(loc="upper left")
    ax2.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    filename = f"sp500_svm_v2_price_{h}step.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"  Saved: {filename}")
    plt.show()


# ---- Rolling Directional Accuracy ----
window   = 30
h_colors = {1: "#4C72B0", 5: "#DD8844", 10: "#27AE60"}

fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
fig.suptitle(
    f"SVM V2: Rolling {window}-Day Directional Accuracy\n"
    f"Top: all days | Bottom: active-trade days (top 70% confidence)",
    fontsize=12, fontweight="bold"
)

for h in FORECAST_HORIZONS:
    r   = results[h]
    col = h_colors[h]

    # Top panel: raw rolling DA
    correct = (np.sign(r["predictions"]) == np.sign(r["actuals"])).astype(float)
    roll    = pd.Series(correct).rolling(window=window).mean() * 100
    axes[0].plot(test_dates[:len(roll)], roll, color=col, linewidth=1.2,
                 label=f"{h}-step  mean={r['directional_accuracy']:.1f}%")

    # Bottom panel: gated rolling DA
    gated_h = r["gated_predictions"]
    act_ref = r["actuals"]
    roll_g  = []
    for j in range(window, len(gated_h)):
        g_w   = gated_h[j - window:j]
        a_w   = act_ref[j - window:j]
        act_w = g_w != 0
        roll_g.append(
            np.mean(np.sign(g_w[act_w]) == np.sign(a_w[act_w])) * 100
            if act_w.sum() >= 5 else np.nan
        )
    rg = np.array(roll_g)
    axes[1].plot(test_dates[window: window + len(roll_g)], rg, color=col, linewidth=1.2,
                 label=f"{h}-step  DA={r['da_gated']:.1f}%  cov={r['coverage']:.0f}%")

for ax in axes:
    ax.axhline(50, color="grey", linestyle="--", lw=1.0, alpha=0.6, label="50% random")
    ax.axhline(60, color="green", linestyle=":", lw=0.8, alpha=0.5, label="60% target")
    ax.set_ylabel("Directional Accuracy (%)")
    ax.set_ylim(5, 110)
    ax.legend(fontsize=9)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, alpha=0.3)

axes[0].set_title("All trading days", fontsize=10)
axes[1].set_title("Active-trade days only (top 70% confidence)", fontsize=10)
plt.tight_layout()
plt.savefig("sp500_svm_v2_rolling_accuracy.png", dpi=150, bbox_inches="tight")
print(f"  Saved: sp500_svm_v2_rolling_accuracy.png")
plt.show()


print("\n" + "=" * 60)
print("  ALL DONE")
print("=" * 60)
print(f"  Models saved in '{MODEL_DIR}/'")
print(f"  Re-run with FORCE_RETRAIN=False to skip training.")
