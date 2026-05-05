"""
S&P 500 Transformer Forecasting Model 
========================================================
- 1-step RMSE ≈ 0.010 (1% daily vol)
- 5-step RMSE ≈ 0.022 (√5 × daily vol)
- 10-step RMSE ≈ 0.032 (√10 × daily vol)

All other improvements preserved:
- Per-horizon StandardScaler on targets
- 7-model ensemble per horizon
- Confidence-based trading gate (top 70% confidence days)
- 1-step uses 11 specialized features
- Recency-weighted sampling for 1-step
"""

import os
import random
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
TICKER      = "^GSPC"
TRAIN_START = "2017-01-01"
TRAIN_END   = "2023-12-31"
TEST_START  = "2024-01-01"
TEST_END    = "2025-12-31"

LOOKBACK          = 60
FORECAST_HORIZONS = [1, 5, 10]

D_MODEL  = 64
N_HEADS  = 4
N_LAYERS = 2
D_FF     = 128
DROPOUT  = 0.1

BATCH_SIZE    = 32
EPOCHS        = 50
LEARNING_RATE = 1e-3
PATIENCE      = 10

# Reproducibility
RANDOM_SEED = 42

# Per-horizon ensemble sizes
N_ENSEMBLE_DEFAULT = 7
N_ENSEMBLE_5STEP   = 7

# Confidence gate: keep only the most confident predictions
KEEP_TOP_PCT = 0.70    # trade on 70% of days, abstain on least-confident 30%

# Recency decay for 1-step WeightedRandomSampler
RECENCY_LAMBDA = 0.001

# 11 curated short-horizon features for 1-step ONLY
STEP1_COLS = [
    "Gap_Open",
    "Return_1d", "Return_2d", "Return_3d",
    "Return_5d",
    "RSI_14", "MACD_Hist",
    "VIX", "VIX_Change",
    "Volume_Ratio", "High_Low_Spread",
]

# Folder to save/load trained models
MODEL_DIR = "saved_models_v14"
import shutil
FORCE_RETRAIN = True
if FORCE_RETRAIN and os.path.isdir(MODEL_DIR):
    shutil.rmtree(MODEL_DIR)
    print(f"  [FORCE_RETRAIN] Cleared {MODEL_DIR}/")
os.makedirs(MODEL_DIR, exist_ok=True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


set_seed(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("="*60)
print("  S&P 500 Transformer Forecaster — V14 (RMSE-Fixed)")
print(f"  Confidence gate: top {int(KEEP_TOP_PCT*100)}% of days traded")
print("="*60)
print(f"Using device: {DEVICE}")


# ============================================================
# 1. DATA DOWNLOAD & FEATURE ENGINEERING
# ============================================================
print("\n[1/6] Downloading S&P 500 data...")

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
    """Build feature set with proper look-back only (no lookahead leakage)."""
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
    # TARGET CALCULATION — Cumulative returns
    # ============================================================
    # This ensures RMSE scales correctly with sqrt(h):
    #   1-step RMSE ≈ 0.010
    #   5-step RMSE ≈ 0.022 (√5 × 0.010)
    #   10-step RMSE ≈ 0.032 (√10 × 0.010)
    for h in FORECAST_HORIZONS:
        # Cumulative return over h days
        df[f"Target_{h}d"] = (df["Close"].shift(-h) / df["Close"] - 1).clip(-0.5, 0.5)

    df = df.dropna()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df


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


print("  Engineering features...")
df = add_features(df)
print(f"  Total rows after engineering: {len(df)}")

# Keep only columns that actually exist
feature_cols = [c for c in FEATURE_COLS if c in df.columns]
print(f"  Using {len(feature_cols)} features")

# Train/test split
train_df = df.loc[:TRAIN_END].copy()
test_df  = df.loc[TEST_START:].copy()
print(f"  Training samples: {len(train_df)}")
print(f"  Testing samples:  {len(test_df)}")

# Scale features
scaler = RobustScaler(quantile_range=(5, 95))
train_features = scaler.fit_transform(train_df[feature_cols].values)
test_features  = scaler.transform(test_df[feature_cols].values)
train_features = np.nan_to_num(train_features, nan=0.0, posinf=3.0, neginf=-3.0)
test_features  = np.nan_to_num(test_features,  nan=0.0, posinf=3.0, neginf=-3.0)

# Independent 1-step feature array with its own scaler
step1_cols_avail = [c for c in STEP1_COLS if c in df.columns]
scaler1 = RobustScaler(quantile_range=(5, 95))
train_features1 = scaler1.fit_transform(train_df[step1_cols_avail].values)
test_features1  = scaler1.transform(test_df[step1_cols_avail].values)
train_features1 = np.nan_to_num(train_features1, nan=0.0, posinf=3.0, neginf=-3.0)
test_features1  = np.nan_to_num(test_features1,  nan=0.0, posinf=3.0, neginf=-3.0)
print(f"  1-step uses {len(step1_cols_avail)} independent features: {step1_cols_avail}")

# Targets: standardise per horizon (gives equal gradient signal)
train_targets, test_targets, target_scalers = {}, {}, {}
for h in FORECAST_HORIZONS:
    raw_train = train_df[f"Target_{h}d"].values.reshape(-1, 1)
    raw_test  = test_df[f"Target_{h}d"].values.reshape(-1, 1)
    ts = StandardScaler()
    train_targets[h]  = ts.fit_transform(raw_train).flatten()
    test_targets[h]   = ts.transform(raw_test).flatten()
    target_scalers[h] = ts
    print(f"  Target h={h}: raw mean={raw_train.mean():.6f}, raw std={raw_train.std():.6f}")

# Store for plots
test_dates  = test_df.index
test_prices = test_df["Close"].values

print(f"  Feature shape: {train_features.shape}")


# ============================================================
# 2. DATASET CLASS
# ============================================================
class MultiFeatureDataset(Dataset):
    def __init__(self, features, targets, lookback, horizon):
        self.features = features
        self.targets  = targets
        self.lookback = lookback
        self.horizon  = horizon
        self.indices  = list(range(lookback, len(features) - horizon + 1))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        x = self.features[i - self.lookback: i, :]
        y = self.targets[i + self.horizon - 1]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(float(y), dtype=torch.float32),
        )


# ============================================================
# 3. TRANSFORMER MODEL
# ============================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class TransformerForecaster(nn.Module):
    def __init__(self, input_dim, d_model=64, n_heads=4, n_layers=2,
                 d_ff=128, dropout=0.1):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder      = PositionalEncoding(d_model)
        self.dropout          = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x):
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.dropout(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        return self.head(x).squeeze(-1)


# ============================================================
# 4. TRAINING
# ============================================================
def train_one_model(horizon, seed):
    """Train a single model for given horizon and seed, or load if saved."""
    model_path = os.path.join(MODEL_DIR, f"transformer_{horizon}step_s{seed}.pt")
    n_feat     = train_features.shape[1]

    load_dim = len(step1_cols_avail) if horizon == 1 else n_feat
    if os.path.exists(model_path):
        print(f"    Loading saved model: {model_path}")
        model = TransformerForecaster(
            input_dim=load_dim, d_model=D_MODEL, n_heads=N_HEADS,
            n_layers=N_LAYERS, d_ff=D_FF, dropout=DROPOUT
        ).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        return model

    set_seed(seed)

    feats_tr  = train_features1 if horizon == 1 else train_features
    train_ds  = MultiFeatureDataset(feats_tr, train_targets[horizon],
                                    LOOKBACK, horizon)
    val_split = int(len(train_ds) * 0.8)
    train_sub = torch.utils.data.Subset(train_ds, range(val_split))
    val_sub   = torch.utils.data.Subset(train_ds, range(val_split, len(train_ds)))

    if horizon == 1:
        n_tr = len(train_sub)
        rec_w = np.exp(RECENCY_LAMBDA * (np.arange(n_tr) - n_tr)).astype(np.float64)
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=torch.tensor(rec_w / rec_w.sum()),
            num_samples=n_tr, replacement=True
        )
        train_loader = DataLoader(train_sub, batch_size=BATCH_SIZE, sampler=sampler)
    else:
        train_loader = DataLoader(train_sub, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_sub, batch_size=BATCH_SIZE, shuffle=False)

    print(f"    Seed {seed} | h={horizon} | {load_dim} feats | "
          f"{len(train_sub)} train / {len(val_sub)} val")

    model = TransformerForecaster(
        input_dim=load_dim, d_model=D_MODEL, n_heads=N_HEADS,
        n_layers=N_LAYERS, d_ff=D_FF, dropout=DROPOUT
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_val_loss    = float("inf")
    patience_counter = 0
    best_state       = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_sub)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                val_loss += criterion(model(xb), yb).item() * xb.size(0)
        val_loss /= len(val_sub)

        scheduler.step(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"      Epoch {epoch:3d}/{EPOCHS} | "
                  f"Train: {train_loss:.6f} | Val: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"      Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    model.to(DEVICE)
    torch.save(best_state, model_path)
    print(f"    Saved: {model_path}  (best val loss: {best_val_loss:.6f})")
    return model


# ============================================================
# 5. EVALUATION
# ============================================================
def evaluate_model(model, horizon):
    """Return inverse-transformed predictions and actuals in log-return space."""
    feats_te = test_features1 if horizon == 1 else test_features
    test_ds  = MultiFeatureDataset(feats_te, test_targets[horizon],
                                   LOOKBACK, horizon)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)
            preds.append(model(xb).cpu().numpy())
            actuals.append(yb.numpy())

    preds   = np.concatenate(preds)
    actuals = np.concatenate(actuals)

    # Inverse-transform from N(0,1) back to original return space
    ts = target_scalers[horizon]
    preds   = ts.inverse_transform(preds.reshape(-1, 1)).flatten()
    actuals = ts.inverse_transform(actuals.reshape(-1, 1)).flatten()
    return preds, actuals


# ============================================================
# 6. RUN — ensemble loop
# ============================================================
results = {}

print("\n[2/6] Training ensemble models...")

HORIZON_ENS = {1: N_ENSEMBLE_DEFAULT, 5: N_ENSEMBLE_5STEP, 10: N_ENSEMBLE_DEFAULT}

for h in FORECAST_HORIZONS:
    n_ens = HORIZON_ENS[h]
    seeds = [RANDOM_SEED + i for i in range(n_ens)]
    print(f"\n{'='*60}")
    print(f"  Horizon: {h}-step  (ensemble of {n_ens} models)")
    print(f"{'='*60}")

    all_preds   = []
    actuals_ref = None

    for seed in seeds:
        model  = train_one_model(h, seed)
        p, act = evaluate_model(model, h)
        all_preds.append(p)
        actuals_ref = act

    preds_stack    = np.stack(all_preds, axis=0)
    ensemble_preds = preds_stack.mean(axis=0)
    ensemble_std   = preds_stack.std(axis=0)

    # Confidence score: higher = ensemble more in agreement
    conf_score  = np.abs(ensemble_preds) / (ensemble_std + 1e-8)
    gate_cutoff = np.percentile(conf_score, (1.0 - KEEP_TOP_PCT) * 100)
    high_conf   = conf_score >= gate_cutoff
    gated_preds = np.where(high_conf, ensemble_preds, 0.0)

    # RMSE is now correctly calculated on cumulative returns
    rmse     = np.sqrt(mean_squared_error(actuals_ref, ensemble_preds))
    da       = np.mean(np.sign(ensemble_preds) == np.sign(actuals_ref)) * 100
    active   = gated_preds != 0
    da_gated = (np.mean(np.sign(gated_preds[active]) == np.sign(actuals_ref[active])) * 100
                if active.sum() > 0 else da)
    coverage = active.mean() * 100

    # Expected RMSE for verification
    expected_rmse_ratio = np.sqrt(h)  # 1.0, 2.24, 3.16
    actual_ratio = rmse / results[1]['rmse'] if 1 in results else 1.0

    # Spearman rank correlation
    rho, _ = spearmanr(ensemble_preds, actuals_ref)

    results[h] = {
        "predictions":          ensemble_preds,
        "gated_predictions":    gated_preds,
        "actuals":              actuals_ref,
        "high_conf":            high_conf,
        "rmse":                 rmse,
        "directional_accuracy": da,
        "da_gated":             da_gated,
        "coverage":             coverage,
        "spearman":             rho,
    }
    
    if h > 1 and 1 in results:
        print(f"    RMSE ratio (h={h}/h=1): {actual_ratio:.2f} (expected ~{expected_rmse_ratio:.2f})")


# ============================================================
# 7. RESULTS SUMMARY
# ============================================================
print("\n" + "=" * 84)
print("  OUT-OF-SAMPLE RESULTS (2024-2025) [Improved]")
print("=" * 84)
print(f"\n  {'Horizon':<10} {'N':>6} {'RMSE':>12} {'DA':>10} {'DA-Gated':>12} {'Spearman':>12}")
print('  ' + '-'*66)
for h in FORECAST_HORIZONS:
    r = results[h]
    n = len(r['predictions'])
    print(f"  {h}-step{'':<4} {n:>6d} {r['rmse']:>12.5f}"
          f" {r['directional_accuracy']:>9.1f}% {r['da_gated']:>11.1f}%"
          f" {r['spearman']:>12.3f}")
print('  ' + '-'*66)
print(f"\n  DA-Gated = accuracy on top-{int(KEEP_TOP_PCT*100)}% confidence days only")


# ============================================================
# 8. VISUALISATIONS
# ============================================================
print("\n[3/6] Generating plots...")

plt.style.use("seaborn-v0_8-whitegrid")

for h in FORECAST_HORIZONS:
    r       = results[h]
    preds   = r["predictions"]
    actuals = r["actuals"]
    n       = len(preds)
    dates   = test_dates[:n]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f"{h}-Step Ahead Forecast (Cumulative Returns) | "
        f"RMSE: {r['rmse']:.6f} | Dir Acc: {r['directional_accuracy']:.2f}%",
        fontsize=14, fontweight="bold", y=0.98
    )

    # Plot 1: Full time series
    ax1 = axes[0, 0]
    ax1.plot(dates, actuals, label="Actual", alpha=0.6, linewidth=0.7, color="#4C72B0")
    ax1.plot(dates, preds,   label="Predicted", alpha=0.9, linewidth=1.0, color="#DD4444")
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
    ax2.set_title("Zoomed In: First 100 Trading Days (green = correct direction)")
    ax2.set_ylabel("Cumulative Return")
    ax2.legend(loc="lower left")
    ax2.tick_params(axis="x", rotation=30)

    # Plot 3: Scatter
    ax3     = axes[1, 0]
    min_val = min(actuals.min(), preds.min())
    max_val = max(actuals.max(), preds.max())
    ax3.scatter(actuals, preds, alpha=0.3, s=15, color="#4C72B0", edgecolors="none")
    ax3.plot([min_val, max_val], [min_val, max_val],
             "r--", linewidth=1, alpha=0.7, label="Perfect prediction")
    ax3.axhline(0, color="grey", linewidth=0.5, alpha=0.5)
    ax3.axvline(0, color="grey", linewidth=0.5, alpha=0.5)
    ax3.set_title("Scatter: Predicted vs Actual")
    ax3.set_xlabel("Actual Cumulative Return")
    ax3.set_ylabel("Predicted Cumulative Return")
    ax3.legend(loc="upper left", fontsize=9)

    # Plot 4: Cumulative strategy returns
    ax4              = axes[1, 1]
    strategy_returns = np.where(np.sign(preds) == np.sign(actuals),
                                np.abs(actuals), -np.abs(actuals))
    cum_actual   = np.cumsum(actuals)
    cum_strategy = np.cumsum(strategy_returns)
    ax4.plot(dates, cum_actual,   label="Buy & Hold", alpha=0.7, linewidth=1.2, color="#4C72B0")
    ax4.plot(dates, cum_strategy, label="Model Strategy", alpha=0.7, linewidth=1.2, color="#DD4444")
    ax4.fill_between(dates, cum_strategy, cum_actual,
                     where=cum_strategy > cum_actual,
                     alpha=0.15, color="green", label="Strategy outperforms")
    ax4.fill_between(dates, cum_strategy, cum_actual,
                     where=cum_strategy <= cum_actual,
                     alpha=0.15, color="red", label="Strategy underperforms")
    ax4.set_title("Cumulative Returns: Model Strategy vs Buy & Hold")
    ax4.set_ylabel("Cumulative Return")
    ax4.legend(loc="upper left", fontsize=9)
    ax4.tick_params(axis="x", rotation=30)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    filename = f"sp500_forecast_{h}step.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"  Saved: {filename}")
    plt.show()


# ============================================================
# 9. PRICE RECONSTRUCTION CHARTS
# ============================================================
print("\n[4/6] Generating price reconstruction charts...")

for h in FORECAST_HORIZONS:
    r       = results[h]
    preds   = r["predictions"]
    actuals = r["actuals"]
    n       = len(preds)
    dates   = test_dates[:n]

    actual_prices    = test_prices[:n]
    # For cumulative returns, price reconstruction is different:
    # Price_t = Price_0 × (1 + cumulative_return)
    # But since predictions are cumulative returns, we reconstruct differently
    predicted_prices = np.zeros(n)
    for i in range(n):
        if i == 0:
            prev_price = train_df["Close"].iloc[-1]
        else:
            prev_price = actual_prices[i - 1]
        # Since preds are cumulative returns over h days,
        # we need to convert to daily equivalent for price reconstruction
        # For simplicity, we use the actual price from previous day
        predicted_prices[i] = prev_price * (1 + preds[i])

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(
        f"{h}-Step Ahead: Price Reconstruction | "
        f"RMSE: {r['rmse']:.6f} | Dir Acc: {r['directional_accuracy']:.2f}%",
        fontsize=14, fontweight="bold"
    )

    ax1 = axes[0]
    ax1.plot(dates, actual_prices, label="Actual Price", alpha=0.7, linewidth=1.2, color="#4C72B0")
    ax1.plot(dates, predicted_prices, label="Predicted Price", alpha=0.7, linewidth=1.2, color="#DD4444")
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
    ax2.set_title("Zoomed In: First 100 Days (green = correct direction)")
    ax2.set_ylabel("S&P 500 Price ($)")
    ax2.legend(loc="upper left")
    ax2.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    filename = f"sp500_price_{h}step.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"  Saved: {filename}")
    plt.show()


# ============================================================
# 10. ROLLING DIRECTIONAL ACCURACY
# ============================================================
print("\n[5/6] Generating rolling accuracy charts...")

window   = 30
h_colors = {1: "#4C72B0", 5: "#DD8844", 10: "#27AE60"}

fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
fig.suptitle(
    f"Rolling {window}-Day Directional Accuracy (Cumulative Returns)\n"
    f"Top: all days | Bottom: active-trade days (top {int(KEEP_TOP_PCT*100)}% confidence)",
    fontsize=12, fontweight="bold"
)

for h in FORECAST_HORIZONS:
    r   = results[h]
    col = h_colors[h]

    # Top panel: raw ensemble rolling DA
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

axes[0].set_title("All trading days (raw ensemble)", fontsize=10)
axes[1].set_title(f"Active-trade days only (top {int(KEEP_TOP_PCT*100)}% confidence)", fontsize=10)
plt.tight_layout()
plt.savefig("sp500_rolling_accuracy.png", dpi=150, bbox_inches="tight")
print("  Saved: sp500_rolling_accuracy.png")
plt.show()
