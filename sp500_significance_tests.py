"""
S&P 500 Forecast Significance Testing (combined)
============================================================
Combines directional and RMSE significance testing in a single
script. Replaces the previous two scripts:
  - sp500_significance_tests_v5_1.py  (PT + DM-directional)
  - sp500_rmse_tests.py               (DM on squared-error)

Tests applied:
  1. Pesaran-Timmermann          (directional accuracy, per model)
  2. Diebold-Mariano directional (0-1 sign-error loss, pairwise)
  3. Diebold-Mariano squared err (squared-error / RMSE, pairwise)

Each test is computed under three views:
  - UNGATED   : all overlapping days
  - SELF-GATED: each model's own (or each pair's intersection) active mask
  - REF-GATED : every model evaluated on REFERENCE_MODEL active days only

Test details (DM):
  - Loss differential: d_i = L_a,i - L_b,i
  - Test statistic:    DM = mean(d) / sqrt(HAC_var(mean(d)))
  - HAC: Newey-West with lag = h-1 (handles overlapping forecast windows)
  - Small-sample: Harvey-Leybourne-Newbold (1997) correction
  - Distribution: t with n-1 degrees of freedom (HLN recommendation)
  - DM-directional: two-sided; DM-squared: two-sided

Inputs : CSVs in ./predictions/. Transformer CSV must contain 'active'.
Outputs: Console tables + significance_pt.csv
                         + significance_dm.csv
                         + significance_rmse.csv
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
HORIZONS = [1, 5, 10]
ALPHA    = 0.05

KEEP_TOP_PCT = 0.70
GATE_ROLL    = 20

SEARCH_DIRS = [".", "./predictions", "./outputs", "./results"]
MODELS = ["RandomWalk", "ARIMA", "SVR", "Transformer"]
REFERENCE_MODEL = "Transformer"

FILE_PATTERNS = {
    "RandomWalk":  ["rw_fc{h}.csv", "randomwalk_fc{h}.csv"],
    "ARIMA":       ["arima_*_fc{h}.csv", "arima_fc{h}.csv", "arma_fc{h}.csv"],
    "SVR":         ["svm_fc{h}.csv", "svr_fc{h}.csv"],
    "Transformer": ["transformer_fc{h}.csv", "tfm_fc{h}.csv"],
}

print("=" * 78)
print("  S&P 500 — Forecast Significance Testing (combined)")
print(f"  Reference gate: {REFERENCE_MODEL} (ensemble disagreement)")
print("=" * 78)


# ─────────────────────────────────────────────────────────────
# 1. LOAD PREDICTIONS
# ─────────────────────────────────────────────────────────────
def find_csv(model, h):
    for pattern in FILE_PATTERNS[model]:
        fname = pattern.format(h=h)
        for base in SEARCH_DIRS:
            matches = glob.glob(os.path.join(base, "**", fname), recursive=True)
            if matches:
                return sorted(matches)[-1]
    return None


def load_predictions(model, h):
    path = find_csv(model, h)
    if path is None:
        return None, None, None, None, None
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    pred_col   = cols.get("pred") or cols.get("prediction") or cols.get("predictions")
    actual_col = cols.get("actual") or cols.get("actuals")
    active_col = cols.get("active")
    if pred_col is None or actual_col is None:
        raise ValueError(
            f"{path}: need columns 'pred' and 'actual', found {list(df.columns)}")
    p = df[pred_col].values.astype(float)
    a = df[actual_col].values.astype(float)
    mask = ~(np.isnan(p) | np.isnan(a))
    p, a = p[mask], a[mask]
    if active_col is not None:
        active = df[active_col].values.astype(bool)[mask]
        gate_source = "saved"
    else:
        active = None
        gate_source = None
    return p, a, active, gate_source, path


def compute_rolling_std_gate(preds, keep_top_pct=KEEP_TOP_PCT, roll=GATE_ROLL):
    s  = pd.Series(preds)
    rs = s.rolling(roll, min_periods=1).std().fillna(s.std()).values
    rs = np.where(rs < 1e-8, 1e-8, rs)
    conf = np.abs(preds) / rs
    threshold = np.percentile(conf, (1 - keep_top_pct) * 100)
    return conf >= threshold


# ─────────────────────────────────────────────────────────────
# 2. SHARED STAT HELPERS
# ─────────────────────────────────────────────────────────────
def newey_west_var(x, lag):
    """Newey-West HAC variance estimator for the sample mean of x."""
    n = len(x)
    x = x - x.mean()
    gamma0 = np.dot(x, x) / n
    var = gamma0
    for k in range(1, lag + 1):
        gamma_k = np.dot(x[k:], x[:-k]) / n
        weight = 1 - k / (lag + 1)
        var += 2 * weight * gamma_k
    return var / n


def _dm_from_loss(d, h):
    """
    Generic Diebold-Mariano test from a precomputed loss-differential
    series d_i = L_a,i - L_b,i. Two-sided p-value with HLN correction.
    """
    n = len(d)
    out = dict(n=n, mean_diff=np.nan, dm_stat=np.nan, p_value=np.nan)
    if n < 10:
        out["mean_diff"] = d.mean() if n > 0 else np.nan
        return out

    mean_d = d.mean()
    out["mean_diff"] = mean_d
    lag = max(h - 1, 0)
    var_mean = newey_west_var(d, lag)

    if var_mean <= 0:
        if abs(mean_d) < 1e-12:
            out["dm_stat"] = 0.0
            out["p_value"] = 1.0
        return out

    dm_stat = mean_d / np.sqrt(var_mean)
    hln_factor = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_stat *= hln_factor
    p_value = 2 * (1 - stats.t.cdf(np.abs(dm_stat), df=n - 1))
    out["dm_stat"] = dm_stat
    out["p_value"] = p_value
    return out


# ─────────────────────────────────────────────────────────────
# 3. INDIVIDUAL TESTS
# ─────────────────────────────────────────────────────────────
def pesaran_timmermann(preds, actuals):
    """One-sided test of directional predictive ability."""
    p_sign = np.sign(preds)
    a_sign = np.sign(actuals)
    n = len(p_sign)
    out = dict(n=n, hit_rate=np.nan, expected=np.nan,
               pt_stat=np.nan, p_value=np.nan, status="ok")
    if n < 10:
        out["status"] = "small_sample"
        return out
    P = np.mean(p_sign == a_sign)
    out["hit_rate"] = P
    if np.all(preds == 0):
        out["status"] = "all_zero"
        return out
    unique_signs = np.unique(p_sign[p_sign != 0])
    if len(unique_signs) < 2:
        out["status"] = "constant_sign"
        if len(unique_signs) == 1 and unique_signs[0] > 0:
            out["expected"] = np.mean(actuals > 0)
        elif len(unique_signs) == 1 and unique_signs[0] < 0:
            out["expected"] = np.mean(actuals < 0)
        return out
    P_y = np.mean(a_sign > 0)
    P_x = np.mean(p_sign > 0)
    P_star = P_y * P_x + (1 - P_y) * (1 - P_x)
    out["expected"] = P_star
    var_P      = (P_star * (1 - P_star)) / n
    var_P_star = ((2 * P_y - 1)**2 * P_x * (1 - P_x) +
                  (2 * P_x - 1)**2 * P_y * (1 - P_y) +
                  4 * P_x * P_y * (1 - P_x) * (1 - P_y) / n) / n
    denom = var_P - var_P_star
    if denom <= 0:
        out["status"] = "degenerate_variance"
        return out
    pt_stat = (P - P_star) / np.sqrt(denom)
    p_value = 1 - stats.norm.cdf(pt_stat)
    out["pt_stat"] = pt_stat
    out["p_value"] = p_value
    return out


def dm_directional(preds_a, preds_b, actuals, h):
    """DM on 0-1 directional loss (sign(pred) != sign(actual))."""
    a = np.sign(actuals)
    loss_a = (np.sign(preds_a) != a).astype(float)
    loss_b = (np.sign(preds_b) != a).astype(float)
    return _dm_from_loss(loss_a - loss_b, h)


def dm_squared_error(preds_a, preds_b, actuals, h):
    """DM on squared-error loss (returns rmse_a, rmse_b for reporting)."""
    err_a = preds_a - actuals
    err_b = preds_b - actuals
    loss_a = err_a ** 2
    loss_b = err_b ** 2
    out = _dm_from_loss(loss_a - loss_b, h)
    out["mse_a"] = loss_a.mean()
    out["mse_b"] = loss_b.mean()
    return out


# ─────────────────────────────────────────────────────────────
# 4. LOAD ALL DATA
# ─────────────────────────────────────────────────────────────
print("\n[1/4] Loading predictions...")
data = {}  # data[model][h] = (preds, actuals, active, gate_source)
for model in MODELS:
    data[model] = {}
    for h in HORIZONS:
        p, a, active, gate_source, path = load_predictions(model, h)
        if p is None:
            print(f"  [MISSING] {model} h={h}")
            continue
        if active is None:
            if np.all(p == 0):
                active = np.ones(len(p), dtype=bool)
                gate_source = "trivial"
            else:
                active = compute_rolling_std_gate(p)
                gate_source = "rolling_std"
        data[model][h] = (p, a, active, gate_source)
        rmse = np.sqrt(np.mean((p - a) ** 2))
        print(f"  {model:12s} h={h:>2d}  n={len(p):>4d}  "
              f"active={active.sum():>4d}  RMSE={rmse:.5f}  "
              f"gate={gate_source}")

print(f"\n  Reference gate source for {REFERENCE_MODEL}:")
for h in HORIZONS:
    if h in data[REFERENCE_MODEL]:
        gs = data[REFERENCE_MODEL][h][3]
        flag = "OK ensemble-based" if gs == "saved" else "FALLBACK to rolling_std"
        print(f"    h={h}: {gs}   {flag}")


# ─────────────────────────────────────────────────────────────
# 5. ALIGNMENT HELPER (for REF-GATED)
# ─────────────────────────────────────────────────────────────
def align_to_reference(p, a, active, ref_len):
    """Tail-trim a model's arrays to match the reference model's length."""
    n_pred = len(p)
    if n_pred >= ref_len:
        offset = n_pred - ref_len
        return (p[offset:offset + ref_len],
                a[offset:offset + ref_len],
                active[offset:offset + ref_len])
    return p, a, active


# ─────────────────────────────────────────────────────────────
# 6. PESARAN-TIMMERMANN TABLES
# ─────────────────────────────────────────────────────────────
def print_pt_row(model, h, p_sub, a_sub, gate_source, label_for_csv, rows_list):
    r = pesaran_timmermann(p_sub, a_sub)
    sig = ""
    if not np.isnan(r["p_value"]):
        if r["p_value"] < 0.01:  sig = "***"
        elif r["p_value"] < 0.05: sig = "**"
        elif r["p_value"] < 0.10: sig = "*"
    note = ""
    if r["status"] == "all_zero":              note = "all-zero"
    elif r["status"] == "constant_sign":       note = "const-sign"
    elif r["status"] == "small_sample":        note = "n<10"
    elif r["status"] == "degenerate_variance": note = "degenerate"
    else:                                      note = sig
    hit_str = f"{r['hit_rate']*100:.1f}%" if not np.isnan(r['hit_rate']) else "N/A"
    exp_str = f"{r['expected']*100:.1f}%" if not np.isnan(r['expected']) else "N/A"
    pt_str  = f"{r['pt_stat']:.3f}"  if not np.isnan(r['pt_stat'])  else "—"
    pv_str  = f"{r['p_value']:.4f}"  if not np.isnan(r['p_value'])  else "—"
    print(f"  {model:<14}{h:>8d}{r['n']:>6d}{hit_str:>11}"
          f"{exp_str:>11}{pt_str:>10}{pv_str:>10}{note:>14}")
    rows_list.append({
        "test": "Pesaran-Timmermann", "subset": label_for_csv,
        "model": model, "horizon": h, "n": r["n"],
        "hit_rate": r["hit_rate"], "expected": r["expected"],
        "statistic": r["pt_stat"], "p_value": r["p_value"],
        "status": r["status"], "gate_source": gate_source,
        "significant_5pct": (not np.isnan(r["p_value"])) and r["p_value"] < ALPHA,
    })


def pt_header(label):
    print(f"\n{label}")
    print("=" * 78)
    print(f"  {'Model':<14}{'Horizon':>8}{'N':>6}{'Hit Rate':>11}"
          f"{'Expected':>11}{'PT Stat':>10}{'p-value':>10}{'Note':>14}")
    print("  " + "-" * 76)


all_pt_rows = []

# UNGATED
pt_header("[2/4a] Pesaran-Timmermann — UNGATED (all days)")
for model in MODELS:
    for h in HORIZONS:
        if h not in data[model]: continue
        p, a, active, gs = data[model][h]
        print_pt_row(model, h, p, a, gs, "UNGATED", all_pt_rows)
print("  " + "-" * 76)

# SELF-GATED
pt_header("[2/4b] Pesaran-Timmermann — SELF-GATED (own active mask)")
for model in MODELS:
    for h in HORIZONS:
        if h not in data[model]: continue
        p, a, active, gs = data[model][h]
        print_pt_row(model, h, p[active], a[active], gs, "SELF-GATED", all_pt_rows)
print("  " + "-" * 76)

# REF-GATED
pt_header(f"[2/4c] Pesaran-Timmermann — REF-GATED ({REFERENCE_MODEL} ensemble mask)")
for model in MODELS:
    for h in HORIZONS:
        if h not in data[model] or h not in data[REFERENCE_MODEL]:
            continue
        p, a, active, gs = data[model][h]
        ref_active = data[REFERENCE_MODEL][h][2]
        n_ref = len(ref_active)
        n_mod = len(p)
        if n_mod >= n_ref:
            offset = n_mod - n_ref
            p_aligned = p[offset:offset + n_ref]
            a_aligned = a[offset:offset + n_ref]
        else:
            p_aligned = p
            a_aligned = a
            ref_active = ref_active[-n_mod:]
        p_sub = p_aligned[ref_active]
        a_sub = a_aligned[ref_active]
        print_pt_row(model, h, p_sub, a_sub, "reference",
                     f"REF-GATED ({REFERENCE_MODEL})", all_pt_rows)
print("  " + "-" * 76)


# ─────────────────────────────────────────────────────────────
# 7. DM TABLES (generic — works for directional or squared-error)
# ─────────────────────────────────────────────────────────────
def print_dm_table(label, mode, csv_subset, *, loss_kind,
                   pairs_include_random_walk=False):
    """
    loss_kind: 'directional' or 'squared'
    pairs_include_random_walk:
      directional DM excludes RandomWalk (sign(0)=0 makes the loss degenerate);
      squared-error DM keeps RandomWalk (the canonical RMSE benchmark).
    """
    print(f"\n{label}")
    print("=" * 90)
    if loss_kind == "squared":
        print(f"  {'Comparison':<28}{'Horizon':>4}{'N':>5}"
              f"{'RMSE A':>11}{'RMSE B':>11}{'DM Stat':>10}{'p-value':>10}{'Better':>14}")
    else:
        print(f"  {'Comparison':<28}{'Horizon':>8}{'N':>6}"
              f"{'DM Stat':>10}{'p-value':>10}{'Better':>15}")
    print("  " + "-" * 88)

    if pairs_include_random_walk:
        models_for_pairs = MODELS
    else:
        models_for_pairs = [m for m in MODELS if m != "RandomWalk"]
    pairs = [(a, b) for i, a in enumerate(models_for_pairs)
                    for b in models_for_pairs[i+1:]]

    rows = []
    for model_a, model_b in pairs:
        for h in HORIZONS:
            if h not in data[model_a] or h not in data[model_b]:
                continue
            pa, aa, ma, _ = data[model_a][h]
            pb, ab, mb, _ = data[model_b][h]

            if mode == "reference":
                if h not in data[REFERENCE_MODEL]:
                    continue
                ref_active = data[REFERENCE_MODEL][h][2]
                ref_len = len(ref_active)
                pa, aa, ma = align_to_reference(pa, aa, ma, ref_len)
                pb, ab, mb = align_to_reference(pb, ab, mb, ref_len)
                idx = ref_active
            else:
                n = min(len(pa), len(pb), len(aa), len(ab))
                pa, pb, aa, ma, mb = pa[:n], pb[:n], aa[:n], ma[:n], mb[:n]
                if mode == "ungated":
                    idx = np.ones(n, dtype=bool)
                elif mode == "self":
                    idx = ma & mb
                else:
                    raise ValueError(mode)

            if idx.sum() < 10:
                print(f"  {model_a} vs {model_b:<14}{h:>4d} "
                      f"sample too small (n={idx.sum()}), skipping")
                continue

            pa_, pb_, aa_ = pa[idx], pb[idx], aa[idx]
            if loss_kind == "directional":
                r = dm_directional(pa_, pb_, aa_, h)
            else:
                r = dm_squared_error(pa_, pb_, aa_, h)

            sig = ""
            better = "—"
            if not np.isnan(r["p_value"]):
                if r["p_value"] < 0.01:  sig = "***"
                elif r["p_value"] < 0.05: sig = "**"
                elif r["p_value"] < 0.10: sig = "*"
                if r["p_value"] < ALPHA:
                    better = model_a if r["mean_diff"] < 0 else model_b
                else:
                    better = "no diff"

            comp = f"{model_a} vs {model_b}"
            dm_str = f"{r['dm_stat']:.3f}"  if not np.isnan(r['dm_stat']) else "—"
            pv_str = f"{r['p_value']:.4f}{sig}" if not np.isnan(r['p_value']) else "—"

            if loss_kind == "squared":
                rmse_a = np.sqrt(r["mse_a"])
                rmse_b = np.sqrt(r["mse_b"])
                print(f"  {comp:<28}{h:>4d}{r['n']:>5d}"
                      f"{rmse_a:>11.5f}{rmse_b:>11.5f}"
                      f"{dm_str:>10}{pv_str:>10}{better:>14}")
                rows.append({
                    "test": "DM-squared-error", "subset": csv_subset,
                    "model_a": model_a, "model_b": model_b,
                    "horizon": h, "n": r["n"],
                    "rmse_a": rmse_a, "rmse_b": rmse_b,
                    "statistic": r["dm_stat"], "p_value": r["p_value"],
                    "mean_loss_diff": r["mean_diff"],
                    "significant_5pct": (not np.isnan(r["p_value"])) and r["p_value"] < ALPHA,
                    "better_model": better,
                })
            else:
                print(f"  {comp:<28}{h:>8d}{r['n']:>6d}"
                      f"{dm_str:>10}{pv_str:>10}{better:>15}")
                rows.append({
                    "test": "DM-directional", "subset": csv_subset,
                    "model_a": model_a, "model_b": model_b,
                    "horizon": h, "n": r["n"],
                    "statistic": r["dm_stat"], "p_value": r["p_value"],
                    "mean_loss_diff": r["mean_diff"],
                    "significant_5pct": (not np.isnan(r["p_value"])) and r["p_value"] < ALPHA,
                    "better_model": better,
                })
    print("  " + "-" * 88)
    return rows


# ─── DM on directional loss (excludes Random Walk) ─────────────
all_dm_rows = []
all_dm_rows += print_dm_table(
    "[3/4a] Diebold-Mariano (directional) — UNGATED (all days)",
    mode="ungated", csv_subset="UNGATED", loss_kind="directional")
all_dm_rows += print_dm_table(
    "[3/4b] Diebold-Mariano (directional) — SELF-GATED (each pair's intersection)",
    mode="self", csv_subset="SELF-GATED", loss_kind="directional")
all_dm_rows += print_dm_table(
    f"[3/4c] Diebold-Mariano (directional) — REF-GATED ({REFERENCE_MODEL} ensemble mask)",
    mode="reference", csv_subset=f"REF-GATED ({REFERENCE_MODEL})",
    loss_kind="directional")


# ─── DM on squared-error loss / RMSE (includes Random Walk) ────
all_rmse_rows = []
all_rmse_rows += print_dm_table(
    "[4/4a] DM on squared error / RMSE — UNGATED (all days)",
    mode="ungated", csv_subset="UNGATED",
    loss_kind="squared", pairs_include_random_walk=True)
all_rmse_rows += print_dm_table(
    "[4/4b] DM on squared error / RMSE — SELF-GATED (intersection)",
    mode="self", csv_subset="SELF-GATED",
    loss_kind="squared", pairs_include_random_walk=True)
all_rmse_rows += print_dm_table(
    f"[4/4c] DM on squared error / RMSE — REF-GATED ({REFERENCE_MODEL} ensemble mask)",
    mode="reference", csv_subset=f"REF-GATED ({REFERENCE_MODEL})",
    loss_kind="squared", pairs_include_random_walk=True)


# ─────────────────────────────────────────────────────────────
# 8. NOTES + SAVE
# ─────────────────────────────────────────────────────────────
print("\nNotes:")
print("  Sig stars: * p<0.10  ** p<0.05  *** p<0.01")
print("  PT is one-sided; both DM tests are two-sided.")
print("  HAC lag = h-1, Harvey-Leybourne-Newbold small-sample correction.")
print("  DM-directional excludes Random Walk (sign(0)=0 → degenerate loss).")
print("  DM-squared keeps Random Walk: it is the canonical RMSE benchmark;")
print("  any model significantly beating it has genuine predictive value.")
print("  'Better' identifies the model with significantly lower loss/RMSE.")
print("  Negative DM stat → first-named model has lower loss.")
print("  Length alignment: ARIMA/SVR have more predictions than the")
print(f"  Transformer (warmup). Their arrays are tail-trimmed to align")
print(f"  with Transformer length under REF-GATED mode.")
print(f"  REF-GATED tests evaluate every model on the SAME days that the")
print(f"  {REFERENCE_MODEL} flagged as high-conviction.")

pt_df   = pd.DataFrame(all_pt_rows)
dm_df   = pd.DataFrame(all_dm_rows)
rmse_df = pd.DataFrame(all_rmse_rows)
pt_df.to_csv("significance_pt.csv",     index=False)
dm_df.to_csv("significance_dm.csv",     index=False)
rmse_df.to_csv("significance_rmse.csv", index=False)
print(f"\n  Saved: significance_pt.csv    ({len(pt_df)} rows)")
print(f"  Saved: significance_dm.csv    ({len(dm_df)} rows)")
print(f"  Saved: significance_rmse.csv  ({len(rmse_df)} rows)")

print("\n[Done]")
