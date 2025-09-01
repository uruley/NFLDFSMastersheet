#!/usr/bin/env python3
"""
INFERENCE: RB (QB/WR-style)
- Backtest: 2024 Wks 13–17 (predictions + actuals)
- 2025 Wk1: use last 2024 lag row per player to seed features
- DK merge via master sheet (preferred) → player_id
- SHAP rationales; value = points / (Salary/1000)
- Outputs: rb_predictions.csv
"""

import warnings, json, pickle, re
import os
from pathlib import Path

import numpy as np
import pandas as pd
import nfl_data_py as nfl

# Avoid Windows/joblib stalls when counting cores / spawning processes
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

warnings.filterwarnings("ignore")

# =========================
# CONFIG / PATHS
# =========================
# Only compute SHAP for LGBM (fast). Skip CatBoost/sklearn to avoid hangs.
SHAP_MODELS = {"lightgbm"}
DEFAULT_MODEL_DIR = "RB_Model"
# Change if you keep master sheet somewhere else:
DEFAULT_MASTER_SHEET = r"C:\Users\ruley\NFLDFSMasterSheet\data\processed\master_sheet_2025.csv"
# Optional helpers (only used if master lacks player_id)
DEFAULT_CROSSWALK = r"C:\Users\ruley\NFLDFSMasterSheet\data\processed\crosswalk_2025.csv"
DEFAULT_ALIASES   = r"C:\Users\ruley\NFLDFSMasterSheet\data\xwalk\aliases.csv"

# Team harmonization (DK → NFL API)
TEAM_MAP = {
    "JAC":"JAX","LA":"LAR","LAR":"LAR","STL":"LAR",
    "OAK":"LVR","LV":"LVR","WSH":"WAS","WAS":"WAS",
    # others map to themselves
}

def map_team(t: str) -> str:
    s = str(t).upper()
    return TEAM_MAP.get(s, s)

def norm_name(s: str) -> str:
    if not isinstance(s, str): return ""
    s = s.lower().strip()
    s = re.sub(r"[.\'\-]", "", s)
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

# =========================
# LOAD ARTIFACTS
# =========================
def load_models(model_dir: Path):
    models = {}
    for name in ["lightgbm","catboost","random_forest","gradient_boosting"]:
        p = model_dir / f"{name}_model.pkl"
        if p.exists():
            with open(p, "rb") as f:
                models[name] = pickle.load(f)
            print(f"✅ Loaded {name}")
        else:
            print(f"⚠️ Missing model: {p.name}")
    return models

def load_encoders(model_dir: Path):
    enc = {}
    p = model_dir / "encoders.pkl"
    if p.exists():
        with open(p, "rb") as f:
            enc = pickle.load(f)
        print("✅ Loaded encoders")
    else:
        print("⚠️ No encoders.pkl found; continuing without (will default to 0)")
    return enc

def load_schema(model_dir: Path):
    p = model_dir / "feature_schema.json"
    with open(p, "r") as f:
        schema = json.load(f)
    feat_cols = schema["columns"]
    print(f"✅ Loaded schema with {len(feat_cols)} features")
    return feat_cols

# =========================
# FEATURE ENGINEERING (same as training)
# =========================
LAG_BASE_COLS = [
    "carries","rushing_yards","rushing_tds",
    "receptions","targets","receiving_yards","receiving_tds",
    "fantasy_points_ppr"
]
LAG_WINDOWS = (3, 5, 10)

def add_rb_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["player_id","season","week"]).copy()
    for col in LAG_BASE_COLS:
        if col not in df.columns:
            df[col] = 0.0
        for k in LAG_WINDOWS:
            df[f"{col}_avg_last_{k}"] = (
                df.groupby("player_id")[col]
                  .shift(1)
                  .rolling(k, min_periods=1)
                  .mean()
            )
    for k in LAG_WINDOWS:
        ca = df.get(f"carries_avg_last_{k}", pd.Series(0.0, index=df.index)).fillna(0)
        ry = df.get(f"rushing_yards_avg_last_{k}", pd.Series(0.0, index=df.index)).fillna(0)
        rc = df.get(f"receptions_avg_last_{k}", pd.Series(0.0, index=df.index)).fillna(0)
        rcy = df.get(f"receiving_yards_avg_last_{k}", pd.Series(0.0, index=df.index)).fillna(0)
        touches = ca + rc
        df[f"yards_per_carry_last_{k}"] = np.where(ca > 0, ry / ca, 0.0)
        df[f"yards_per_reception_last_{k}"] = np.where(rc > 0, rcy / rc, 0.0)
        df[f"touches_avg_last_{k}"] = touches
        df[f"yards_per_touch_last_{k}"] = np.where(touches > 0, (ry + rcy) / touches, 0.0)

    df["early_season"] = (df["week"] <= 4).astype(int)
    df["mid_season"]   = ((df["week"] > 4) & (df["week"] <= 12)).astype(int)
    df["late_season"]  = (df["week"] > 12).astype(int)
    df["week_progression"] = df["week"] / 18.0
    return df

def apply_encoders(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    df = df.copy()
    # season_type default
    if "season_type" in df.columns:
        df["season_type"] = df["season_type"].fillna("REG")
    else:
        df["season_type"] = "REG"
    # map each encoder
    for logical, key in [("recent_team","team"),("opponent_team","opponent"),("season_type","season_type")]:
        if key in encoders:
            le = encoders[key]
            def map_one(x):
                s = "REG" if (logical=="season_type" and (x is None or pd.isna(x))) else (str(x) if not pd.isna(x) else "UNK")
                return le.transform([s if s in le.classes_ else "UNK"])[0]
            df[f"{key}_encoded"] = df[logical].map(map_one) if logical in df.columns else 0
        else:
            df[f"{key}_encoded"] = 0
    return df

def prepare_X(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    # ensure all schema cols exist
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0
    return df[feature_cols].fillna(0.0)

# =========================
# SHAP utilities
# =========================
def shap_for_lightgbm(model, X):
    try:
        sv = model.predict(X, pred_contrib=True)
        base = sv[:, -1]
        vals = sv[:, :-1]
        return base, vals
    except Exception:
        return None, None

def shap_for_catboost(model, X):
    try:
        import catboost as cb
        pool = cb.Pool(X)
        sv = model.get_feature_importance(type='ShapValues', data=pool)
        base = sv[:, -1]
        vals = sv[:, :-1]
        return base, vals
    except Exception:
        return None, None

def shap_for_sklearn_tree(model, X):
    try:
        import shap as shaplib
        explainer = shaplib.TreeExplainer(model)
        vals = explainer.shap_values(X)
        base = np.full(X.shape[0], explainer.expected_value, dtype=float)
        return base, vals
    except Exception:
        return None, None

def ensemble_shap(shap_dict):
    bases, vals = [], []
    for b, v in shap_dict.values():
        if b is not None and v is not None:
            bases.append(b); vals.append(v)
    if not bases:
        return None, None
    base_e = np.mean(np.vstack(bases), axis=0)
    vals_e = np.mean(np.stack(vals, axis=0), axis=0)
    return base_e, vals_e

def top_rationales(X, feature_names, base, shap_vals, k=3, decimals=2):
    if base is None or shap_vals is None:
        return ["(no SHAP available)"] * len(X)
    out = []
    for i in range(len(X)):
        contrib = shap_vals[i]
        order = np.argsort(-np.abs(contrib))
        tops = []
        for j in order[:k]:
            sign = "↑" if contrib[j] > 0 else "↓"
            tops.append(f"{feature_names[j]} {sign} {abs(contrib[j]):.{decimals}f}")
        pred = base[i] + contrib.sum()
        out.append(f"~{pred:.1f} pts (base {base[i]:.1f}) • " + ", ".join(tops))
    return out

# =========================
# DK mapping helpers
# =========================
def attach_player_id_from_master(dk_rb: pd.DataFrame) -> pd.DataFrame:
    """If master has player_id, return dk_rb with that column."""
    if "player_id" in dk_rb.columns and dk_rb["player_id"].notna().any():
        return dk_rb
    return dk_rb  # no-op; we’ll try crosswalk in the caller

def try_crosswalk(dk_rb: pd.DataFrame, crosswalk_path: str, aliases_path: str | None) -> pd.DataFrame:
    """Fallback mapping if master lacks ids."""
    dk_rb = dk_rb.copy()
    dk_rb["name_norm"] = dk_rb["Name"].map(norm_name)
    dk_rb["team_norm"] = dk_rb["TeamAbbrev"].map(map_team)
    dk_rb["pos_norm"]  = dk_rb["Position"].astype(str).str.upper()

    try:
        cross = pd.read_csv(crosswalk_path)
        cross["name_norm"] = cross["player_name"].map(norm_name)
        cross["team_norm"] = cross["recent_team"].map(map_team)
        # primary
        m = dk_rb.merge(
            cross[["name_norm","team_norm","player_id"]],
            on=["name_norm","team_norm"], how="left"
        )
    except Exception:
        m = dk_rb.copy()
        m["player_id"] = np.nan

    # alias rescue
    if aliases_path:
        try:
            aliases = pd.read_csv(aliases_path)
            # guess alias column
            name_col = next((c for c in ["alias","alt_name","player_name","name"] if c in aliases.columns), None)
            if name_col:
                aliases["name_norm"] = aliases[name_col].map(norm_name)
                miss = m["player_id"].isna()
                if miss.any():
                    fill = m.loc[miss, ["name_norm"]].merge(
                        aliases[["name_norm","player_id"]].drop_duplicates(),
                        on="name_norm", how="left"
                    )["player_id"].values
                    m.loc[miss, "player_id"] = fill
        except Exception:
            pass

    return m

# =========================
# MAIN INFERENCE
# =========================
def run_inference(
    model_dir=DEFAULT_MODEL_DIR,
    master_sheet_path=DEFAULT_MASTER_SHEET,
    crosswalk_path=DEFAULT_CROSSWALK,
    aliases_path=DEFAULT_ALIASES,
    upcoming_season=2025, upcoming_week=1,
    backtest_weeks=(13,14,15,16,17),
    output_csv="rb_predictions.csv"
):
    print("=== RB Model Inference ===")
    model_dir = Path(model_dir)

    # Artifacts
    models = load_models(model_dir)
    encoders = load_encoders(model_dir)
    feature_cols = load_schema(model_dir)
    
    # Make RF single-threaded to avoid loky stalls on Windows
    rf = models.get("random_forest")
    if rf is not None and hasattr(rf, "n_jobs"):
        try:
            rf.set_params(n_jobs=1)
        except Exception:
            try:
                rf.n_jobs = 1
            except Exception:
                pass

    # ---------------- Historical backtest (2024 w13-17) ----------------
    print("Loading 2024 RB data for backtest...")
    wk24 = nfl.import_weekly_data([2024])
    rb24 = wk24[wk24["position"]=="RB"].copy()
    rb24["actual_points"] = rb24["fantasy_points_ppr"].astype(float)

    rb24 = add_rb_lag_features(rb24)
    rb24 = apply_encoders(rb24, encoders)

    backtest = rb24[rb24["week"].isin(backtest_weeks)].copy()
    X_bt = prepare_X(backtest, feature_cols)

    print("Generating backtest predictions...")
    preds_bt = {}
    shap_dict = {}
    for name, model in models.items():
        p = model.predict(X_bt)
        preds_bt[name] = p

        # SHAP: only for LightGBM (fast). Others = None to skip heavy compute.
        if name in SHAP_MODELS:
            base, vals = shap_for_lightgbm(model, X_bt)
        else:
            base, vals = (None, None)
        shap_dict[name] = (base, vals)

    # ensemble = simple mean
    if preds_bt:
        ensemble_bt = np.mean(np.vstack([v for v in preds_bt.values()]), axis=0)
        preds_bt["ensemble"] = ensemble_bt
        base_e, vals_e = ensemble_shap(shap_dict)
    else:
        ensemble_bt = np.array([])
        base_e, vals_e = (None, None)

    backtest_out = backtest[[
        "player_id","player_name","recent_team","position","season","week","actual_points"
    ]].copy()
    for name, v in preds_bt.items():
        backtest_out[f"predicted_points_{name}"] = v
    backtest_out["shap_rationale"] = top_rationales(X_bt, feature_cols, base_e, vals_e, k=3)

    # ---------------- Upcoming Week 1 (2025) ----------------
    print("Loading DK master sheet and building 2025 Wk1 rows...")
    
    # --- Build 2025 Week 1 from latest 2024 features ---
    # 1) Get training feature list
    with open(Path(model_dir) / "feature_schema.json", "r") as f:
        feature_schema = json.load(f)
    feature_names = feature_schema["columns"]

    # 2) Build full 2024 feature table (same function used for backtest) 
    #    and slice the *latest row per player*
    wb_2024 = nfl.import_weekly_data([2024])
    rb_2024 = wb_2024[wb_2024["position"] == "RB"].copy()
    feat_2024 = add_rb_lag_features(rb_2024)                # <-- your existing feature builder
    feat_2024 = apply_encoders(feat_2024, encoders)          # <-- add encoders to get encoded columns
    latest_2024 = (
        feat_2024.sort_values(["player_id","season","week"])
                 .groupby("player_id")
                 .tail(1)
    )

    # Keep only the columns the model expects (+ id)
    latest_slice = latest_2024[["player_id"] + [c for c in feature_names if c in latest_2024.columns]].copy()

    # 3) Read Master Sheet and prefer its player_id
    dk = pd.read_csv(master_sheet_path)
    dk_rb = dk[dk["Position"] == "RB"].copy()

    # If Master Sheet already has player_id, great; otherwise try crosswalk (optional)
    if "player_id" not in dk_rb.columns or dk_rb["player_id"].isna().all():
        try:
            # optional fallback only if you want it
            cw = pd.read_csv("../../data/processed/crosswalk_2025.csv")
            cw["name_norm"] = cw["player_name"].str.lower().str.strip()
            dk_rb["name_norm"] = dk_rb["Name"].str.lower().str.strip()
            dk_rb = dk_rb.merge(
                cw[["player_id","recent_team","name_norm"]],
                on=["name_norm"], how="left"
            )
        except Exception:
            pass  # keep going with whatever IDs we have

    # 4) Merge DK → latest feature slice by player_id
    wk1 = dk_rb.merge(latest_slice, on="player_id", how="left")

    # 5) Stamp meta fields used downstream
    wk1["season"] = upcoming_season
    wk1["week"] = upcoming_week
    wk1["season_type"] = "REG"
    wk1["position"] = "RB"
    wk1["recent_team"] = wk1.get("TeamAbbrev", wk1.get("recent_team", "UNK"))
    wk1["player_name"] = wk1.get("Name", wk1.get("player_name", "Unknown"))
    wk1["opponent_team"] = np.nan  # Will be filled by encoder with 'UNK'
    wk1["actual_points"] = np.nan
    wk1["dk_salary"] = wk1["Salary"].astype(float)

    # 6) Median-fill any lag features that are still missing (rookies, new signings)
    #    Use medians computed from *backtest* rows so distribution is realistic.
    bt_full = feat_2024[feature_names].copy()
    med = bt_full.median(numeric_only=True)
    for col in feature_names:
        if col not in wk1.columns:
            wk1[col] = med.get(col, 0.0)
    wk1[feature_names] = wk1[feature_names].fillna(med)

    # 7) Now prepare features and predict as usual
    wk1_feats = apply_encoders(wk1, encoders)
    X_up = prepare_X(wk1_feats, feature_names)
    
    print("Generating 2025 Wk1 predictions...")
    preds_up = {}
    shap_up_dict = {}
    for name, model in models.items():
        p = model.predict(X_up)
        preds_up[name] = p

        # SHAP: only for LightGBM (fast). Others = None to skip heavy compute.
        if name in SHAP_MODELS:
            b, v = shap_for_lightgbm(model, X_up)
        else:
            b, v = (None, None)
        shap_up_dict[name] = (b, v)

    if preds_up:
        ens_up = np.mean(np.vstack([v for v in preds_up.values()]), axis=0)
        preds_up["ensemble"] = ens_up
        base_up_e, vals_up_e = ensemble_shap(shap_up_dict)
    else:
        ens_up = np.array([])
        base_up_e, vals_up_e = (None, None)

    up_out = wk1[[
        "player_id","player_name","recent_team","position","season","week"
    ]].copy()
    for name, v in preds_up.items():
        up_out[f"predicted_points_{name}"] = v
    up_out["dk_salary"] = wk1["dk_salary"].fillna(0).astype(float)
    up_out["value"] = np.where(up_out["dk_salary"]>0, preds_up["ensemble"]/(up_out["dk_salary"]/1000.0), np.nan)
    up_out["actual_points"] = np.nan
    up_out["shap_rationale"] = top_rationales(X_up, feature_names, base_up_e, vals_up_e, k=3)
    
    # Add historical features for Dashboard.py enrichment
    # Targets (last 3 and 5 weeks)
    up_out["targets_l3"] = wk1.get("targets_avg_last_3", 0.0).fillna(0.0)
    up_out["targets_l5"] = wk1.get("targets_avg_last_5", 0.0).fillna(0.0)
    
    # Routes (last 3 and 5 weeks) - use targets as proxy if routes not available
    up_out["routes_l3"] = wk1.get("targets_avg_last_3", 0.0).fillna(0.0)  # Proxy for routes
    up_out["routes_l5"] = wk1.get("targets_avg_last_5", 0.0).fillna(0.0)  # Proxy for routes
    
    # Rush attempts (last 3 and 5 weeks)
    up_out["rush_att_l3"] = wk1.get("carries_avg_last_3", 0.0).fillna(0.0)
    up_out["rush_att_l5"] = wk1.get("carries_avg_last_5", 0.0).fillna(0.0)
    
    # Snaps (last 3 and 5 weeks) - use touches as proxy if snaps not available
    up_out["snaps_l3"] = wk1.get("touches_avg_last_3", 0.0).fillna(0.0)  # Proxy for snaps
    up_out["snaps_l5"] = wk1.get("touches_avg_last_5", 0.0).fillna(0.0)  # Proxy for snaps
    
    # Target share (last 3 weeks) - use touches as proxy
    up_out["target_share_l3"] = wk1.get("touches_avg_last_3", 0.0).fillna(0.0)  # Proxy for target share
    up_out["route_share_l3"] = wk1.get("touches_avg_last_3", 0.0).fillna(0.0)  # Proxy for route share
    
    # Red zone data (2024 season totals)
    up_out["rz_tgts_2024"] = wk1.get("targets_avg_last_10", 0.0).fillna(0.0) * 10  # Estimate from 10-week avg
    up_out["rz_rush_2024"] = wk1.get("carries_avg_last_10", 0.0).fillna(0.0) * 10  # Estimate from 10-week avg

    # ---------------- Combine & Save ----------------
    out = pd.concat([backtest_out, up_out], ignore_index=True)
    use_key = "predicted_points_ensemble" if "predicted_points_ensemble" in out.columns else list(preds_bt.keys())[0]
    out = out.sort_values(["season","week", use_key], ascending=[True, True, False])

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    print(f"✅ Saved {len(out)} rows to {output_csv}")

    # quick summary
    print("\n=== SUMMARY ===")
    # backtest metrics if available
    try:
        sub = backtest_out.dropna(subset=["actual_points"])
        y = sub["actual_points"].astype(float)
        yhat = sub["predicted_points_ensemble"].astype(float)
        mae = (y - yhat).abs().mean()
        rmse = np.sqrt(((y - yhat)**2).mean())
        corr = y.corr(yhat)
        print(f"Backtest (2024 w13–17): rows={len(sub)}, MAE={mae:.2f}, RMSE={rmse:.2f}, Corr={corr:.3f}")
    except Exception:
        print("Backtest metrics unavailable (missing ensemble or actuals).")

    wk1 = out[(out["season"]==upcoming_season) & (out["week"]==upcoming_week)]
    if "value" in wk1.columns:
        print(f"2025 Wk1 rows: {len(wk1)}, top value example:")
        print(wk1[["player_name","recent_team","dk_salary","value","predicted_points_ensemble"]].head(10).to_string(index=False))

    return out

def main():
    import argparse
    parser = argparse.ArgumentParser(description="RB Inference (QB/WR-style)")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--master-sheet", default=DEFAULT_MASTER_SHEET)
    parser.add_argument("--crosswalk", default=DEFAULT_CROSSWALK)
    parser.add_argument("--aliases", default=DEFAULT_ALIASES)
    parser.add_argument("--upcoming-season", type=int, default=2025)
    parser.add_argument("--upcoming-week", type=int, default=1)
    parser.add_argument("--output", default="rb_predictions.csv")
    args = parser.parse_args()

    run_inference(
        model_dir=args.model_dir,
        master_sheet_path=args.master_sheet,
        crosswalk_path=args.crosswalk,
        aliases_path=args.aliases,
        upcoming_season=args.upcoming_season,
        upcoming_week=args.upcoming_week,
        output_csv=args.output
    )

if __name__ == "__main__":
    main()
