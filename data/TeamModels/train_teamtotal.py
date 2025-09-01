#!/usr/bin/env python3
"""
Game Outcome Model w/ Vegas export + Fast Wins
- Score prediction (home/away) + calibrated win prob
- Vegas lines & model edges/picks in 2024 CSV
- Short-week & division-game flags
- Pace & environment proxies
- Team priors (shrunken per-team residual intercepts)
- Quantile heads for totals (P25/P50/P75)
- Optional SHAP (mean |SHAP|) saved to CSV

Usage:
  python train_teamtotal.py --mode=train    # Train and backtest on 2024
  python train_teamtotal.py --mode=infer --season=2025 --weeks=1  # Predict 2025 Week 1

Outputs in data/outputs/:
  predictions_2024.csv (train mode)
  predictions_<season>w<week>.csv (infer mode)
  metrics.txt
  shap_home_meanabs.csv / shap_away_meanabs.csv  (if shap is installed)
"""

import os, warnings, math, re, argparse
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

import nfl_data_py as nfl
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, accuracy_score, brier_score_loss, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

# ----------------------------
# CLI Setup
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train or infer team total predictions")
    parser.add_argument('--mode', choices=['train', 'infer'], default='train',
                       help='Mode: train (backtest 2024) or infer (predict future weeks)')
    parser.add_argument('--season', type=int, default=2025,
                       help='Season for inference (default: 2025)')
    parser.add_argument('--weeks', type=str, default='1',
                       help='Weeks to predict (e.g., "1", "1-3", "1-3,5,7-9")')
    parser.add_argument('--outdir', type=Path, default=Path('data/outputs'),
                       help='Output directory (default: data/outputs/)')
    parser.add_argument('--concat-out', type=Path,
                       help='Optional: concatenate all weeks into single CSV file')
    return parser.parse_args()

def parse_weeks(weeks_str):
    """Parse week selection string like "1", "1-3", "1-3,5,7-9" into list of weeks."""
    weeks = []
    for part in weeks_str.split(','):
        if '-' in part:
            start, end = map(int, part.split('-'))
            weeks.extend(range(start, end + 1))
        else:
            weeks.append(int(part))
    return sorted(list(set(weeks)))  # Remove duplicates and sort

# ----------------------------
# Helpers
# ----------------------------
DIV_MAP = {
    # AFC East
    "BUF":"AFC_E", "MIA":"AFC_E", "NE":"AFC_E", "NYJ":"AFC_E",
    # AFC North
    "BAL":"AFC_N", "CIN":"AFC_N", "CLE":"AFC_N", "PIT":"AFC_N",
    # AFC South
    "HOU":"AFC_S", "IND":"AFC_S", "JAX":"AFC_S", "JAC":"AFC_S", "TEN":"AFC_S",
    # AFC West
    "DEN":"AFC_W", "KC":"AFC_W", "LAC":"AFC_W", "LV":"AFC_W",
    # NFC East
    "DAL":"NFC_E", "NYG":"NFC_E", "PHI":"NFC_E", "WAS":"NFC_E",
    "WSH":"NFC_E",
    # NFC North
    "CHI":"NFC_N", "DET":"NFC_N", "GB":"NFC_N", "MIN":"NFC_N",
    # NFC South
    "ATL":"NFC_S", "CAR":"NFC_S", "NO":"NFC_S", "TB":"NFC_S",
    # NFC West
    "ARI":"NFC_W", "LA":"NFC_W", "LAR":"NFC_W", "SEA":"NFC_W", "SF":"NFC_W",
}

def safe_bool(x):
    return 1 if bool(x) else 0

def is_precip_from_string(s):
    if pd.isna(s): return 0
    s = str(s).lower()
    return 1 if any(w in s for w in ["rain","snow","showers","sleet","thunder","storms"]) else 0

# ----------------------------
# Feature Engineering Pipeline
# ----------------------------
def build_features(seasons, target_season=None, target_weeks=None):
    """
    Build features for training or inference.
    
    Args:
        seasons: List of seasons to use for historical data
        target_season: If provided, build features for this season's weeks
        target_weeks: If provided, list of weeks to build features for
    
    Returns:
        df: Feature DataFrame
        feature_columns: List of actual feature column names
    """
    print(f"Building features for seasons: {seasons}")
    
    # 1) Load schedules (REG)
    sched = nfl.import_schedules(seasons)
    sched = sched[sched["game_type"] == "REG"].copy()

    for c in ["game_id","season","week","gameday","home_team","away_team","home_score","away_score",
              "spread_line","total_line","roof","surface","weekday","game_weather","temp","wind"]:
        if c not in sched.columns: sched[c] = np.nan

    # --- Robust Vegas column mapping (fallbacks) ---
    # Some sources expose 'over_under_line' or 'line'; map into total_line/spread_line when needed.
    if "over_under_line" in sched.columns:
        if "total_line" not in sched.columns or sched["total_line"].isna().all():
            sched["total_line"] = sched["over_under_line"]
        else:
            sched["total_line"] = sched["total_line"].fillna(sched["over_under_line"])
    if "line" in sched.columns:
        sched["spread_line"] = sched.get("spread_line", np.nan)
        sched["spread_line"] = sched["spread_line"].fillna(sched["line"])

    sched["gameday"] = pd.to_datetime(sched["gameday"])

    # 2) Team-game long table for rolling form + rest & pace
    home = sched[["game_id","season","week","gameday","home_team","away_team","home_score","away_score"]].copy()
    home.rename(columns={"home_team":"team","away_team":"opp","home_score":"points_for","away_score":"points_against"}, inplace=True)
    home["is_home"] = 1

    away = sched[["game_id","season","week","gameday","home_team","away_team","home_score","away_score"]].copy()
    away.rename(columns={"away_team":"team","home_team":"opp","away_score":"points_for","home_score":"points_against"}, inplace=True)
    away["is_home"] = 0

    team_game = pd.concat([home, away], ignore_index=True)
    team_game.sort_values(["team","gameday"], inplace=True)

    # Rolling 3-game means (form proxies), shift to avoid leakage
    for col in ["points_for","points_against"]:
        team_game[f"{col}_l3"] = team_game.groupby("team")[col].shift(1).rolling(3, min_periods=1).mean()

    # Rest + short week
    team_game["prev_gameday"] = team_game.groupby("team")["gameday"].shift(1)
    team_game["rest_days"] = (team_game["gameday"] - team_game["prev_gameday"]).dt.days
    team_game["rest_days"] = team_game["rest_days"].fillna(10).clip(0, 30)
    team_game["short_week"] = (team_game["rest_days"] <= 4).astype(int)

    # Pace proxy (neutral seconds/play) from PBP (best effort)
    def add_neutral_pace(team_game, seasons):
        try:
            pbp = nfl.import_pbp_data(seasons)
            # Filter to normal offensive plays with a posteam and neutral situation
            pbp = pbp[~pbp["posteam"].isna()].copy()
            # neutral: abs(score_differential) <= 7 and qtr in 1-3 (avoid late-game antics)
            pbp = pbp[(pbp["qtr"]<=3) & (pbp["score_differential"].abs()<=7)]
            # compute sec/play using game_seconds_remaining differences per posteam per game
            pbp = pbp.sort_values(["game_id","posteam","game_seconds_remaining"], ascending=[True, True, False])
            pbp["next_sec"] = pbp.groupby(["game_id","posteam"])["game_seconds_remaining"].shift(1)
            pbp["sec_per_play"] = (pbp["next_sec"] - pbp["game_seconds_remaining"]).clip(lower=0, upper=60)
            pace_game = pbp.groupby(["game_id","posteam"])["sec_per_play"].mean().reset_index()
            pace_game.rename(columns={"posteam":"team","sec_per_play":"neutral_sec_play"}, inplace=True)

            out = team_game.merge(pace_game, on=["game_id","team"], how="left")
            out["neutral_sec_play_l3"] = out.groupby("team")["neutral_sec_play"].shift(1).rolling(3, min_periods=1).mean()
            return out
        except Exception as e:
            # fallback: use league median ~27s/play for neutral situations, no rolling variation
            out = team_game.copy()
            out["neutral_sec_play"] = 27.0
            out["neutral_sec_play_l3"] = 27.0
            return out

    team_game = add_neutral_pace(team_game, seasons)

    feat_team = team_game[[
        "game_id","team",
        "points_for_l3","points_against_l3",
        "rest_days","short_week","neutral_sec_play_l3"
    ]].copy()

    # Merge features back to wide schedule
    home_feat = feat_team.rename(columns={
        "team":"home_team",
        "points_for_l3":"home_pf_l3",
        "points_against_l3":"home_pa_l3",
        "rest_days":"home_rest",
        "short_week":"home_short_week",
        "neutral_sec_play_l3":"home_pace_l3"
    })
    away_feat = feat_team.rename(columns={
        "team":"away_team",
        "points_for_l3":"away_pf_l3",
        "points_against_l3":"away_pa_l3",
        "rest_days":"away_rest",
        "short_week":"away_short_week",
        "neutral_sec_play_l3":"away_pace_l3"
    })

    df = sched.merge(home_feat, on=["game_id","home_team"], how="left") \
              .merge(away_feat, on=["game_id","away_team"], how="left")

    # Fill NAs for early-season
    for c in ["home_pf_l3","home_pa_l3","home_rest","home_short_week","home_pace_l3",
              "away_pf_l3","away_pa_l3","away_rest","away_short_week","away_pace_l3"]:
        if c in df.columns:  # Only fill columns that exist
            if c.endswith("short_week"):
                df[c] = df[c].fillna(0)
            elif c.endswith("pace_l3"):
                df[c] = df[c].fillna(27.0)
            elif c.endswith("_rest"):
                # default rest ≈ 10 days if unknown
                df[c] = df[c].fillna(10)
            else:
                # pf_l3 / pa_l3 fallback: league-ish average points
                df[c] = df[c].fillna(23.0)
        else:
            print(f"Warning: Column {c} not found in dataframe")
            print(f"Available columns: {df.columns.tolist()}")

    # Division-game flag
    df["home_div"] = df["home_team"].map(DIV_MAP)
    df["away_div"] = df["away_team"].map(DIV_MAP)
    df["is_division_game"] = (df["home_div"] == df["away_div"]).astype(int)

    # Environment
    df["is_dome"] = df["roof"].astype(str).str.lower().isin(["dome","closed"]).astype(int)
    df["temp_num"] = pd.to_numeric(df["temp"], errors="coerce").fillna(df["temp"].median() if "temp" in df else 65)
    df["wind_num"] = pd.to_numeric(df["wind"], errors="coerce").fillna(df["wind"].median() if "wind" in df else 5)
    df["is_precip"] = df["game_weather"].apply(is_precip_from_string).astype(int)

    # 3) Features + One-hots
    onehot = pd.get_dummies(df[["home_team","away_team"]], prefix=["home","away"], dtype=int)

    # Print available columns after merge to debug feature selection
    print(f"Available columns after merge: {sorted(df.columns.tolist())}")
    
    # Base features (use actual column names present after merges)
    base_feats = [
        # Market
        "spread_line","total_line",
        # Rolling form + rest & pace (use actual column names)
        "home_pf_l3","home_pa_l3","home_rest_y","home_short_week","home_pace_l3",
        "away_pf_l3","away_pa_l3","away_rest_y","away_short_week","away_pace_l3",
        # Environment
        "is_division_game","is_dome","temp_num","wind_num","is_precip",
    ]
    
    # Verify base features; fail explicitly if any are missing
    missing_feats = [f for f in base_feats if f not in df.columns]
    if missing_feats:
        print("ERROR: Missing required features:", missing_feats)
        print("Available features:", sorted(df.columns.tolist()))
        raise KeyError(f"Missing required features: {missing_feats}")
    
    X_all = pd.concat([df[base_feats], onehot], axis=1).fillna(0)
    
    # Print final feature columns
    print(f"Final feature columns ({len(X_all.columns)}): {sorted(X_all.columns.tolist())}")
    
    return df, X_all

# ----------------------------
# Training Mode
# ----------------------------
def train_mode(args):
    """Train models and backtest on 2024 data."""
    print("=== TRAINING MODE ===")
    
    SEASONS = list(range(2014, 2025))  # 2014–2024 inclusive
    
    # Build features
    df, X_all = build_features(SEASONS)
    
    y_home = df["home_score"].astype(float)
    y_away = df["away_score"].astype(float)
    y_total = y_home + y_away

    train_idx = df["season"] < 2024
    test_idx  = df["season"] == 2024

    X_train, X_test = X_all[train_idx], X_all[test_idx]
    y_train_home, y_test_home = y_home[train_idx], y_home[test_idx]
    y_train_away, y_test_away = y_away[train_idx], y_away[test_idx]
    y_train_total, y_test_total = y_total[train_idx], y_total[test_idx]

    # 4) Train main score models
    lgb_params = dict(
        n_estimators=1200,
        learning_rate=0.02,
        max_depth=-1,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        min_child_samples=20,
        random_state=42
    )

    m_home = lgb.LGBMRegressor(**lgb_params)
    m_away = lgb.LGBMRegressor(**lgb_params)
    m_home.fit(X_train, y_train_home)
    m_away.fit(X_train, y_train_away)

    pred_home_tr = m_home.predict(X_train)
    pred_away_tr = m_away.predict(X_train)
    pred_home_te = m_home.predict(X_test)
    pred_away_te = m_away.predict(X_test)

    # 5) Team priors (shrunken residual intercepts)
    train_rows = df[train_idx].copy()
    train_rows["resid_home"] = y_train_home.values - pred_home_tr
    train_rows["resid_away"] = y_train_away.values - pred_away_tr

    # offense residuals for each team computed from both contexts
    off_rows = pd.DataFrame({
        "team": pd.concat([train_rows["home_team"], train_rows["away_team"]], ignore_index=True),
        "resid": pd.concat([train_rows["resid_home"], train_rows["resid_away"]], ignore_index=True)
    })
    grp = off_rows.groupby("team")["resid"].agg(["mean","size"]).reset_index()
    k = 8.0  # shrinkage strength
    grp["shrunken"] = grp["mean"] * (grp["size"] / (grp["size"] + k))
    team_off_bias = dict(zip(grp["team"], grp["shrunken"]))

    def bias_for(team):
        return team_off_bias.get(team, 0.0)

    # apply to test predictions
    pred_home_te = pred_home_te + df.loc[test_idx, "home_team"].map(bias_for).values
    pred_away_te = pred_away_te + df.loc[test_idx, "away_team"].map(bias_for).values

    # Non-negativity and reasonable caps
    pred_home_te = np.clip(pred_home_te, 0, 70)
    pred_away_te = np.clip(pred_away_te, 0, 70)

    # 6) Outcome model + calibration (Platt -> Isotonic)
    margin_tr = pred_home_tr - pred_away_tr
    y_win_tr = (y_train_home > y_train_away).astype(int)

    logit = LogisticRegression(solver="lbfgs")
    logit.fit(margin_tr.reshape(-1,1), y_win_tr)

    margin_te = pred_home_te - pred_away_te
    prob_raw_te = logit.predict_proba(margin_te.reshape(-1,1))[:,1]

    # Isotonic calibration on train (map raw prob -> calibrated prob)
    prob_raw_tr = logit.predict_proba(margin_tr.reshape(-1,1))[:,1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(prob_raw_tr, y_win_tr)
    prob_cal_te = iso.transform(prob_raw_te)

    # 7) Quantile heads for totals (P25/50/75)
    def quant_model(alpha):
        p = dict(lgb_params)
        p.update(dict(objective="quantile", alpha=alpha, n_estimators=1000, learning_rate=0.03))
        return lgb.LGBMRegressor(**p)

    mq25 = quant_model(0.25).fit(X_train, y_train_total)
    mq50 = quant_model(0.50).fit(X_train, y_train_total)
    mq75 = quant_model(0.75).fit(X_train, y_train_total)

    q25_te = mq25.predict(X_test)
    q50_te = mq50.predict(X_test)
    q75_te = mq75.predict(X_test)

    # 8) Metrics (2024)
    mae_home = mean_absolute_error(y_test_home, pred_home_te)
    mae_away = mean_absolute_error(y_test_away, pred_away_te)
    acc_scores = np.mean((pred_home_te > pred_away_te) == (y_test_home > y_test_away))
    brier_raw = brier_score_loss((y_test_home > y_test_away).astype(int), prob_raw_te)
    brier_cal = brier_score_loss((y_test_home > y_test_away).astype(int), prob_cal_te)

    # 9) Build 2024 output with Vegas & edges
    out = df[test_idx][["season","week","game_id","home_team","away_team",
                        "home_score","away_score","spread_line","total_line"]].copy()
    out.rename(columns={"home_score":"actual_home_score","away_score":"actual_away_score"}, inplace=True)

    out["pred_home_score"] = pred_home_te
    out["pred_away_score"] = pred_away_te
    out["pred_total"]  = out["pred_home_score"] + out["pred_away_score"]
    out["pred_margin"] = out["pred_home_score"] - out["pred_away_score"]

    out["actual_total"]  = out["actual_home_score"] + out["actual_away_score"]
    out["actual_margin"] = out["actual_home_score"] - out["actual_away_score"]

    # quantile heads
    out["pred_total_p25"] = q25_te
    out["pred_total_p50"] = q50_te
    out["pred_total_p75"] = q75_te

    # calibrated win prob
    out["home_win_prob_raw"] = prob_raw_te
    out["home_win_prob"]     = prob_cal_te
    out["home_win_pred"]     = (out["home_win_prob"] >= 0.5).astype(int)
    out["home_win_actual"]   = (out["actual_home_score"] > out["actual_away_score"]).astype(int)

    # edges vs Vegas
    out["edge_total"]  = out["pred_total"]  - out["total_line"]
    out["edge_spread"] = out["pred_margin"] - out["spread_line"]   # >0 : model leans Home ATS

    out["pick_total"]  = np.where(out["edge_total"] > 0, "Over", np.where(out["edge_total"] < 0, "Under", "Push"))
    out["pick_spread"] = np.where(out["edge_spread"] > 0, "Home", np.where(out["edge_spread"] < 0, "Away", "Push"))

    # what happened vs market
    out["cover_actual"] = np.where(out["actual_margin"] - out["spread_line"] > 0, "Home",
                            np.where(out["actual_margin"] - out["spread_line"] < 0, "Away", "Push"))
    out["total_actual"] = np.where(out["actual_total"] - out["total_line"] > 0, "Over",
                            np.where(out["actual_total"] - out["total_line"] < 0, "Under", "Push"))

    # add feature flags for analysis convenience
    out["home_short_week"] = df.loc[test_idx, "home_short_week"].values
    out["away_short_week"] = df.loc[test_idx, "away_short_week"].values
    out["is_division_game"] = df.loc[test_idx, "is_division_game"].values
    out["is_dome"] = df.loc[test_idx, "is_dome"].values
    out["temp_num"] = df.loc[test_idx, "temp_num"].values
    out["wind_num"] = df.loc[test_idx, "wind_num"].values
    out["is_precip"] = df.loc[test_idx, "is_precip"].values
    out["home_pace_l3"] = df.loc[test_idx, "home_pace_l3"].values
    out["away_pace_l3"] = df.loc[test_idx, "away_pace_l3"].values

    out_path = args.outdir/"predictions_2024.csv"
    out.to_csv(out_path, index=False)

    # 10) Write metrics & (optional) SHAP
    with open(args.outdir/"metrics.txt","w") as f:
        f.write("Score MAE (2024):\n")
        f.write(f"  Home MAE: {mae_home:.3f}\n  Away MAE: {mae_away:.3f}\n")
        f.write(f"Winners from scores (acc): {acc_scores:.3f}\n")
        f.write(f"Brier (raw): {brier_raw:.3f}\n")
        f.write(f"Brier (calibrated): {brier_cal:.3f}\n")

    # Optional SHAP export (skip if shap missing)
    try:
        import shap
        expl_home = shap.TreeExplainer(m_home)
        expl_away = shap.TreeExplainer(m_away)
        # sample for speed
        Xs = X_train.sample(min(5000, len(X_train)), random_state=42)
        shap_home = np.abs(expl_home.shap_values(Xs)).mean(axis=0)
        shap_away = np.abs(expl_away.shap_values(Xs)).mean(axis=0)
        pd.DataFrame({"feature": X_train.columns, "mean_abs_shap": shap_home}).sort_values("mean_abs_shap", ascending=False)\
            .to_csv(args.outdir/"shap_home_meanabs.csv", index=False)
        pd.DataFrame({"feature": X_train.columns, "mean_abs_shap": shap_away}).sort_values("mean_abs_shap", ascending=False)\
            .to_csv(args.outdir/"shap_away_meanabs.csv", index=False)
    except Exception:
        pass

    print(f"✅ Wrote {out_path}")
    print(f"✅ Metrics -> {args.outdir/'metrics.txt'}")
    
    return m_home, m_away, logit, iso, team_off_bias, X_all.columns.tolist()

# ----------------------------
# Inference Mode
# ----------------------------
def infer_mode(args):
    """Infer predictions for future weeks."""
    print(f"=== INFERENCE MODE ===")
    print(f"Season: {args.season}")
    print(f"Weeks: {args.weeks}")
    
    # Parse weeks
    target_weeks = parse_weeks(args.weeks)
    print(f"Parsed weeks: {target_weeks}")
    
    # Build features for training (use <=2024 data)
    train_seasons = list(range(2014, 2025))  # 2014-2024 for training
    print(f"Training on seasons: {train_seasons}")
    
    # Train models first
    m_home, m_away, logit, iso, team_off_bias, feature_columns = train_mode(args)
    
    # Now build features for target season/weeks
    print(f"\nBuilding features for {args.season} weeks {target_weeks}")
    all_seasons = list(range(2014, args.season + 1))  # Include target season for schedule
    df_target, X_target = build_features(all_seasons, args.season, target_weeks)
    
    # Filter to target weeks
    target_mask = (df_target['season'] == args.season) & (df_target['week'].isin(target_weeks))
    df_target_weeks = df_target[target_mask].copy()
    X_target_weeks = X_target[target_mask].copy()
    
    print(f"Target games: {len(df_target_weeks)}")
    
    # --- Align target matrix to the training feature set (one-hot categories may differ) ---
    # Ensure every training feature exists in target; add missing with 0 and order identically.
    X_target_weeks = X_target_weeks.reindex(columns=feature_columns, fill_value=0)
    
    # Make predictions
    pred_home = m_home.predict(X_target_weeks)
    pred_away = m_away.predict(X_target_weeks)
    
    # Apply team bias
    pred_home = pred_home + df_target_weeks["home_team"].map(team_off_bias).fillna(0).values
    pred_away = pred_away + df_target_weeks["away_team"].map(team_off_bias).fillna(0).values
    
    # Clip predictions
    pred_home = np.clip(pred_home, 0, 70)
    pred_away = np.clip(pred_away, 0, 70)
    
    # Calculate totals and margins
    pred_total = pred_home + pred_away
    pred_margin = pred_home - pred_away
    
    # Win probability
    margin_for_prob = pred_margin.reshape(-1, 1)
    prob_raw = logit.predict_proba(margin_for_prob)[:, 1]
    home_win_prob = iso.transform(prob_raw)
    
    # Quantile predictions for totals
    def quant_model(alpha):
        p = dict(
            n_estimators=1200,
            learning_rate=0.02,
            max_depth=-1,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=1.0,
            min_child_samples=20,
            random_state=42
        )
        p.update(dict(objective="quantile", alpha=alpha, n_estimators=1000, learning_rate=0.03))
        return lgb.LGBMRegressor(**p)
    
    # Train quantile models on full training data
    train_seasons = list(range(2014, 2025))
    df_train, X_train = build_features(train_seasons)
    y_train_total = df_train["home_score"].fillna(0) + df_train["away_score"].fillna(0)
    
    mq25 = quant_model(0.25).fit(X_train, y_train_total)
    mq50 = quant_model(0.50).fit(X_train, y_train_total)
    mq75 = quant_model(0.75).fit(X_train, y_train_total)
    
    # Align target to the quantile models' training columns too
    quant_train_cols = X_train.columns
    X_target_for_quant = X_target_weeks.reindex(columns=quant_train_cols, fill_value=0)
    pred_total_p25 = mq25.predict(X_target_for_quant)
    pred_total_p50 = mq50.predict(X_target_for_quant)
    pred_total_p75 = mq75.predict(X_target_for_quant)
    
    # Build output for each week
    all_week_outputs = []  # Store all weekly outputs for potential concatenation
    
    for week in target_weeks:
        week_mask = df_target_weeks['week'] == week
        if not week_mask.any():
            print(f"Warning: No games found for week {week}")
            continue
            
        df_week = df_target_weeks[week_mask].copy()
        pred_home_week = pred_home[week_mask]
        pred_away_week = pred_away[week_mask]
        pred_total_week = pred_total[week_mask]
        pred_margin_week = pred_margin[week_mask]
        home_win_prob_week = home_win_prob[week_mask]
        pred_total_p25_week = pred_total_p25[week_mask]
        pred_total_p50_week = pred_total_p50[week_mask]
        pred_total_p75_week = pred_total_p75[week_mask]
        
        # Create output DataFrame with exact column order
        out = pd.DataFrame({
            'season': df_week['season'],
            'week': df_week['week'],
            'game_id': df_week['game_id'],
            'home_team': df_week['home_team'],
            'away_team': df_week['away_team'],
            'actual_home_score': df_week['home_score'],  # Will be NaN for upcoming games
            'actual_away_score': df_week['away_score'],  # Will be NaN for upcoming games
            'pred_home_score': pred_home_week,
            'pred_away_score': pred_away_week,
            'spread_line': df_week['spread_line'],
            'total_line': df_week['total_line'],
            'pred_total': pred_total_week,
            'pred_margin': pred_margin_week,
            'pred_total_p25': pred_total_p25_week,
            'pred_total_p50': pred_total_p50_week,
            'pred_total_p75': pred_total_p75_week,
            'home_win_prob': home_win_prob_week,
            'edge_total': pred_total_week - df_week['total_line'],
            'edge_spread': pred_margin_week - df_week['spread_line'],
            'pick_total': np.where(pred_total_week > df_week['total_line'], "Over", 
                                 np.where(pred_total_week < df_week['total_line'], "Under", "Push")),
            'pick_spread': np.where(pred_margin_week > df_week['spread_line'], "Home",
                                  np.where(pred_margin_week < df_week['spread_line'], "Away", "Push"))
        })
        
        # Store for potential concatenation
        all_week_outputs.append(out)
        
        # Write weekly file
        week_str = f"{week:02d}"  # Zero-padded
        out_path = args.outdir / f"predictions_{args.season}w{week_str}.csv"
        out.to_csv(out_path, index=False)
        
        # Calculate coverage stats
        n_games = len(out)
        n_with_spread = out['spread_line'].notna().sum()
        n_with_total = out['total_line'].notna().sum()
        spread_coverage = n_with_spread / n_games * 100
        total_coverage = n_with_total / n_games * 100
        
        # Calculate stats
        mean_pred_total = out['pred_total'].mean()
        median_pred_total = out['pred_total'].median()
        mean_edge_total = out['edge_total'].mean()
        median_edge_total = out['edge_total'].median()
        mean_edge_spread = out['edge_spread'].mean()
        
        print(f"\n✅ Week {week}: {out_path}")
        print(f"   Games: {n_games}")
        print(f"   Coverage: {spread_coverage:.0f}% spread_line, {total_coverage:.0f}% total_line")
        print(f"   Mean pred_total: {mean_pred_total:.1f}")
        print(f"   Median pred_total: {median_pred_total:.1f}")
        print(f"   Mean edge_total: {mean_edge_total:.1f}")
        print(f"   Median edge_total: {median_edge_total:.1f}")
        print(f"   Mean edge_spread: {mean_edge_spread:.1f}")
        
        # Print sample
        print(f"   Sample predictions:")
        sample_cols = ['home_team', 'away_team', 'spread_line', 'total_line', 'pred_total', 'edge_total']
        print(out[sample_cols].head(3).to_string(index=False))
    
    # Handle concatenation if requested
    if args.concat_out and all_week_outputs:
        print(f"\n=== CONCATENATING WEEKS ===")
        concatenated = pd.concat(all_week_outputs, ignore_index=True)
        concatenated = concatenated.sort_values(['week', 'home_team'])
        
        # Write concatenated file
        args.concat_out.parent.mkdir(parents=True, exist_ok=True)
        concatenated.to_csv(args.concat_out, index=False)
        
        # Print concatenated summary
        total_games = len(concatenated)
        total_with_spread = concatenated['spread_line'].notna().sum()
        total_with_total = concatenated['total_line'].notna().sum()
        total_spread_coverage = total_with_spread / total_games * 100
        total_total_coverage = total_with_total / total_games * 100
        
        print(f"✅ Concatenated output: {args.concat_out}")
        print(f"   Total games: {total_games}")
        print(f"   Overall coverage: {total_spread_coverage:.0f}% spread_line, {total_total_coverage:.0f}% total_line")
        print(f"   Mean pred_total: {concatenated['pred_total'].mean():.1f}")
        print(f"   Mean edge_total: {concatenated['edge_total'].mean():.1f}")
        print(f"   Mean edge_spread: {concatenated['edge_spread'].mean():.1f}")

# ----------------------------
# Main
# ----------------------------
def main():
    args = parse_args()
    
    # Create output directory
    args.outdir.mkdir(parents=True, exist_ok=True)
    
    if args.mode == 'train':
        train_mode(args)
    elif args.mode == 'infer':
        infer_mode(args)
    else:
        print(f"Unknown mode: {args.mode}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
