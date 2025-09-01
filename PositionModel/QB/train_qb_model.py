#!/usr/bin/env python3
"""
QB Model Training Script (WR-style, no log transform)
- Uses lagged features + efficiency + context
- Trains ensemble (LightGBM, CatBoost, RandomForest, GradientBoosting)
- Target = fantasy_points_ppr (clipped at 50)
- Includes time-series cross-validation (CV) for realistic metrics
"""

import os
os.environ["JOBLIB_MULTIPROCESSING"] = "0"
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
from pathlib import Path
import nfl_data_py as nfl

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import lightgbm as lgb
import catboost as cb
import pickle, json, warnings

warnings.filterwarnings("ignore")

# ---------------------------
# Data Loading & Feature Engineering
# ---------------------------

def load_qb_data(seasons=[2020, 2021, 2022, 2023, 2024]):
    print(f"Loading QB data for seasons: {seasons}")
    weekly = nfl.import_weekly_data(seasons)
    df = weekly[weekly["position"] == "QB"].copy()
    df = df[df["attempts"] >= 10].copy()  # filter out low-usage/inactive weeks
    print(f"Loaded {len(df)} QB weekly rows")
    
    # Add defense-vs-QB allowed features
    def_feats = add_defense_allowed_features(weekly)
    df = df.merge(def_feats, left_on=["season","week","opponent_team"], right_on=["season","week","def_team"], how="left")
    df.drop(columns=["def_team"], inplace=True)
    
    return df

def add_defense_allowed_features(weekly_all):
    qb = weekly_all[weekly_all["position"] == "QB"][["season","week","opponent_team","fantasy_points_ppr"]].copy()
    qb.rename(columns={"opponent_team":"def_team","fantasy_points_ppr":"qb_fp_allowed"}, inplace=True)
    qb = qb.sort_values(["def_team","season","week"])
    for w in (5, 10):
        qb[f"def_qb_fp_allowed_last_{w}"] = (
            qb.groupby("def_team")["qb_fp_allowed"]
              .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        )
    qb = qb.drop(columns=["qb_fp_allowed"])
    return qb

def create_lagged_features(df):
    """Add rolling averages with shift(1) to avoid leakage."""
    df = df.sort_values(["player_id", "season", "week"])
    lag_features = [
        "completions","attempts","passing_yards","passing_tds","interceptions",
        "sacks","carries","rushing_yards","rushing_tds","target_share","air_yards_share"
    ]
    for w in [3, 5, 10]:
        for feat in lag_features:
            if feat in df.columns:
                df[f"{feat}_avg_last_{w}"] = (
                    df.groupby("player_id")[feat]
                      .transform(lambda x: x.shift(1).rolling(window=w, min_periods=1).mean())
                )

    # fantasy history (shift(1) roll; last_10 uses 10-game window, not shift(10))
    df["fantasy_points_avg_last_3"] = df.groupby("player_id")["fantasy_points_ppr"].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
    )
    df["fantasy_points_avg_last_10"] = df.groupby("player_id")["fantasy_points_ppr"].transform(
        lambda x: x.shift(1).rolling(window=10, min_periods=1).mean()
    )
    df["fantasy_points_std_last_5"] = df.groupby("player_id")["fantasy_points_ppr"].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=2).std()
    )
    return df

def create_derived_features(df):
    """Add efficiency & context features computed from lagged stats only."""
    df["completion_rate_last_3"] = (df["completions_avg_last_3"] / df["attempts_avg_last_3"]).fillna(0.6).clip(0, 0.85)
    df["yards_per_attempt_last_3"] = (df["passing_yards_avg_last_3"] / df["attempts_avg_last_3"]).fillna(7.0).clip(0, 12)
    df["td_rate_last_3"] = (df["passing_tds_avg_last_3"] / df["attempts_avg_last_3"]).fillna(0.04).clip(0, 0.12)
    df["int_rate_last_3"] = (df["interceptions_avg_last_3"] / df["attempts_avg_last_3"]).fillna(0.02).clip(0, 0.1)
    df["yards_per_rush_last_3"] = (df["rushing_yards_avg_last_3"] / df["carries_avg_last_3"]).fillna(4.0).clip(0, 10)

    df["early_season"] = (df["week"] <= 4).astype(int)
    df["mid_season"] = ((df["week"] > 4) & (df["week"] <= 12)).astype(int)
    df["late_season"] = (df["week"] > 12).astype(int)
    df["week_progression"] = df["week"] / 18
    df["is_consistent"] = (df["fantasy_points_std_last_5"] < 8).astype(int)
    return df

def prepare_features():
    """Master feature list — MUST match inference."""
    return [
        "completions_avg_last_3","completions_avg_last_5","completions_avg_last_10",
        "attempts_avg_last_3","attempts_avg_last_5","attempts_avg_last_10",
        "passing_yards_avg_last_3","passing_yards_avg_last_5","passing_yards_avg_last_10",
        "passing_tds_avg_last_3","passing_tds_avg_last_5","passing_tds_avg_last_10",
        "interceptions_avg_last_3","interceptions_avg_last_5",
        "sacks_avg_last_3","sacks_avg_last_5",
        "carries_avg_last_3","carries_avg_last_5","carries_avg_last_10",
        "rushing_yards_avg_last_3","rushing_yards_avg_last_5","rushing_yards_avg_last_10",
        "rushing_tds_avg_last_3","rushing_tds_avg_last_5","rushing_tds_avg_last_10",
        "fantasy_points_avg_last_3","fantasy_points_avg_last_10","fantasy_points_std_last_5",
        "completion_rate_last_3","yards_per_attempt_last_3","td_rate_last_3","int_rate_last_3","yards_per_rush_last_3",
        "week","early_season","mid_season","late_season","week_progression","is_consistent",
        "team_encoded","opponent_encoded","season_type_encoded",
        "def_qb_fp_allowed_last_5","def_qb_fp_allowed_last_10"
    ]

def encode_categoricals(df, feature_cols):
    """
    Label-encode categoricals, and guarantee an 'UNK' bucket exists so inference
    can safely map unseen teams/opponents/season types.
    """
    encoders = {}

    def fit_with_unk(series, unk_label):
        classes = pd.Index([unk_label]).append(pd.Index(series.dropna().unique())).unique()
        le = LabelEncoder()
        le.fit(classes)
        return le

    # recent_team -> team_encoded
    if "recent_team" in df.columns:
        enc = fit_with_unk(df["recent_team"], "UNK")
        df["team_encoded"] = enc.transform(df["recent_team"].fillna("UNK"))
        encoders["team"] = enc
    else:
        df["team_encoded"] = 0

    # opponent_team -> opponent_encoded
    if "opponent_team" in df.columns:
        enc = fit_with_unk(df["opponent_team"], "UNK")
        df["opponent_encoded"] = enc.transform(df["opponent_team"].fillna("UNK"))
        encoders["opponent"] = enc
    else:
        df["opponent_encoded"] = 0

    # season_type -> season_type_encoded
    if "season_type" in df.columns:
        enc = fit_with_unk(df["season_type"], "REG")
        df["season_type_encoded"] = enc.transform(df["season_type"].fillna("REG"))
        encoders["season_type"] = enc
    else:
        df["season_type_encoded"] = 0

    return df, feature_cols, encoders

# ---------------------------
# Training + CV
# ---------------------------

def train_models(df, feature_cols, target_col="fantasy_points_ppr"):
    y = df[target_col].clip(lower=0, upper=50)
    X = df[feature_cols].fillna(df[feature_cols].mean()).values.astype(float)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    models = {
        "lightgbm": lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, random_state=42),
        "catboost": cb.CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6, verbose=0, random_seed=42),
        "random_forest": RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42),
    }

    trained, metrics = {}, {}

    # --- explicit time holdout: train <= W12 of 2024; validate on 2024 W13+ ---
    holdout_train = ((df["season"] < 2024) | ((df["season"] == 2024) & (df["week"] <= 12)))
    holdout_val   = ((df["season"] == 2024) & (df["week"] >= 13))

    X_train = Xs[holdout_train.values]
    y_train = y[holdout_train.values]
    X_val   = Xs[holdout_val.values]
    y_val   = y[holdout_val.values]

    for name, model in models.items():
        print(f"Training {name}...")
        # fit on full for artifacts
        model.fit(Xs, y)
        preds = model.predict(Xs)
        train_rmse = mean_squared_error(y, preds, squared=False)
        train_mae  = mean_absolute_error(y, preds)
        train_r2   = r2_score(y, preds)

        # holdout eval
        mh = model.__class__(**model.get_params())
        mh.fit(X_train, y_train)
        y_hat = mh.predict(X_val)
        val_rmse = mean_squared_error(y_val, y_hat, squared=False)
        val_mae  = mean_absolute_error(y_val, y_hat)
        val_r2   = r2_score(y_val, y_hat)

        metrics[name] = {
            "train_rmse": float(train_rmse), "train_mae": float(train_mae), "train_r2": float(train_r2),
            "holdout_rmse": float(val_rmse), "holdout_mae": float(val_mae), "holdout_r2": float(val_r2),
            "holdout_detail": "train: 2020-2024W12, val: 2024W13+"
        }
        print(f"  {name}: Train R²={train_r2:.3f} | Holdout R²={val_r2:.3f}")
        trained[name] = model

    return trained, feature_cols, scaler, metrics

def save_artifacts(models, encoders, feature_cols, scaler, metrics, out_dir="QB_Model"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        with open(out / f"{name}_model.pkl", "wb") as f:
            pickle.dump(model, f)
    with open(out / "encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)
    with open(out / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(out / "feature_schema.json", "w") as f:
        json.dump({"columns": feature_cols}, f, indent=2)
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("✅ Saved all artifacts")

def main():
    df = load_qb_data()
    df = create_lagged_features(df)
    df = create_derived_features(df)
    
    # IMPORTANT: build lags per-player first, THEN sort globally for time-based eval
    df = df.sort_values(["season", "week"]).reset_index(drop=True)
    
    features = prepare_features()
    df, features, encoders = encode_categoricals(df, features)
    models, feature_cols, scaler, metrics = train_models(df, features, "fantasy_points_ppr")
    save_artifacts(models, encoders, feature_cols, scaler, metrics, "QB_Model")
    print("🎯 QB training with legitimate time-based holdout complete!")

if __name__ == "__main__":
    main()
