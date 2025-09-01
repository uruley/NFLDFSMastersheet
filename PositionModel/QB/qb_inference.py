#!/usr/bin/env python3
"""
QB Model Inference Script (WR-style, no leakage)
- Past 5 weeks of 2024 (with actuals + predictions)
- 2025 Week 1 projections from 2024 lag stats (no placeholder stats)
- Joins DraftKings salaries + computes value
- SHAP explanations for ensemble
"""

import os
os.environ["JOBLIB_MULTIPROCESSING"] = "0"
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
import pickle, json, warnings
import re
from pathlib import Path
import nfl_data_py as nfl
import catboost as cb
import shap

warnings.filterwarnings("ignore")

# ---------------------------
# Helper Functions
# ---------------------------

def _norm_name(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()
    # remove punctuation & common suffixes
    s = re.sub(r"[.\'\-]", "", s)
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def _safe_read_csv(p):
    try:
        return pd.read_csv(p)
    except Exception:
        return None

class QBInferenceEngine:
    def __init__(self, model_dir="QB_Model"):
        self.model_dir = Path(model_dir)
        self.models = {}
        self.encoders = {}
        self.scaler = None
        self.feature_names = []
        self._load_models()
        self._load_encoders()
        self._load_scaler()
        self._load_schema()

    # ---------- loads ----------
    def _load_models(self):
        files = {
            "lightgbm": "lightgbm_model.pkl",
            "catboost": "catboost_model.pkl",
            "random_forest": "random_forest_model.pkl",
            "gradient_boosting": "gradient_boosting_model.pkl",
        }
        for name, f in files.items():
            path = self.model_dir / f
            if path.exists():
                with open(path, "rb") as fh:
                    self.models[name] = pickle.load(fh)
                print(f"✅ Loaded {name}")
            else:
                print(f"⚠️ Missing {f}")
        
        # Force single-thread for Windows stability
        def _force_single_thread(m):
            name = m.__class__.__name__.lower()
            # scikit-learn & LGBM wrappers accept n_jobs
            try: m.set_params(n_jobs=1)
            except: pass
            # LightGBM accepts num_threads (in addition to n_jobs)
            if "lgbm" in name or "lightgbm" in str(type(m)).lower():
                try: m.set_params(num_threads=1)
                except: pass
            # CatBoost uses thread_count
            if "catboost" in name:
                try: m.set_params(thread_count=1)
                except: pass
            return m

        for k in list(self.models.keys()):
            self.models[k] = _force_single_thread(self.models[k])

    def _load_encoders(self):
        p = self.model_dir / "encoders.pkl"
        if p.exists():
            with open(p, "rb") as f:
                self.encoders = pickle.load(f)
            print("✅ Loaded encoders")
        else:
            self.encoders = {}

    def _load_scaler(self):
        p = self.model_dir / "scaler.pkl"
        if p.exists():
            with open(p, "rb") as f:
                self.scaler = pickle.load(f)
            print("✅ Loaded scaler")
        else:
            self.scaler = None

    def _load_schema(self):
        p = self.model_dir / "feature_schema.json"
        with open(p, "r") as f:
            schema = json.load(f)
        self.feature_names = schema["columns"]
        print(f"✅ Loaded schema with {len(self.feature_names)} features")

    # ---------- feature builders (match training) ----------
    def create_lagged_features(self, df):
        df = df.sort_values(["player_id", "season", "week"])
        lag_features = [
            "completions","attempts","passing_yards","passing_tds","interceptions",
            "sacks","carries","rushing_yards","rushing_tds","target_share","air_yards_share"
        ]
        for w in [3,5,10]:
            for feat in lag_features:
                if feat in df.columns:
                    df[f"{feat}_avg_last_{w}"] = df.groupby("player_id")[feat].transform(
                        lambda x: x.shift(1).rolling(window=w, min_periods=1).mean()
                    )
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

    def create_derived_features(self, df):
        df["completion_rate_last_3"] = (df["completions_avg_last_3"] / df["attempts_avg_last_3"]).fillna(0.6).clip(0,0.85)
        df["yards_per_attempt_last_3"] = (df["passing_yards_avg_last_3"] / df["attempts_avg_last_3"]).fillna(7.0).clip(0,12)
        df["td_rate_last_3"] = (df["passing_tds_avg_last_3"] / df["attempts_avg_last_3"]).fillna(0.04).clip(0,0.12)
        df["int_rate_last_3"] = (df["interceptions_avg_last_3"] / df["attempts_avg_last_3"]).fillna(0.02).clip(0,0.1)
        df["yards_per_rush_last_3"] = (df["rushing_yards_avg_last_3"] / df["carries_avg_last_3"]).fillna(4.0).clip(0,10)

        df["early_season"] = (df["week"] <= 4).astype(int)
        df["mid_season"] = ((df["week"] > 4) & (df["week"] <= 12)).astype(int)
        df["late_season"] = (df["week"] > 12).astype(int)
        df["week_progression"] = df["week"] / 18
        df["is_consistent"] = (df["fantasy_points_std_last_5"] < 8).astype(int)
        return df
    
    def add_defense_allowed_features(self, weekly_all):
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

    def encode_categoricals(self, df):
        # team
        if "team" in self.encoders:
            enc = self.encoders["team"]
            df["team_encoded"] = df["recent_team"].map(
                lambda x: enc.transform([x])[0] if x in enc.classes_ else enc.transform(["UNK"])[0]
            )
        else:
            df["team_encoded"] = 0
        # opponent
        if "opponent" in self.encoders:
            enc = self.encoders["opponent"]
            df["opponent_encoded"] = df["opponent_team"].map(
                lambda x: enc.transform([x])[0] if x in enc.classes_ else enc.transform(["UNK"])[0]
            )
        else:
            df["opponent_encoded"] = 0
        # season_type
        if "season_type" in self.encoders:
            enc = self.encoders["season_type"]
            df["season_type_encoded"] = df["season_type"].fillna("REG").map(
                lambda x: enc.transform([x])[0] if x in enc.classes_ else enc.transform(["REG"])[0]
            )
        else:
            df["season_type_encoded"] = 0

        return df

    def prepare_features(self, df):
        # ensure all columns exist
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = np.nan

        if self.scaler is not None and hasattr(self.scaler, "mean_"):
            # fill with the same means the scaler saw at training time
            fill_map = dict(zip(self.feature_names, self.scaler.mean_))
            X_df = df[self.feature_names].copy().fillna(fill_map)
            X = self.scaler.transform(X_df.values)
        else:
            X_df = df[self.feature_names].copy().fillna(0)
            X = X_df.values

        return X, df

    # ---------- SHAP ----------
    def shap_for_lightgbm(self, model, X):
        sv = model.predict(X, pred_contrib=True)
        return sv[:, -1], sv[:, :-1]

    def shap_for_catboost(self, model, X):
        pool = cb.Pool(X)
        sv = model.get_feature_importance(type="ShapValues", data=pool)
        return sv[:, -1], sv[:, :-1]

    def shap_for_sklearn_tree(self, model, X):
        explainer = shap.TreeExplainer(model)
        vals = explainer.shap_values(X)
        if isinstance(vals, list):
            vals = vals[0]
        base = np.full(X.shape[0], explainer.expected_value)
        return base, vals

    def ensemble_shap(self, shap_dict):
        bases = [b for b, v in shap_dict.values()]
        vals  = [v for b, v in shap_dict.values()]
        base_e = np.mean(np.vstack(bases), axis=0)
        vals_e = np.mean(np.stack(vals, axis=0), axis=0)
        return base_e, vals_e

    def top_rationale_rows(self, X, feature_names, base, shap_vals, k=3, decimals=2):
        out = []
        n, d = shap_vals.shape
        for i in range(n):
            contrib = shap_vals[i]
            order = np.argsort(-np.abs(contrib))
            tops = []
            for j in order[:k]:
                sign = "↑" if contrib[j] > 0 else "↓"
                tops.append(f"{feature_names[j]} {sign} {abs(contrib[j]):.{decimals}f}")
            pred = base[i] + contrib.sum()
            out.append(f"~{pred:.1f} pts (base {base[i]:.1f}) • " + ", ".join(tops))
        return out

    # ---------- Main ----------
    def run_inference(
        self,
        seasons=[2024],
        past_weeks=[13,14,15,16,17],
        upcoming_season=2025,
        upcoming_week=1,
        master_sheet_path=None,
        output_file="outputs/projections/qb_predictions.csv"
    ):
        print("=== QB Model Inference ===")

        # 1) load historical seasons (e.g., 2024) and build lag features
        hist = nfl.import_weekly_data(seasons)
        hist = hist[hist["position"] == "QB"].copy()
        hist = hist[hist["attempts"] >= 10]
        hist = self.create_lagged_features(hist)
        hist = self.create_derived_features(hist)
        
        # Add defense-vs-QB allowed features
        def_feats = self.add_defense_allowed_features(hist)
        hist = hist.merge(def_feats, left_on=["season","week","opponent_team"], right_on=["season","week","def_team"], how="left")
        hist.drop(columns=["def_team"], inplace=True)

        # 2) past weeks slice (with actuals)
        past = hist[hist["week"].isin(past_weeks)].copy()
        past["actual_points"] = past["fantasy_points_ppr"]

        # ===== UPCOMING FROM DK (robust, uses master_sheet_2025 with player_id) =====
        upcoming = pd.DataFrame()
        if master_sheet_path:
            dk = pd.read_csv(master_sheet_path)

            # 1) DK QB slice with salary + id
            dk_qb = dk[dk["Position"] == "QB"].copy()
            # normalize column casing
            for c in ["Name","TeamAbbrev","Salary","player_id","join_key","name_team_key"]:
                if c not in dk_qb.columns:
                    dk_qb[c] = np.nan

            dk_qb["dk_name"]   = dk_qb["Name"].astype(str)
            dk_qb["dk_team"]   = dk_qb["TeamAbbrev"].astype(str)
            dk_qb["dk_salary"] = pd.to_numeric(dk_qb["Salary"], errors="coerce")
            dk_qb["player_id"] = dk_qb["player_id"].astype(str)

            # Helper to normalize names for fallback
            def _norm_name(s: str) -> str:
                if not isinstance(s, str): return ""
                s = s.lower().strip()
                s = re.sub(r"[.\'\-]", "", s)
                s = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", s)
                s = re.sub(r"\s+", " ", s)
                return s.strip()

            # 2) Build latest 2024 features per player_id from hist
            latest = (
                hist.sort_values(["player_id","season","week"])
                    .groupby("player_id")
                    .tail(1)
                    .copy()
            )

            # Fallback join_key on the hist side (name|team|QB) for rows missing id
            latest["name_norm"] = latest["player_name"].astype(str).map(_norm_name)
            latest["join_key_fb"] = latest["name_norm"] + "|" + latest["recent_team"].astype(str) + "|QB"

            # 3) Primary join: DK -> latest by player_id
            has_id_mask = dk_qb["player_id"].notna() & (dk_qb["player_id"].str.len() > 0)
            dk_with_feats = dk_qb.merge(
                latest, on="player_id", how="left", suffixes=("_dk","_hist")
            )

            # 4) Fallback for rows without player_id or where id didn't hit a 2024 row (rookies/new QBs)
            need_fb = dk_with_feats["player_name"].isna()
            if need_fb.any():
                # build join_key on DK side
                dk_with_feats.loc[:, "name_norm"] = dk_with_feats["dk_name"].astype(str).map(_norm_name)
                dk_with_feats.loc[:, "join_key_fb"] = dk_with_feats["name_norm"] + "|" + dk_with_feats["dk_team"] + "|QB"

                fb_map = latest[["join_key_fb","player_id","player_name","recent_team"]].drop_duplicates()
                fb_join = dk_with_feats.loc[need_fb, ["join_key_fb"]].merge(
                    fb_map, on="join_key_fb", how="left"
                )

                # fill from fallback where available
                for col in ["player_id","player_name","recent_team"]:
                    dk_with_feats.loc[need_fb, col] = dk_with_feats.loc[need_fb, col].fillna(pd.Series(fb_join[col].values, index=dk_with_feats.loc[need_fb].index))

            # 5) Stamp upcoming fields
            dk_with_feats["season"] = upcoming_season
            dk_with_feats["week"] = upcoming_week
            dk_with_feats["season_type"] = "REG"
            dk_with_feats["position"] = "QB"
            dk_with_feats["actual_points"] = np.nan
            dk_with_feats["dk_salary"] = dk_with_feats["dk_salary"].fillna(0).astype(float)

            # Prefer hist player_name when available; otherwise fall back to DK name
            if "player_name" in dk_with_feats.columns:
                dk_with_feats["player_name"] = dk_with_feats["player_name"].fillna(dk_with_feats["dk_name"])
            else:
                dk_with_feats["player_name"] = dk_with_feats["dk_name"]

            # 6) Keep just one name column and clean up
            upcoming = dk_with_feats
            # optional: drop helper cols
            drop_helpers = ["name_norm","join_key_fb"]
            for c in drop_helpers:
                if c in upcoming.columns:
                    upcoming.drop(columns=c, inplace=True)

            # 7) Logging
            matched_cnt = int(upcoming["player_name"].notna().sum())
            total_cnt = int(len(upcoming))
            print(f"DK→2024 link (by player_id, with fallback): {matched_cnt}/{total_cnt} rows have features")
            
            # Add defense features to upcoming data (only where opponent_team is known)
            if len(upcoming) > 0:
                upcoming_def_feats = self.add_defense_allowed_features(hist)
                has_opp = upcoming["opponent_team"].notna()
                if has_opp.any():
                    upcoming.loc[has_opp, :] = upcoming.loc[has_opp, :].merge(
                        upcoming_def_feats,
                        left_on=["season","week","opponent_team"],
                        right_on=["season","week","def_team"],
                        how="left"
                    )
                    if "def_team" in upcoming.columns:
                        upcoming.drop(columns=["def_team"], inplace=True)
                # fill unknown opponents with league-average per season
                for col in ["def_qb_fp_allowed_last_5","def_qb_fp_allowed_last_10"]:
                    if col not in upcoming.columns:
                        upcoming[col] = np.nan
                    season_means = hist.groupby("season")[col.replace("def_","")].mean() if col.replace("def_","") in hist.columns else hist.groupby("season")["fantasy_points_ppr"].mean()
                    upcoming[col] = upcoming[col].fillna(upcoming["season"].map(season_means))

        # 4) combine and (re)derive features → encode → prepare matrix
        data = pd.concat([past, upcoming], ignore_index=True, sort=False)
        data = self.create_derived_features(data)
        data = self.encode_categoricals(data)
        X, data = self.prepare_features(data)

        # 5) predict per model + ensemble
        preds, shap_dict = {}, {}
        for name, model in self.models.items():
            yhat = model.predict(X)
            preds[name] = np.clip(yhat, 0, 50)
            if name == "lightgbm":
                shap_dict[name] = self.shap_for_lightgbm(model, X)
            elif name == "catboost":
                shap_dict[name] = self.shap_for_catboost(model, X)
            else:
                shap_dict[name] = self.shap_for_sklearn_tree(model, X)

        # Weighted ensemble by holdout R²
        import json, pathlib, math
        
        metrics_path = pathlib.Path("QB_Model/metrics.json")
        if metrics_path.exists():
            m = json.loads(metrics_path.read_text())
            r2s = {
                "lightgbm": max(0.0, float(m["lightgbm"]["holdout_r2"])),
                "catboost": max(0.0, float(m["catboost"]["holdout_r2"])),
                "random_forest": max(0.0, float(m["random_forest"]["holdout_r2"])),
                "gradient_boosting": max(0.0, float(m["gradient_boosting"]["holdout_r2"])),
            }
        else:
            r2s = {"lightgbm": 1, "catboost": 1, "random_forest": 1, "gradient_boosting": 1}
        
        Z = sum(r2s.values()) or 1.0
        w = {k: v / Z for k, v in r2s.items()}
        
        # Calculate weighted ensemble
        data["predicted_points_ensemble"] = (
            w["lightgbm"]          * preds["lightgbm"] +
            w["catboost"]          * preds["catboost"] +
            w["random_forest"]     * preds["random_forest"] +
            w["gradient_boosting"] * preds["gradient_boosting"]
        )
        
        # Add individual model predictions
        for name, arr in preds.items():
            data[f"predicted_points_{name}"] = arr

        # 6) SHAP ensemble rationale (before cold-start shrink to maintain row alignment)
        base_e, vals_e = self.ensemble_shap(shap_dict)
        data["shap_rationale"] = self.top_rationale_rows(X, self.feature_names, base_e, vals_e)

        # 7) Cold-start shrink at inference

        def _compute_player_means(history_df):
            # history_df must include: player_id, fantasy_points_ppr, season, week, attempts
            hist = history_df.sort_values(["player_id","season","week"]).copy()
            hist["eligible"] = (hist["attempts"] >= 10).astype(int)
            # last3 mean for each player excluding current row; we'll merge values by last known week
            hist["fp_last3"] = hist.groupby("player_id")["fantasy_points_ppr"].transform(
                lambda x: x.shift(1).rolling(3, min_periods=1).mean()
            )
            # last10 count of eligible games
            hist["g_last10"] = hist.groupby("player_id")["eligible"].transform(
                lambda x: x.shift(1).rolling(10, min_periods=1).sum()
            )
            # Keep last row per (player, season, week) for merge keys
            return hist[["player_id","season","week","fp_last3","g_last10"]]

        # Apply cold-start shrink
        if len(past) > 0:  # Only if we have historical data
            # Get historical data for cold-start calculation
            df_hist = past.copy()
            global_mean = float(df_hist["fantasy_points_ppr"].clip(lower=0, upper=50).mean())
            
            pm = _compute_player_means(df_hist)
            data = data.merge(pm, on=["player_id","season","week"], how="left")
            
            data["fp_last3"].fillna(global_mean, inplace=True)
            data["g_last10"].fillna(0, inplace=True)
            
            prior = 0.6 * data["fp_last3"] + 0.4 * global_mean
            w = (data["g_last10"] / 4.0).clip(lower=0.0, upper=1.0)
            
            data["predicted_points_final"] = (w * data["predicted_points_ensemble"] + (1 - w) * prior).clip(lower=0, upper=50)
            
            # Also export predicted_points_final as the value column you feed into downstream tools
            data.rename(columns={"predicted_points_final":"predicted_points"}, inplace=True)
        else:
            # If no historical data, use ensemble as final prediction
            data["predicted_points"] = data["predicted_points_ensemble"]

        # 8) value from DK salary if present
        if "dk_salary" in data.columns:
            data["value"] = data["predicted_points"] / (data["dk_salary"] / 1000)
        else:
            data["dk_salary"] = 0
            data["value"] = np.nan

        # 8) output cols
        cols = [
            "player_id","player_name","recent_team","position","season","week",
            "actual_points","dk_salary","value","predicted_points_ensemble","predicted_points",
            "predicted_points_lightgbm","predicted_points_catboost",
            "predicted_points_random_forest","predicted_points_gradient_boosting",
            "shap_rationale"
        ]
        # ensure presence
        for c in cols:
            if c not in data.columns:
                data[c] = np.nan

        out = data[cols].copy()
        
        # Keep the highest predicted_points per (player_id, season, week)
        out = (out.sort_values(["player_id","season","week","predicted_points"], ascending=[True,True,True,False])
               .drop_duplicates(subset=["player_id","season","week"], keep="first")
               .reset_index(drop=True))
        
        # ensure output dir exists, then write
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index=False)
        print(f"✅ Saved predictions to {output_file}")
        return out


def main():
    engine = QBInferenceEngine("QB_Model")
    out = engine.run_inference(
        seasons=[2024],
        past_weeks=[13,14,15,16,17],
        upcoming_season=2025,
        upcoming_week=1,
        master_sheet_path=r"C:\Users\ruley\NFLDFSMasterSheet\data\processed\master_sheet_2025.csv",
        output_file="outputs/projections/qb_predictions.csv"
    )
    if out is not None:
        print(out.head(15).to_string(index=False))

if __name__ == "__main__":
    main()
