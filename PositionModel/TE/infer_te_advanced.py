#!/usr/bin/env python3
"""
Advanced TE Model Inference Script (Fixed for Feature Mismatch and Multi-Week Support)
Works with ensemble models (LightGBM, CatBoost, Random Forest, Gradient Boosting)
Uses lagged features, proper scaling, and ensures 5+ past weeks
"""
import pandas as pd
import numpy as np
import pickle
import json
import os
import re
from pathlib import Path
import nfl_data_py as nfl
from sklearn.preprocessing import LabelEncoder, StandardScaler
import catboost as cb
import shap
import warnings
warnings.filterwarnings('ignore')

# --- Windows-safe thread env vars + small helpers ---
# Avoid Windows/joblib stalls when counting cores / spawning
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

TEAM_MAP = {"JAC":"JAX","LA":"LAR","LAR":"LAR","STL":"LAR","OAK":"LVR","LV":"LVR","WSH":"WAS","WAS":"WAS"}
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

# --- TE lag builder (parallel to RB) ---
TE_LAG_BASE = [
    "receptions","targets","receiving_yards","receiving_tds",
    "carries","rushing_yards","fantasy_points_ppr"
]
TE_WINDOWS = (3,5,10)

def add_te_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["player_id","season","week"]).copy()
    
    # Create lag features for base columns
    for col in TE_LAG_BASE:
        if col not in df.columns:
            df[col] = 0.0
        # Create lag column (previous week)
        df[f"{col}_lag"] = df.groupby("player_id")[col].shift(1).fillna(0)
        
        # Create rolling averages
        for k in TE_WINDOWS:
            df[f"{col}_avg_last_{k}"] = (
                df.groupby("player_id")[col].shift(1).rolling(k, min_periods=1).mean()
            )

    # Create derived features from lag data
    for k in TE_WINDOWS:
        rc = df.get(f"receptions_avg_last_{k}", pd.Series(0.0, index=df.index)).fillna(0)
        tg = df.get(f"targets_avg_last_{k}", pd.Series(0.0, index=df.index)).fillna(0)
        rcy = df.get(f"receiving_yards_avg_last_{k}", pd.Series(0.0, index=df.index)).fillna(0)
        ca  = df.get(f"carries_avg_last_{k}", pd.Series(0.0, index=df.index)).fillna(0)
        ry  = df.get(f"rushing_yards_avg_last_{k}", pd.Series(0.0, index=df.index)).fillna(0)

        df[f"yards_per_reception_last_{k}"] = np.where(rc > 0, rcy/rc, 0.0)
        df[f"yards_per_target_last_{k}"]    = np.where(tg > 0, rcy/tg, 0.0)
        df[f"yards_per_rush_last_{k}"]      = np.where(ca > 0, ry/ca, 0.0)

        touches = rc + ca
        df[f"touches_avg_last_{k}"] = touches
        df[f"yards_per_touch_last_{k}"] = np.where(touches > 0, (rcy+ry)/touches, 0.0)

    # Create additional required features
    df["early_season"] = (df["week"] <= 4).astype(int)
    df["mid_season"]   = ((df["week"] > 4) & (df["week"] <= 12)).astype(int)
    df["late_season"]  = (df["week"] > 12).astype(int)
    df["week_progression"] = df["week"]/18.0
    
    # Create missing features that the model expects
    df["target_share_lag"] = df.get("target_share", 0.0)
    df["air_yards_share_lag"] = df.get("air_yards_share", 0.0)
    df["wopr_lag"] = df.get("wopr", 0.0)
    df["receiving_epa_lag"] = df.get("receiving_epa", 0.0)
    df["receiving_first_downs_lag"] = df.get("receiving_first_downs", 0.0)
    df["catch_rate_lag"] = np.where(df.get("targets_lag", 0) > 0, df.get("receptions_lag", 0) / df.get("targets_lag", 1), 0.0)
    df["yards_per_reception_lag"] = np.where(df.get("receptions_lag", 0) > 0, df.get("receiving_yards_lag", 0) / df.get("receptions_lag", 1), 0.0)
    df["yards_per_target_lag"] = np.where(df.get("targets_lag", 0) > 0, df.get("receiving_yards_lag", 0) / df.get("targets_lag", 1), 0.0)
    df["yards_per_rush_lag"] = np.where(df.get("carries_lag", 0) > 0, df.get("rushing_yards_lag", 0) / df.get("carries_lag", 1), 0.0)
    df["total_yards_lag"] = df.get("receiving_yards_lag", 0) + df.get("rushing_yards_lag", 0)
    df["total_touches_lag"] = df.get("receptions_lag", 0) + df.get("carries_lag", 0)
    df["red_zone_targets_share_lag"] = 0.0  # Default value
    
    return df

class TEInferenceEngine:
    """Advanced TE inference engine with SHAP explanations and dynamic weighting."""

    def __init__(self, model_dir: str):
        """Initialize the inference engine."""
        self.model_dir = Path(model_dir)
        self.models = {}
        self.encoders = {}
        self.scaler = None
        self.feature_schema = {}
        self.feature_names = []

        # Load all artifacts (scaler before feature schema)
        self._load_models()
        self._load_encoders()
        self._load_scaler()  # Load scaler before feature schema
        self._load_feature_schema()  # This can now use scaler if schema is missing

    def _load_models(self):
        """Load all trained models."""
        print(f"Loading models from {self.model_dir}...")
        model_files = {
            'lightgbm': 'lightgbm_model.pkl',
            'catboost': 'catboost_model.pkl',
            'random_forest': 'random_forest_model.pkl',
            'gradient_boosting': 'gradient_boosting_model.pkl'
        }
        for name, filename in model_files.items():
            model_path = self.model_dir / filename
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    self.models[name] = pickle.load(f)
                print(f"✅ Loaded {name} model")
            else:
                print(f"⚠️  {name} model not found at {model_path}")
        print(f"Loaded {len(self.models)} models")

    def _load_encoders(self):
        """Load trained encoders."""
        encoders_path = self.model_dir / "encoders.pkl"
        if encoders_path.exists():
            with open(encoders_path, 'rb') as f:
                self.encoders = pickle.load(f)
            print(f"✅ Loaded {len(self.encoders)} encoders")
        else:
            print(f"❌ Encoders not found at {encoders_path}")

    def _load_scaler(self):
        """Load trained scaler."""
        scaler_path = self.model_dir / "scaler.pkl"
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"✅ Loaded scaler")
        else:
            print(f"⚠️ Scaler not found at {scaler_path}, proceeding without scaling")

    def _load_feature_schema(self):
        """Load feature schema."""
        schema_path = self.model_dir / "feature_schema.json"
        if schema_path.exists():
            try:
                with open(schema_path, 'r') as f:
                    self.feature_schema = json.load(f)
                self.feature_names = self.feature_schema['columns']
                print(f"✅ Loaded feature schema with {len(self.feature_names)} features")
            except Exception as e:
                print(f"❌ Error loading feature schema: {e}")
                self.feature_names = []
        else:
            print(f"⚠️ Feature schema not found at {schema_path}")
            # Try to get features from scaler if available
            if hasattr(self, 'scaler') and self.scaler is not None and hasattr(self.scaler, 'feature_names_in_'):
                self.feature_names = list(self.scaler.feature_names_in_)
                print(f"✅ Using feature names from scaler: {len(self.feature_names)} features")
            else:
                self.feature_names = []

    def load_historical_data(self, seasons=[2024], weeks=None):
        """Load historical TE data from NFL API for past weeks and prior week for lagging."""
        print(f"Loading historical TE data for seasons: {seasons}, weeks: {weeks}")
        
        # CRITICAL: Always load at least one week before the earliest requested week for lagging
        if weeks:
            min_week = min(weeks)
            max_week = max(weeks)
            
            # Load from (min_week - 1) to max_week to ensure we have lag data
            load_weeks = list(range(max(1, min_week - 1), min(18, max_week + 1)))
            print(f"Loading weeks {load_weeks[0]}-{load_weeks[-1]} to ensure lag features are available")
            
            weekly_data = nfl.import_weekly_data(seasons)
            te_data = weekly_data[weekly_data['position'] == 'TE'].copy()
            te_data = te_data[te_data['week'].isin(load_weeks)].copy()
            
            # Mark which weeks we actually want predictions for
            te_data['prediction_week'] = te_data['week'].isin(weeks)
        else:
            weekly_data = nfl.import_weekly_data(seasons)
            te_data = weekly_data[weekly_data['position'] == 'TE'].copy()
            te_data['prediction_week'] = True
        
        # Add red_zone_targets column if it doesn't exist (set to 0)
        if 'red_zone_targets' not in te_data.columns:
            te_data['red_zone_targets'] = 0
            print("Note: red_zone_targets not in data, using 0 as default")
            
        print(f"Loaded {len(te_data)} historical TE records")
        print(f"Weeks loaded: {sorted(te_data['week'].unique())}")
        print(f"Records for prediction: {te_data['prediction_week'].sum()}")
        return te_data

    def load_upcoming_data(self, season=2025, week=1, master_sheet_path=None):
        """Load data for upcoming week with 2024 Week 17 lagged features."""
        print(f"Loading upcoming TE data for season {season}, week {week}")
        
        if not master_sheet_path:
            print("⚠️ No master sheet path provided, using placeholder data")
            # Fallback to placeholder if no master sheet
            lag_data = self.load_historical_data(seasons=[2024], weeks=[17])
            
            # Build aggregation dict dynamically based on available columns
            agg_dict = {}
            for col in ['receptions', 'targets', 'receiving_yards', 'receiving_tds', 
                        'rushing_yards', 'carries', 'target_share', 'air_yards_share', 
                        'wopr', 'receiving_epa', 'receiving_first_downs']:
                if col in lag_data.columns:
                    agg_dict[col] = 'mean'
            
            # Handle red_zone_targets specially
            if 'red_zone_targets' in lag_data.columns:
                agg_dict['red_zone_targets'] = 'mean'
            
            lag_stats = lag_data.groupby('player_id').agg(agg_dict).reset_index()
            
            # Add red_zone_targets column if it wasn't in the data
            if 'red_zone_targets' not in lag_stats.columns:
                lag_stats['red_zone_targets'] = 0
            
            upcoming_data = pd.DataFrame({
                'player_id': ['00-0035662', '00-0036973', '00-0038111', '00-0037744', '00-0039338'],
                'player_name': ['T.Kelce', 'G.Kittle', 'M.Andrews', 'T.McBride', 'B.Bowers'],
                'recent_team': ['KC', 'SF', 'BAL', 'ARI', 'LV'],
                'position': ['TE', 'TE', 'TE', 'TE', 'TE'],
                'season': [season, season, season, season, season],
                'week': [week, week, week, week, week],
                'opponent_team': ['BAL', 'NYJ', 'KC', 'LAR', 'LAC'],
                'season_type': ['REG', 'REG', 'REG', 'REG', 'REG']
            })
            
            upcoming_data = upcoming_data.merge(lag_stats, on='player_id', how='left').fillna(0)
            
            # Create lagged features
            for col in ['receptions', 'targets', 'receiving_yards', 'receiving_tds', 'rushing_yards', 'carries',
                        'target_share', 'air_yards_share', 'wopr', 'receiving_epa', 'receiving_first_downs', 'red_zone_targets']:
                if col in upcoming_data.columns:
                    upcoming_data[f'{col}_lag'] = upcoming_data[col]
                    upcoming_data = upcoming_data.drop(columns=[col])
            
            print(f"Loaded {len(upcoming_data)} upcoming TE records with lagged features (placeholder)")
            return upcoming_data
        
        # --- Build 2025 Week 1 from latest 2024 features ---
        print("Building 2025 Week 1 from latest 2024 features...")
        
        # 1) Get training feature list
        if not hasattr(self, 'feature_names') or not self.feature_names:
            print("⚠️ No feature schema loaded, using placeholder data")
            return self.load_upcoming_data(season, week, None)  # Fallback to placeholder
        
        # 2) Build full 2024 feature table (same function used for backtest) 
        #    and slice the *latest row per player*
        print("Loading 2024 data and building features...")
        wb_2024 = nfl.import_weekly_data([2024])
        te_2024 = wb_2024[wb_2024["position"] == "TE"].copy()
        
        # Build features using the same method as training
        te_2024 = self.create_derived_features(te_2024)
        te_2024 = self.encode_categorical_features(te_2024)
        
        # Get latest 2024 row per player
        latest_2024 = (
            te_2024.sort_values(["player_id","season","week"])
                     .groupby("player_id")
                     .tail(1)
        )
        
        # Keep only the columns the model expects (+ id)
        feature_cols = [c for c in self.feature_names if c in latest_2024.columns]
        latest_slice = latest_2024[["player_id"] + feature_cols].copy()
        
        # 3) Read Master Sheet and prefer its player_id
        print(f"Loading master sheet from {master_sheet_path}")
        dk = pd.read_csv(master_sheet_path)
        dk_te = dk[dk["Position"] == "TE"].copy()
        
        # If Master Sheet already has player_id, great; otherwise try crosswalk (optional)
        if "player_id" not in dk_te.columns or dk_te["player_id"].isna().all():
            try:
                # optional fallback only if you want it
                cw = pd.read_csv("../../data/processed/crosswalk_2025.csv")
                cw["name_norm"] = cw["player_name"].str.lower().str.strip()
                dk_te["name_norm"] = dk_te["Name"].str.lower().str.strip()
                dk_te = dk_te.merge(
                    cw[["player_id","recent_team","name_norm"]],
                    on=["name_norm"], how="left"
                )
            except Exception:
                pass  # keep going with whatever IDs we have
        
        # 4) Merge DK → latest feature slice by player_id
        wk1 = dk_te.merge(latest_slice, on="player_id", how="left")
        
        # 5) Stamp meta fields used downstream
        wk1["season"] = season
        wk1["week"] = week
        wk1["season_type"] = "REG"
        wk1["position"] = "TE"
        wk1["recent_team"] = wk1.get("TeamAbbrev", wk1.get("recent_team", "UNK"))
        wk1["player_name"] = wk1.get("Name", wk1.get("player_name", "Unknown"))
        wk1["opponent_team"] = np.nan  # Will be filled by encoder with 'UNK'
        wk1["actual_points"] = np.nan
        
        # 6) Median-fill any lag features that are still missing (rookies, new signings)
        #    Use medians computed from *backtest* rows so distribution is realistic.
        bt_full = te_2024[feature_cols].copy()
        med = bt_full.median(numeric_only=True)
        for col in feature_cols:
            if col not in wk1.columns:
                wk1[col] = med.get(col, 0.0)
        wk1[feature_cols] = wk1[feature_cols].fillna(med)
        
        print(f"✅ Built Week 1 data: {len(wk1)} TEs with {len(feature_cols)} features")
        return wk1

    def calculate_actual_points(self, te_data):
        """Calculate actual PPR fantasy points for historical data."""
        print("Calculating actual PPR fantasy points...")
        if 'receptions' in te_data.columns:
            te_data['actual_points'] = (
                te_data['receptions'] * 1.0 +
                te_data['receiving_yards'] * 0.1 +
                te_data['rushing_yards'] * 0.1 +
                te_data['receiving_tds'] * 6.0 +
                te_data['rushing_tds'] * 6.0 +
                te_data.get('return_tds', 0) * 6.0 +
                te_data.get('two_point_conversions', 0) * 2.0 -
                te_data.get('fumbles_lost', 0) * 2.0
            ).fillna(0)
            print(f"Calculated actual points for {len(te_data)} records")
        else:
            te_data['actual_points'] = np.nan
        return te_data

    def create_lagged_features(self, te_data):
        """Create lagged features for forecasting (exact features model was trained on)."""
        print("Creating lagged features...")
        te_data = te_data.sort_values(['player_id', 'season', 'week']).reset_index(drop=True)
        
        lag_cols = [
            'receptions', 'targets', 'receiving_yards', 'receiving_tds', 'rushing_yards', 'carries',
            'target_share', 'air_yards_share', 'wopr', 'receiving_epa', 'receiving_first_downs',
            'red_zone_targets'
        ]
        
        for col in lag_cols:
            if col in te_data.columns:
                te_data[f'{col}_lag'] = te_data.groupby('player_id')[col].shift(1).fillna(0)
            else:
                # If column doesn't exist, create lag column with 0s
                te_data[f'{col}_lag'] = 0
                
        print("Lagged features created")
        return te_data

    def create_derived_features(self, te_data):
        """Create derived features using lagged stats (same as training)."""
        print("Creating derived features...")
        
        # Safe division helper
        def safe_divide(numerator, denominator, default=0):
            return np.where(denominator != 0, numerator / denominator, default)
        
        te_data['catch_rate_lag'] = safe_divide(
            te_data.get('receptions_lag', 0), 
            te_data.get('targets_lag', 0), 
            0
        )
        te_data['yards_per_reception_lag'] = safe_divide(
            te_data.get('receiving_yards_lag', 0),
            te_data.get('receptions_lag', 0),
            0
        )
        te_data['yards_per_target_lag'] = safe_divide(
            te_data.get('receiving_yards_lag', 0),
            te_data.get('targets_lag', 0),
            0
        )
        te_data['yards_per_rush_lag'] = safe_divide(
            te_data.get('rushing_yards_lag', 0),
            te_data.get('carries_lag', 0),
            0
        )
        
        te_data['total_yards_lag'] = te_data.get('receiving_yards_lag', 0) + te_data.get('rushing_yards_lag', 0)
        te_data['total_touches_lag'] = te_data.get('receptions_lag', 0) + te_data.get('carries_lag', 0)
        
        te_data['red_zone_targets_share_lag'] = safe_divide(
            te_data.get('red_zone_targets_lag', 0),
            np.maximum(te_data.get('targets_lag', 0), 1),
            0
        )
        
        te_data['early_season'] = (te_data['week'] <= 4).astype(int)
        te_data['mid_season'] = ((te_data['week'] > 4) & (te_data['week'] <= 12)).astype(int)
        te_data['late_season'] = (te_data['week'] > 12).astype(int)
        te_data['week_progression'] = te_data['week'] / 18
        
        # Clip values to reasonable ranges
        te_data['catch_rate_lag'] = te_data['catch_rate_lag'].clip(0, 1)
        te_data['yards_per_reception_lag'] = te_data['yards_per_reception_lag'].clip(0, 50)
        te_data['yards_per_target_lag'] = te_data['yards_per_target_lag'].clip(0, 50)
        te_data['yards_per_rush_lag'] = te_data['yards_per_rush_lag'].clip(0, 20)
        te_data['red_zone_targets_share_lag'] = te_data['red_zone_targets_share_lag'].clip(0, 1)
        
        return te_data

    def encode_categorical_features(self, te_data):
        """Encode categorical features using trained encoders."""
        print("Encoding categorical features...")
        
        if 'team' in self.encoders:
            team_encoder = self.encoders['team']
            te_data['team_encoded'] = te_data['recent_team'].map(
                lambda x: team_encoder.transform([x])[0] if x in team_encoder.classes_ else team_encoder.transform(['UNK'])[0] if 'UNK' in team_encoder.classes_ else 0
            )
        else:
            te_data['team_encoded'] = 0
            
        if 'opponent' in self.encoders:
            opponent_encoder = self.encoders['opponent']
            te_data['opponent_encoded'] = te_data['opponent_team'].map(
                lambda x: opponent_encoder.transform([x])[0] if x in opponent_encoder.classes_ else opponent_encoder.transform(['UNK'])[0] if 'UNK' in opponent_encoder.classes_ else 0
            )
        else:
            te_data['opponent_encoded'] = 0
            
        if 'season_type' in self.encoders:
            season_type_encoder = self.encoders['season_type']
            te_data['season_type_encoded'] = te_data['season_type'].fillna('REG').map(
                lambda x: season_type_encoder.transform([x])[0] if x in season_type_encoder.classes_ else season_type_encoder.transform(['REG'])[0] if 'REG' in season_type_encoder.classes_ else 0
            )
        else:
            te_data['season_type_encoded'] = 0
            
        print("Categorical features encoded")
        return te_data

    def prepare_features(self, te_data):
        """Prepare features for inference (same as training)."""
        print("Preparing features for inference...")
        
        # Ensure all required features exist
        if self.feature_names:
            model_features = self.feature_names
            print(f"Models expect {len(model_features)} features")
        else:
            print("⚠️ No feature schema found, using all numeric columns")
            model_features = [col for col in te_data.columns if te_data[col].dtype in ['float64', 'float32', 'int64', 'int32']]
        
        for col in model_features:
            if col not in te_data.columns:
                print(f"Adding missing feature: {col}")
                te_data[col] = 0
        
        # Apply scaling if available
        if self.scaler is not None:
            try:
                X = te_data[model_features].fillna(0)
                X_scaled = self.scaler.transform(X)
                print(f"✅ Applied scaler to {len(model_features)} features")
                return X_scaled, te_data
            except Exception as e:
                print(f"⚠️ Scaler failed: {e}")
                print("Proceeding without scaling")
                X = te_data[model_features].fillna(0)
                return X.values, te_data
        else:
            # No scaler available
            X = te_data[model_features].fillna(0)
            print(f"No scaler available, using unscaled features")
            return X.values, te_data

    def compute_model_weights(self, te_data, predictions):
        """Compute dynamic weights based on MAE and correlation for historical data."""
        # Default weights if no actual points available
        if 'actual_points' not in te_data.columns or te_data['actual_points'].isna().all():
            return {
                'lightgbm': 0.25,
                'catboost': 0.35,
                'random_forest': 0.35,
                'gradient_boosting': 0.05
            }
        
        weights = {}
        actuals = te_data['actual_points'].dropna()
        valid_idx = te_data['actual_points'].notna()
        
        for name, pred in predictions.items():
            pred_valid = pred[valid_idx]
            if len(pred_valid) > 1:
                mae = np.mean(np.abs(pred_valid - actuals))
                corr = np.corrcoef(pred_valid, actuals)[0, 1] if len(pred_valid) > 1 else 0
                corr = corr if not np.isnan(corr) else 0
                weights[name] = (1 / (mae + 1e-6)) * (corr + 1) / 2
            else:
                weights[name] = 0.25
        
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        else:
            weights = {k: 0.25 for k in predictions.keys()}
        
        print(f"Computed dynamic weights: {weights}")
        return weights

    def shap_for_lightgbm(self, model, X):
        """Compute SHAP values for LightGBM."""
        sv = model.predict(X, pred_contrib=True)
        base = sv[:, -1]
        vals = sv[:, :-1]
        return base, vals

    def shap_for_catboost(self, model, X):
        """Compute SHAP values for CatBoost."""
        pool = cb.Pool(X)
        sv = model.get_feature_importance(type='ShapValues', data=pool)
        base = sv[:, -1]
        vals = sv[:, :-1]
        return base, vals

    def shap_for_sklearn_tree(self, model, X):
        """Compute SHAP values for sklearn tree models (Random Forest, Gradient Boosting)."""
        explainer = shap.TreeExplainer(model)
        vals = explainer.shap_values(X)
        base = np.full(X.shape[0], explainer.expected_value, dtype=float)
        return base, vals

    def ensemble_shap(self, shap_dict):
        """Compute ensemble SHAP values by averaging."""
        bases = []
        vals = []
        for b, v in shap_dict.values():
            if b is not None and v is not None:
                bases.append(b)
                vals.append(v)
        
        if not bases or not vals:
            # Return default values if no valid SHAP data
            return np.array([0.0]), np.array([[0.0]])
            
        base_e = np.mean(np.vstack(bases), axis=0)
        vals_e = np.mean(np.stack(vals, axis=0), axis=0)
        return base_e, vals_e

    def top_rationale_rows(self, X, feature_names, base, shap_vals, k=3, decimals=2):
        """Generate readable SHAP rationales for each prediction."""
        # Handle both DataFrame and numpy array
        if hasattr(X, "values"):
            Xv = X.values
        elif isinstance(X, np.ndarray):
            Xv = X
        else:
            Xv = np.array(X)
            
        n, d = shap_vals.shape
        out = []
        for i in range(n):
            contrib = shap_vals[i]
            order = np.argsort(-np.abs(contrib))
            tops = []
            for j in order[:k]:
                if j < len(feature_names):  # Safety check
                    sign = "↑" if contrib[j] > 0 else "↓"
                    tops.append(f"{feature_names[j]} {sign} {abs(contrib[j]):.{decimals}f}")
            pred = base[i] + contrib.sum()
            out.append(f"~{pred:.1f} pts (base {base[i]:.1f}) • " + ", ".join(tops))
        return out

    def predict(self, X, te_data, use_ensemble=True):
        """Make predictions and compute SHAP values."""
        print("Making predictions and computing SHAP values...")
        predictions = {}
        shap_dict = {}

        for name, model in self.models.items():
            try:
                pred = model.predict(X)
                predictions[name] = pred
                
                if name == 'lightgbm':
                    base, vals = self.shap_for_lightgbm(model, X)
                elif name == 'catboost':
                    base, vals = self.shap_for_catboost(model, X)
                else:
                    base, vals = self.shap_for_sklearn_tree(model, X)
                    
                shap_dict[name] = (base, vals)
                print(f"✅ {name}: Generated {len(pred)} predictions and SHAP values")
            except Exception as e:
                print(f"❌ {name}: Error - {e}")

        if use_ensemble and len(predictions) > 1:
            weights = self.compute_model_weights(te_data, predictions)
            ensemble_pred = np.average(
                [pred for name, pred in predictions.items()], 
                axis=0, 
                weights=[weights[name] for name in predictions]
            )
            predictions['ensemble'] = ensemble_pred
            base_e, vals_e = self.ensemble_shap(shap_dict)
            shap_dict['ensemble'] = (base_e, vals_e)
            print(f"✅ ensemble: Generated {len(ensemble_pred)} predictions and SHAP values with weights {weights}")

        # Generate rationales using the correct feature names
        if predictions:
            rationale_model = 'ensemble' if use_ensemble and 'ensemble' in predictions else list(predictions.keys())[0]
            if rationale_model in shap_dict:
                base, vals = shap_dict[rationale_model]
                # Use the feature names from the schema (what models were trained on)
                feature_names_for_shap = self.feature_names if self.feature_names else []
                rationales = self.top_rationale_rows(X, feature_names_for_shap, base, vals, k=3, decimals=2)
            else:
                rationales = ["No SHAP explanation available"] * X.shape[0]
        else:
            rationales = []

        return predictions, rationales

    def run_inference(self, seasons=[2024], past_weeks=None, upcoming_season=2025, upcoming_week=1, master_sheet_path=None, use_ensemble=True, output_file=None):
        """Run inference for at least 5 past weeks (with actual points) and upcoming week."""
        print("🚀 TE Advanced Model Inference")
        print("=" * 50)

        # Ensure at least 5 valid past weeks
        if past_weeks is None:
            past_weeks = [13, 14, 15, 16, 17]
        else:
            valid_weeks = [w for w in past_weeks if 1 <= w <= 17]
            if len(valid_weeks) < len(past_weeks):
                print(f"⚠️ Invalid weeks filtered out: {set(past_weeks) - set(valid_weeks)}")
            past_weeks = valid_weeks
            if len(past_weeks) < 5:
                print(f"Warning: Fewer than 5 past weeks provided ({past_weeks}). Adding more weeks to reach 5.")
                last_week = max(past_weeks) if past_weeks else 17
                while len(past_weeks) < 5 and last_week > 1:
                    last_week -= 1
                    if last_week not in past_weeks and 1 <= last_week <= 17:
                        past_weeks.append(last_week)
                past_weeks.sort()
                print(f"Updated past weeks: {past_weeks}")

        # --- BACKTEST (2024 weeks 13-17) ---
        print("Loading 2024 TE data for backtest...")
        wk24 = nfl.import_weekly_data([2024])
        te24 = wk24[wk24["position"]=="TE"].copy()
        te24["actual_points"] = te24["fantasy_points_ppr"].astype(float)

        # Build lags/derived/enc before slicing holdout
        te24 = add_te_lag_features(te24)
        te24 = self.encode_categorical_features(te24)

        backtest = te24[te24["week"].isin(past_weeks)].copy()
        X_bt = self.prepare_features(backtest)[0] if hasattr(self, "prepare_features") else backtest[self.feature_names].fillna(0).values

        # Predict + SHAP (limit SHAP to LightGBM to avoid hangs)
        preds_bt = {}
        shap_dict = {}
        for name, model in self.models.items():
            p = model.predict(X_bt)
            preds_bt[name] = p
            if name == "lightgbm":
                b, v = self.shap_for_lightgbm(model, X_bt)
            else:
                b, v = (None, None)
            shap_dict[name] = (b, v)

        ens_bt = np.mean(np.vstack([v for v in preds_bt.values()]), axis=0)
        base_bt, vals_bt = self.ensemble_shap(shap_dict)

        backtest_out = backtest[["player_id","player_name","recent_team","position","season","week","actual_points"]].copy()
        for name, v in preds_bt.items():
            backtest_out[f"predicted_points_{name}"] = v
        backtest_out["predicted_points_ensemble"] = ens_bt
        backtest_out["shap_rationale"] = self.top_rationale_rows(X_bt, self.feature_names, base_bt, vals_bt, k=3, decimals=2)

        # --- WEEK 1 (2025) from latest 2024 features ---
        print("Building 2025 Wk1 TE rows from Master Sheet…")

        # 1) schema
        feature_names = self.feature_names

        # 2) full 2024 feature table then take latest row per player
        wb_2024 = nfl.import_weekly_data([2024])
        raw_2024 = wb_2024[wb_2024["position"]=="TE"].copy()
        feat_2024 = add_te_lag_features(raw_2024)
        feat_2024 = self.encode_categorical_features(feat_2024)
        latest_2024 = (
            feat_2024.sort_values(["player_id","season","week"])
                     .groupby("player_id")
                     .tail(1)
        )
        latest_slice = latest_2024[["player_id"] + [c for c in feature_names if c in latest_2024.columns]].copy()

        # 3) read Master Sheet (preferred id source)
        dk = pd.read_csv(master_sheet_path)
        dk_te = dk[dk["Position"]=="TE"].copy()

        # if ids missing, try crosswalk (optional)
        if "player_id" not in dk_te.columns or dk_te["player_id"].isna().all():
            try:
                cw = pd.read_csv(r"C:\Users\ruley\NFLDFSMasterSheet\data\processed\crosswalk_2025.csv")
                cw["name_norm"] = cw["player_name"].map(norm_name)
                cw["team_norm"] = cw["recent_team"].map(map_team)
                dk_te["name_norm"] = dk_te["Name"].map(norm_name)
                dk_te["team_norm"] = dk_te["TeamAbbrev"].map(map_team)
                dk_te = dk_te.merge(
                    cw[["name_norm","team_norm","player_id"]],
                    on=["name_norm","team_norm"], how="left"
                )
            except Exception:
                pass

        # 4) DK → latest features by player_id
        wk1 = dk_te.merge(latest_slice, on="player_id", how="left")

        # 5) Meta fields
        wk1["season"] = upcoming_season
        wk1["week"]   = upcoming_week
        wk1["season_type"] = "REG"
        wk1["position"] = "TE"
        wk1["recent_team"] = wk1.get("TeamAbbrev", wk1.get("recent_team", "UNK"))
        wk1["player_name"] = wk1.get("Name", wk1.get("player_name", "Unknown"))
        wk1["actual_points"] = np.nan
        wk1["dk_salary"] = wk1["Salary"].astype(float)

        # 6) Median fill for missing features (rookies etc.)
        med = feat_2024[feature_names].median(numeric_only=True)
        for col in feature_names:
            if col not in wk1.columns:
                wk1[col] = med.get(col, 0.0)
        wk1[feature_names] = wk1[feature_names].fillna(med)

        # 7) Predict Week 1 from exact model features
        X_up = wk1[feature_names].fillna(0).values
        preds_up = {}
        shap_up = {}
        for name, model in self.models.items():
            preds_up[name] = model.predict(X_up)
            if name == "lightgbm":
                b, v = self.shap_for_lightgbm(model, X_up)
            else:
                b, v = (None, None)
            shap_up[name] = (b, v)
        ens_up = np.mean(np.vstack([v for v in preds_up.values()]), axis=0)
        b_up, v_up = self.ensemble_shap(shap_up)

        up_out = wk1[["player_id","player_name","recent_team","position","season","week"]].copy()
        for name, v in preds_up.items():
            up_out[f"predicted_points_{name}"] = v
        up_out["predicted_points_ensemble"] = ens_up
        up_out["dk_salary"] = wk1["dk_salary"].fillna(0).astype(float)
        up_out["value"] = np.where(up_out["dk_salary"]>0, ens_up/(up_out["dk_salary"]/1000.0), np.nan)
        up_out["actual_points"] = np.nan
        up_out["shap_rationale"] = self.top_rationale_rows(X_up, feature_names, b_up, v_up, k=3, decimals=2)
        
        # Add historical features for TEs (similar to RB/WR models)
        if 'targets' in raw_2024.columns:
            print("Calculating historical features for TEs...")
            
            # Calculate historical averages for each player
            historical_features = []
            for player_id in up_out['player_id'].unique():
                player_data = raw_2024[raw_2024['player_id'] == player_id].copy()
                if len(player_data) >= 3:  # Need at least 3 weeks of data
                    # Sort by season and week to get most recent data first
                    player_data = player_data.sort_values(['season', 'week'], ascending=[False, False])
                    
                    # Calculate 3-week and 5-week averages for available columns
                    targets_l3 = player_data['targets'].head(3).mean() if 'targets' in player_data.columns else 0.0
                    targets_l5 = player_data['targets'].head(5).mean() if 'targets' in player_data.columns else 0.0
                    target_share_l3 = player_data['target_share'].head(3).mean() if 'target_share' in player_data.columns else 0.0
                    
                    # For TEs, we don't have routes, snaps, or route_share in the current model
                    # So we'll set these to 0.0 for consistency with the expected schema
                    routes_l3 = 0.0
                    routes_l5 = 0.0
                    snaps_l3 = 0.0
                    snaps_l5 = 0.0
                    route_share_l3 = 0.0
                    
                    # Add to historical features list
                    historical_features.append({
                        'player_id': player_id,
                        'targets_l3': targets_l3,
                        'targets_l5': targets_l5,
                        'routes_l3': routes_l3,
                        'routes_l5': routes_l5,
                        'snaps_l3': snaps_l3,
                        'snaps_l5': snaps_l5,
                        'target_share_l3': target_share_l3,
                        'route_share_l3': route_share_l3,
                        'rz_tgts_2024': 0.0,  # Placeholder for red zone targets
                        'rz_rush_2024': 0.0   # Placeholder for red zone rushes (TEs don't rush much)
                    })
            
            # Create historical features DataFrame and merge with output
            if historical_features:
                hist_df = pd.DataFrame(historical_features)
                up_out = up_out.merge(hist_df, on='player_id', how='left')
                
                # Fill NaN values with 0.0
                hist_cols = ['targets_l3', 'targets_l5', 'routes_l3', 'routes_l5', 'snaps_l3', 'snaps_l5', 
                           'target_share_l3', 'route_share_l3', 'rz_tgts_2024', 'rz_rush_2024']
                for col in hist_cols:
                    if col in up_out.columns:
                        up_out[col] = up_out[col].fillna(0.0)
                
                print(f"✅ Added historical features for {len(historical_features)} TE players")
            else:
                print("⚠️ No historical features calculated (insufficient data)")
        else:
            print("⚠️ Missing required columns for historical features calculation")

        # --- COMBINE & SAVE ---
        out = pd.concat([backtest_out, up_out], ignore_index=True)
        out = out.sort_values(["season","week","predicted_points_ensemble"], ascending=[True,True,False])
        out.to_csv(output_file, index=False)
        print(f"✅ TE rows saved to {output_file}")
        print(f"2025 Wk1 TEs: {len(up_out)} (nonzero salaries: {(up_out['dk_salary']>0).sum()})")

        return out

def main():
    """Main inference function."""
    import argparse
    parser = argparse.ArgumentParser(description="Advanced TE Model Inference")
    parser.add_argument("--model-dir", default="TEModel_Advanced", help="Model directory")
    parser.add_argument("--seasons", nargs='+', type=int, default=[2024], help="Seasons for historical data")
    parser.add_argument("--past-weeks", nargs='+', type=int, default=[13, 14, 15, 16, 17], help="Past weeks for actual points (defaults to 13-17)")
    parser.add_argument("--upcoming-season", type=int, default=2025, help="Season for upcoming week")
    parser.add_argument("--upcoming-week", type=int, default=1, help="Upcoming week to predict (defaults to 1)")
    parser.add_argument("--master-sheet", default=r"C:\Users\ruley\NFLDFSMasterSheet\data\processed\master_sheet_2025.csv", help="Path to DK salary master sheet")
    parser.add_argument("--output", default="te_predictions_with_salaries.csv", help="Output file")
    parser.add_argument("--no-ensemble", action="store_true", help="Don't use ensemble predictions")
    args = parser.parse_args()

    try:
        engine = TEInferenceEngine(args.model_dir)
    except Exception as e:
        print(f"❌ Failed to initialize inference engine: {e}")
        return

    result = engine.run_inference(
        seasons=args.seasons,
        past_weeks=args.past_weeks,
        upcoming_season=args.upcoming_season,
        upcoming_week=args.upcoming_week,
        master_sheet_path=args.master_sheet,
        use_ensemble=not args.no_ensemble,
        output_file=args.output
    )

    if result is not None:
        print(f"\n🎯 Inference completed successfully!")
        print(f"Results saved to: {args.output}")
        print(f"Models used: {list(engine.models.keys())}")
    else:
        print(f"\n❌ Inference failed!")

if __name__ == "__main__":
    main()