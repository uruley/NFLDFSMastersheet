#!/usr/bin/env python3
"""
Advanced WR Model Inference Script (Updated for DK Salaries)
Works with ensemble models (LightGBM, CatBoost, Random Forest, Gradient Boosting)
Uses current-week features, outputs 5+ past weeks, upcoming week with DK salaries
"""
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
import nfl_data_py as nfl
from sklearn.preprocessing import LabelEncoder, StandardScaler
import catboost as cb
import shap
import warnings
warnings.filterwarnings('ignore')

class WRInferenceEngine:
    """Advanced WR inference engine with SHAP explanations, dynamic weighting, and DK salaries."""

    def __init__(self, model_dir: str):
        """Initialize the inference engine."""
        self.model_dir = Path(model_dir)
        self.models = {}
        self.encoders = {}
        self.scaler = None
        self.feature_schema = {}
        self.feature_names = []

        # Load all artifacts
        self._load_models()
        self._load_encoders()
        self._load_scaler()
        self._load_feature_schema()

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
            print(f"❌ Feature schema not found at {schema_path}")
            self.feature_names = []

    def load_historical_data(self, seasons=[2024], weeks=None):
        """Load historical WR data from NFL API for past weeks."""
        print(f"Loading historical WR data for seasons: {seasons}, weeks: {weeks}")
        weekly_data = nfl.import_weekly_data(seasons)
        wr_data = weekly_data[weekly_data['position'] == 'WR'].copy()
        if weeks:
            valid_weeks = [w for w in weeks if 1 <= w <= 17]
            if len(valid_weeks) < len(weeks):
                print(f"⚠️ Invalid weeks filtered out: {set(weeks) - set(valid_weeks)}")
            wr_data = wr_data[wr_data['week'].isin(valid_weeks)].copy()
        print(f"Loaded {len(wr_data)} historical WR records")
        print(f"Available columns: {list(wr_data.columns)}")
        return wr_data

    def load_upcoming_data(self, season=2025, week=1, master_sheet_path=None):
        """Load data for upcoming week with realistic current-week features."""
        print(f"Loading upcoming WR data for season {season}, week {week}")
        
        if not master_sheet_path:
            print("⚠️ No master sheet path provided, using placeholder data")
            # Fallback to placeholder if no master sheet
            upcoming_data = pd.DataFrame({
                'player_id': ['00-0037238', '00-0031381', '00-0039337', '00-0036972', '00-0036264'],
                'player_name': ['D.London', 'D.Adams', 'M.Nabers', 'J.Jefferson', 'J.Chase'],
                'recent_team': ['ATL', 'NYJ', 'NYG', 'MIN', 'CIN'],
                'position': ['WR', 'WR', 'WR', 'WR', 'WR'],
                'season': [season, season, season, season, season],
                'week': [week, week, week, week, week],
                'opponent_team': ['TB', 'SF', 'MIN', 'NYG', 'NE'],
                'season_type': ['REG', 'REG', 'REG', 'REG', 'REG'],
                # Realistic current-week features based on 2024 averages
                'receptions': [5.5, 6.0, 5.0, 6.5, 6.0],
                'targets': [8.0, 9.0, 7.5, 9.5, 8.5],
                'receiving_yards': [65.0, 80.0, 60.0, 85.0, 75.0],
                'receiving_tds': [0.4, 0.5, 0.3, 0.6, 0.5],
                'rushing_yards': [0.0, 0.0, 5.0, 0.0, 0.0],
                'carries': [0.0, 0.0, 0.5, 0.0, 0.0],
                'target_share': [0.22, 0.25, 0.20, 0.28, 0.24],
                'air_yards_share': [0.25, 0.30, 0.22, 0.35, 0.28],
                'wopr': [0.40, 0.45, 0.38, 0.50, 0.43],
                'receiving_epa': [2.0, 2.5, 1.8, 3.0, 2.3],
                'receiving_first_downs': [3.0, 3.5, 2.8, 4.0, 3.2]
            })
            print(f"Loaded {len(upcoming_data)} upcoming WR records (placeholder)")
            return upcoming_data
        
        # --- Build 2025 Week 1 from latest 2024 features ---
        print("Building 2025 Week 1 from latest 2024 features...")
        
        # 1) Get training feature list
        if not self.feature_names:
            print("⚠️ No feature schema loaded, using placeholder data")
            return self.load_upcoming_data(season, week, None)  # Fallback to placeholder
        
        # 2) Build full 2024 feature table (same function used for backtest) 
        #    and slice the *latest row per player*
        print("Loading 2024 data and building features...")
        wb_2024 = nfl.import_weekly_data([2024])
        wr_2024 = wb_2024[wb_2024["position"] == "WR"].copy()
        
        # Build features using the same method as training
        wr_2024 = self.create_derived_features(wr_2024)
        wr_2024 = self.encode_categorical_features(wr_2024)
        
        # Get latest 2024 row per player
        latest_2024 = (
            wr_2024.sort_values(["player_id","season","week"])
                     .groupby("player_id")
                     .tail(1)
        )
        
        # Keep only the columns the model expects (+ id)
        feature_cols = [c for c in self.feature_names if c in latest_2024.columns]
        latest_slice = latest_2024[["player_id"] + feature_cols].copy()
        
        # 3) Read Master Sheet and prefer its player_id
        print(f"Loading master sheet from {master_sheet_path}")
        dk = pd.read_csv(master_sheet_path)
        dk_wr = dk[dk["Position"] == "WR"].copy()
        
        # If Master Sheet already has player_id, great; otherwise try crosswalk (optional)
        if "player_id" not in dk_wr.columns or dk_wr["player_id"].isna().all():
            try:
                # optional fallback only if you want it
                cw = pd.read_csv("../../data/processed/crosswalk_2025.csv")
                cw["name_norm"] = cw["player_name"].str.lower().str.strip()
                dk_wr["name_norm"] = dk_wr["Name"].str.lower().str.strip()
                dk_wr = dk_wr.merge(
                    cw[["player_id","recent_team","name_norm"]],
                    on=["name_norm"], how="left"
                )
            except Exception:
                pass  # keep going with whatever IDs we have
        
        # 4) Merge DK → latest feature slice by player_id
        wk1 = dk_wr.merge(latest_slice, on="player_id", how="left")
        
        # 5) Stamp meta fields used downstream
        wk1["season"] = season
        wk1["week"] = week
        wk1["season_type"] = "REG"
        wk1["position"] = "WR"
        wk1["recent_team"] = wk1.get("TeamAbbrev", wk1.get("recent_team", "UNK"))
        wk1["player_name"] = wk1.get("Name", wk1.get("player_name", "Unknown"))
        wk1["opponent_team"] = np.nan  # Will be filled by encoder with 'UNK'
        wk1["actual_points"] = np.nan
        
        # 6) Median-fill any lag features that are still missing (rookies, new signings)
        #    Use medians computed from *backtest* rows so distribution is realistic.
        bt_full = wr_2024[feature_cols].copy()
        med = bt_full.median(numeric_only=True)
        for col in feature_cols:
            if col not in wk1.columns:
                wk1[col] = med.get(col, 0.0)
        wk1[feature_cols] = wk1[feature_cols].fillna(med)
        
        print(f"✅ Built Week 1 data: {len(wk1)} WRs with {len(feature_cols)} features")
        return wk1

    def load_dk_salaries(self, master_sheet_path, season, week):
        """Load DraftKings salaries from master sheet for specific season and week."""
        print(f"Loading DK salaries from {master_sheet_path} for season {season}, week {week}...")
        try:
            dk_data = pd.read_csv(master_sheet_path)
            # Filter for WRs only (master sheet is already for 2025 Week 1)
            dk_data = dk_data[dk_data['Position'] == 'WR'].copy()
            dk_data = dk_data.rename(columns={'Name': 'player_name', 'Salary': 'dk_salary', 'TeamAbbrev': 'recent_team'})
            # Add season and week columns since master sheet doesn't have them
            dk_data['season'] = season
            dk_data['week'] = week
            print(f"Loaded {len(dk_data)} DK salary records for WRs")
            return dk_data[['player_id', 'player_name', 'recent_team', 'season', 'week', 'dk_salary', 'join_key']]
        except Exception as e:
            print(f"❌ Error loading DK salaries: {e}")
            return pd.DataFrame(columns=['player_id', 'player_name', 'recent_team', 'season', 'week', 'dk_salary', 'join_key'])

    def calculate_actual_points(self, wr_data):
        """Calculate actual PPR fantasy points for historical data."""
        print("Calculating actual PPR fantasy points...")
        if 'receptions' in wr_data.columns:
            wr_data['actual_points'] = (
                wr_data['receptions'] * 1.0 +
                wr_data['receiving_yards'] * 0.1 +
                wr_data['rushing_yards'] * 0.1 +
                wr_data['receiving_tds'] * 6.0 +
                wr_data['rushing_tds'] * 6.0 +
                wr_data.get('return_tds', 0) * 6.0 +
                wr_data.get('two_point_conversions', 0) * 2.0 -
                wr_data.get('fumbles_lost', 0) * 2.0
            ).fillna(0)
            print(f"Calculated actual points for {len(wr_data)} records")
        else:
            wr_data['actual_points'] = np.nan
        return wr_data

    def create_derived_features(self, wr_data):
        """Create derived features using current-week stats (same as training)."""
        print("Creating derived features...")
        wr_data['catch_rate'] = (wr_data.get('receptions', 0) / wr_data.get('targets', 1)).fillna(0)
        wr_data['yards_per_reception'] = (wr_data.get('receiving_yards', 0) / wr_data.get('receptions', 1)).fillna(0)
        wr_data['yards_per_target'] = (wr_data.get('receiving_yards', 0) / wr_data.get('targets', 1)).fillna(0)
        wr_data['yards_per_rush'] = (wr_data.get('rushing_yards', 0) / wr_data.get('carries', 1)).fillna(0)
        wr_data['total_yards'] = wr_data.get('receiving_yards', 0) + wr_data.get('rushing_yards', 0)
        wr_data['total_touches'] = wr_data.get('receptions', 0) + wr_data.get('carries', 0)
        wr_data['early_season'] = (wr_data['week'] <= 4).astype(int)
        wr_data['mid_season'] = ((wr_data['week'] > 4) & (wr_data['week'] <= 12)).astype(int)
        wr_data['late_season'] = (wr_data['week'] > 12).astype(int)
        wr_data['week_progression'] = wr_data['week'] / 18
        wr_data['catch_rate'] = wr_data['catch_rate'].clip(0, 1)
        wr_data['yards_per_reception'] = wr_data['yards_per_reception'].clip(0, 50)
        wr_data['yards_per_target'] = wr_data['yards_per_target'].clip(0, 50)
        wr_data['yards_per_rush'] = wr_data['yards_per_rush'].clip(0, 20)
        return wr_data

    def encode_categorical_features(self, wr_data):
        """Encode categorical features using trained encoders."""
        print("Encoding categorical features...")
        
        # Handle missing values first
        if "season_type" in wr_data.columns:
            wr_data["season_type"] = wr_data["season_type"].fillna("REG")
        else:
            wr_data["season_type"] = "REG"
        
        # Map each encoder with fallback handling
        for logical, key in [("recent_team","team"),("opponent_team","opponent"),("season_type","season_type")]:
            if key in self.encoders:
                le = self.encoders[key]
                def map_one(x):
                    if logical == "season_type" and (x is None or pd.isna(x)):
                        s = "REG"
                    else:
                        s = str(x) if not pd.isna(x) else "UNK"
                    # Try to transform, fallback to first class if unknown
                    try:
                        if s in le.classes_:
                            return le.transform([s])[0]
                        else:
                            # Use first class as fallback for unknown labels
                            return le.transform([le.classes_[0]])[0]
                    except Exception:
                        return 0
                wr_data[f"{key}_encoded"] = wr_data[logical].map(map_one) if logical in wr_data.columns else 0
            else:
                wr_data[f"{key}_encoded"] = 0
        
        print("Categorical features encoded")
        return wr_data

    def prepare_features(self, wr_data):
        """Prepare features for inference (same as training)."""
        print("Preparing features for inference...")
        wr_data = self.create_derived_features(wr_data)
        wr_data = self.encode_categorical_features(wr_data)
        available_features = [col for col in self.feature_names if col in wr_data.columns]
        missing_features = [col for col in self.feature_names if col not in wr_data.columns]
        print(f"Available features: {len(available_features)}/{len(self.feature_names)}")
        if missing_features:
            print(f"Missing features: {missing_features}")
            for col in missing_features:
                wr_data[col] = 0
        X = wr_data[self.feature_names].fillna(0)
        if self.scaler is not None:
            try:
                X = self.scaler.transform(X)
            except ValueError as e:
                print(f"❌ Scaler error: {e}")
                print(f"Feature names in X: {list(X.columns)}")
                print(f"Expected by scaler: {self.scaler.feature_names_in_}")
                raise
        print(f"Feature matrix shape: {X.shape}")
        return X, wr_data

    def compute_model_weights(self, wr_data, predictions):
        """Compute dynamic weights based on MAE and correlation for historical data."""
        if 'actual_points' not in wr_data.columns or wr_data['actual_points'].isna().all():
            return {
                'lightgbm': 0.25,
                'catboost': 0.35,
                'random_forest': 0.35,
                'gradient_boosting': 0.05
            }
        weights = {}
        actuals = wr_data['actual_points'].dropna()
        valid_idx = wr_data['actual_points'].notna()
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
            bases.append(b)
            vals.append(v)
        base_e = np.mean(np.vstack(bases), axis=0)
        vals_e = np.mean(np.stack(vals, axis=0), axis=0)
        return base_e, vals_e

    def top_rationale_rows(self, X, feature_names, base, shap_vals, k=3, decimals=2):
        """Generate readable SHAP rationales for each prediction."""
        Xv = X.values if hasattr(X, "values") else X
        n, d = shap_vals.shape
        out = []
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

    def predict(self, X, wr_data, use_ensemble=True):
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
            weights = self.compute_model_weights(wr_data, predictions)
            ensemble_pred = np.average([pred for name, pred in predictions.items()], axis=0, weights=[weights[name] for name in predictions])
            predictions['ensemble'] = ensemble_pred
            base_e, vals_e = self.ensemble_shap(shap_dict)
            shap_dict['ensemble'] = (base_e, vals_e)
            print(f"✅ ensemble: Generated {len(ensemble_pred)} predictions and SHAP values with weights {weights}")

        rationale_model = 'ensemble' if use_ensemble and 'ensemble' in predictions else list(predictions.keys())[0]
        base, vals = shap_dict[rationale_model]
        rationales = self.top_rationale_rows(X, self.feature_names, base, vals, k=3, decimals=2)

        return predictions, rationales

    def run_inference(self, seasons=[2024], past_weeks=None, upcoming_season=2025, upcoming_week=1, master_sheet_path=None, use_ensemble=True, output_file=None):
        """Run inference for at least 5 past weeks and upcoming week with DK salaries."""
        print("🚀 WR Advanced Model Inference")
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

        # Load historical data for past weeks
        historical_data = pd.DataFrame()
        if seasons and past_weeks:
            historical_data = self.load_historical_data(seasons, past_weeks)
            historical_data = self.calculate_actual_points(historical_data)

        # Load upcoming week data
        upcoming_data = pd.DataFrame()
        if upcoming_season and upcoming_week:
            upcoming_data = self.load_upcoming_data(upcoming_season, upcoming_week, master_sheet_path)

        # Combine datasets
        wr_data = pd.concat([historical_data, upcoming_data], ignore_index=True)
        if len(wr_data) == 0:
            print("❌ No data found for specified seasons/weeks")
            return None

        # Filter to requested prediction weeks
        prediction_weeks = past_weeks + [upcoming_week] if upcoming_season and upcoming_week else past_weeks
        wr_data = wr_data[wr_data['week'].isin(prediction_weeks)].copy()

        # Ensure at least 5 past weeks are represented
        past_weeks_in_data = wr_data[wr_data['season'] == 2024]['week'].unique()
        if len(past_weeks_in_data) < 5 and past_weeks:
            print(f"Warning: Only {len(past_weeks_in_data)} past weeks found in data: {past_weeks_in_data}. Expected at least 5.")

        # Prepare features
        X, wr_data = self.prepare_features(wr_data)

        # Make predictions and get SHAP rationales
        predictions, rationales = self.predict(X, wr_data, use_ensemble)
        if not predictions:
            print("❌ No predictions generated")
            return None

        # Create output
        output = wr_data[['player_id', 'player_name', 'recent_team', 'position', 'season', 'week', 'actual_points']].copy()
        for name, pred in predictions.items():
            output[f'predicted_points_{name}'] = pred
        output['shap_rationale'] = rationales
        
        # Add historical features for WRs (similar to RB model)
        if 'targets' in wr_data.columns:
            print("Calculating historical features for WRs...")
            
            # Calculate historical averages for each player
            historical_features = []
            for player_id in output['player_id'].unique():
                player_data = wr_data[wr_data['player_id'] == player_id].copy()
                if len(player_data) >= 3:  # Need at least 3 weeks of data
                    # Sort by season and week to get most recent data first
                    player_data = player_data.sort_values(['season', 'week'], ascending=[False, False])
                    
                    # Calculate 3-week and 5-week averages for available columns
                    targets_l3 = player_data['targets'].head(3).mean() if 'targets' in player_data.columns else 0.0
                    targets_l5 = player_data['targets'].head(5).mean() if 'targets' in player_data.columns else 0.0
                    target_share_l3 = player_data['target_share'].head(3).mean() if 'target_share' in player_data.columns else 0.0
                    
                    # For WRs, we don't have routes, snaps, or route_share in the current model
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
                        'rz_rush_2024': 0.0   # Placeholder for red zone rushes (WRs don't rush much)
                    })
            
            # Create historical features DataFrame and merge with output
            if historical_features:
                hist_df = pd.DataFrame(historical_features)
                output = output.merge(hist_df, on='player_id', how='left')
                
                # Fill NaN values with 0.0
                hist_cols = ['targets_l3', 'targets_l5', 'routes_l3', 'routes_l5', 'snaps_l3', 'snaps_l5', 
                           'target_share_l3', 'route_share_l3', 'rz_tgts_2024', 'rz_rush_2024']
                for col in hist_cols:
                    if col in output.columns:
                        output[col] = output[col].fillna(0.0)
                
                print(f"✅ Added historical features for {len(historical_features)} WR players")
            else:
                print("⚠️ No historical features calculated (insufficient data)")
        else:
            print("⚠️ Missing required columns for historical features calculation")

        # Load DK salaries for 2025 Week 1
        if master_sheet_path and upcoming_season and upcoming_week:
            dk_salaries = self.load_dk_salaries(master_sheet_path, upcoming_season, upcoming_week)
            # Match on player_id (primary) or join_key (fallback)
            output = output.merge(
                dk_salaries[['player_id', 'dk_salary', 'join_key']],
                on=['player_id'],
                how='left'
            )
            # For rows without player_id match, try join_key
            unmatched = output[output['dk_salary'].isna() & (output['season'] == upcoming_season) & (output['week'] == upcoming_week)]
            if not unmatched.empty:
                output.loc[unmatched.index, 'join_key'] = output.loc[unmatched.index, 'player_name'].str.lower() + '|' + output.loc[unmatched.index, 'recent_team'] + '|WR'
                output = output.drop(columns=['dk_salary']).merge(
                    dk_salaries[['join_key', 'dk_salary']],
                    on='join_key',
                    how='left'
                )
            output['dk_salary'] = output['dk_salary'].fillna(0).astype(int)  # Zero for historical weeks or unmatched players
            print(f"Matched {len(output[output['dk_salary'] > 0])} players with DK salaries for 2025 Week 1")

        # Sort by season, week, and ensemble predictions
        if 'ensemble' in predictions:
            output = output.sort_values(['season', 'week', 'predicted_points_ensemble'], ascending=[True, True, False])
        else:
            first_model = list(predictions.keys())[0]
            output = output.sort_values(['season', 'week', f'predicted_points_{first_model}'], ascending=[True, True, False])

        # Save results
        if output_file:
            output.to_csv(output_file, index=False)
            print(f"✅ Results saved to {output_file}")

        # Show top predictions
        print(f"\nTop 15 WR predictions with actual points (where available), SHAP rationales, and DK salaries (2025 Week 1):")
        display_cols = ['player_name', 'recent_team', 'season', 'week', 'actual_points', 'shap_rationale', 'dk_salary']
        display_cols.extend([f'predicted_points_{name}' for name in predictions.keys()])
        print(output[display_cols].head(15).to_string(index=False))

        return output

def main():
    """Main inference function."""
    import argparse
    parser = argparse.ArgumentParser(description="Advanced WR Model Inference")
    parser.add_argument("--model-dir", default="WRModel_Advanced", help="Model directory")
    parser.add_argument("--seasons", nargs='+', type=int, default=[2024], help="Seasons for historical data")
    parser.add_argument("--past-weeks", nargs='+', type=int, default=[13, 14, 15, 16, 17], help="Past weeks for actual points (defaults to 13-17)")
    parser.add_argument("--upcoming-season", type=int, default=2025, help="Season for upcoming week")
    parser.add_argument("--upcoming-week", type=int, default=1, help="Upcoming week to predict (defaults to 1)")
    parser.add_argument("--master-sheet", default=r"C:\Users\ruley\NFLDFSMasterSheet\data\processed\master_sheet_2025.csv", help="Path to DK salary master sheet")
    parser.add_argument("--output", default="wr_predictions_with_salaries.csv", help="Output file")
    parser.add_argument("--no-ensemble", action="store_true", help="Don't use ensemble predictions")
    args = parser.parse_args()

    try:
        engine = WRInferenceEngine(args.model_dir)
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