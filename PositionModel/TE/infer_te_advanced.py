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
from pathlib import Path
import nfl_data_py as nfl
from sklearn.preprocessing import LabelEncoder, StandardScaler
import catboost as cb
import shap
import warnings
warnings.filterwarnings('ignore')

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

    def load_upcoming_data(self, season=2025, week=1):
        """Load data for upcoming week with 2024 Week 17 lagged features."""
        print(f"Loading upcoming TE data for season {season}, week {week}")
        print("Loading 2024 Week 17 for lagged features...")
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
            'week': [week, week, week, week, week],  # Fixed typo here
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
        
        print(f"Loaded {len(upcoming_data)} upcoming TE records with lagged features")
        return upcoming_data

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
        
        # Create lagged features
        te_data = self.create_lagged_features(te_data)
        
        # Create derived features
        te_data = self.create_derived_features(te_data)
        
        # Encode categorical features
        te_data = self.encode_categorical_features(te_data)
        
        # Use feature schema for the final feature set (what models expect)
        if self.feature_names:
            model_features = self.feature_names
            print(f"Models expect {len(model_features)} features: {model_features}")
        else:
            print("⚠️ No feature schema found, using all numeric columns")
            model_features = [col for col in te_data.columns if te_data[col].dtype in ['float64', 'float32', 'int64', 'int32']]
        
        # Ensure all required features exist
        for col in model_features:
            if col not in te_data.columns:
                print(f"Adding missing feature: {col}")
                te_data[col] = 0
        
        # Apply scaling if available
        if self.scaler is not None:
            # Check what features the scaler expects
            if hasattr(self.scaler, 'feature_names_in_'):
                scaler_features = list(self.scaler.feature_names_in_)
                print(f"Scaler expects {len(scaler_features)} features")
                
                # Ensure we have all features the scaler needs
                for col in scaler_features:
                    if col not in te_data.columns:
                        print(f"Adding missing scaler feature: {col}")
                        te_data[col] = 0
                
                # Create matrix with scaler's expected features
                X_for_scaler = te_data[scaler_features].fillna(0)
                
                # Apply scaling
                X_scaled_full = self.scaler.transform(X_for_scaler)
                print(f"✅ Applied scaler to {X_scaled_full.shape[1]} features")
                
                # Now extract only the features that models expect
                # Map model features to their indices in the scaler output
                model_feature_indices = []
                for feat in model_features:
                    if feat in scaler_features:
                        idx = scaler_features.index(feat)
                        model_feature_indices.append(idx)
                    else:
                        print(f"⚠️ Model feature '{feat}' not found in scaler features")
                        # This shouldn't happen if everything is configured correctly
                        # but we'll handle it by adding a zero column
                        model_feature_indices.append(-1)
                
                # Extract the subset of scaled features that models need
                if -1 in model_feature_indices:
                    # Some features are missing, need to handle specially
                    X_scaled = np.zeros((X_scaled_full.shape[0], len(model_features)))
                    for i, idx in enumerate(model_feature_indices):
                        if idx != -1:
                            X_scaled[:, i] = X_scaled_full[:, idx]
                else:
                    # All features found, simple indexing
                    X_scaled = X_scaled_full[:, model_feature_indices]
                
                print(f"✅ Final feature matrix shape: {X_scaled.shape} (extracted {len(model_features)} features from {len(scaler_features)} scaled features)")
                return X_scaled, te_data
            else:
                # Scaler doesn't have feature names, try direct transformation
                X = te_data[model_features].fillna(0)
                try:
                    X_scaled = self.scaler.transform(X)
                    print(f"✅ Applied scaler directly")
                    return X_scaled, te_data
                except Exception as e:
                    print(f"⚠️ Scaler failed: {e}")
                    print("Proceeding without scaling")
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
            bases.append(b)
            vals.append(v)
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

    def run_inference(self, seasons=[2024], past_weeks=None, upcoming_season=2025, upcoming_week=1, use_ensemble=True, output_file=None):
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

        # Load historical data for past weeks (includes prior weeks for lagging)
        historical_data = pd.DataFrame()
        if seasons and past_weeks:
            historical_data = self.load_historical_data(seasons, past_weeks)
            historical_data = self.calculate_actual_points(historical_data)

        # Load upcoming week data
        upcoming_data = pd.DataFrame()
        if upcoming_season and upcoming_week:
            upcoming_data = self.load_upcoming_data(upcoming_season, upcoming_week)
            upcoming_data['prediction_week'] = True  # Mark for predictions

        # Combine datasets
        te_data = pd.concat([historical_data, upcoming_data], ignore_index=True)
        if len(te_data) == 0:
            print("❌ No data found for specified seasons/weeks")
            return None

        # Show stats about the data
        print(f"\nData summary:")
        print(f"  Total records: {len(te_data)}")
        print(f"  Unique players: {te_data['player_id'].nunique()}")
        print(f"  Weeks in data: {sorted(te_data['week'].unique())}")
        
        # Prepare features (this creates lagged features)
        X, te_data = self.prepare_features(te_data)

        # Make predictions and get SHAP rationales
        predictions, rationales = self.predict(X, te_data, use_ensemble)
        if not predictions:
            print("❌ No predictions generated")
            return None

        # Create output - ONLY for weeks we want predictions for
        prediction_mask = te_data['prediction_week'] == True
        output_data = te_data[prediction_mask].copy()
        
        # Create output DataFrame with predictions
        output = output_data[['player_id', 'player_name', 'recent_team', 'position', 'season', 'week', 'actual_points']].copy()
        
        # Add predictions (subset to prediction weeks only)
        for name, pred in predictions.items():
            output[f'predicted_points_{name}'] = pred[prediction_mask]
        
        # Add rationales (subset to prediction weeks only)
        all_rationales = np.array(rationales)
        output['shap_rationale'] = all_rationales[prediction_mask]

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

        # Show top predictions per week
        print(f"\nTop TE predictions by week:")
        for (season, week), group in output.groupby(['season', 'week']):
            print(f"\n--- Season {season}, Week {week} ---")
            display_cols = ['player_name', 'recent_team', 'actual_points']
            if 'predicted_points_ensemble' in output.columns:
                display_cols.append('predicted_points_ensemble')
            else:
                display_cols.append(f'predicted_points_{list(predictions.keys())[0]}')
            print(group[display_cols].head(10).to_string(index=False))

        return output

def main():
    """Main inference function."""
    import argparse
    parser = argparse.ArgumentParser(description="Advanced TE Model Inference")
    parser.add_argument("--model-dir", default="TEModel_Advanced", help="Model directory")
    parser.add_argument("--seasons", nargs='+', type=int, default=[2024], help="Seasons for historical data")
    parser.add_argument("--past-weeks", nargs='+', type=int, default=[13, 14, 15, 16, 17], help="Past weeks for actual points (defaults to 13-17)")
    parser.add_argument("--upcoming-season", type=int, default=2025, help="Season for upcoming week")
    parser.add_argument("--upcoming-week", type=int, default=1, help="Upcoming week to predict (defaults to 1)")
    parser.add_argument("--output", default="te_predictions_advanced.csv", help="Output file")
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