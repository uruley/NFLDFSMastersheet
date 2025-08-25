#!/usr/bin/env python3
"""
Advanced TE Model Inference Script
Works with the newly trained ensemble model (LightGBM, CatBoost, Random Forest, Gradient Boosting)
Follows the exact same pattern as your WR model
"""
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
import nfl_data_py as nfl
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class TEInferenceEngine:
    """Advanced TE inference engine for ensemble models."""
    
    def __init__(self, model_dir: str):
        """Initialize the inference engine."""
        self.model_dir = Path(model_dir)
        self.models = {}
        self.encoders = {}
        self.feature_schema = {}
        self.feature_names = []
        
        # Load all artifacts
        self._load_models()
        self._load_encoders()
        self._load_feature_schema()
        
    def _load_models(self):
        """Load all trained models."""
        print(f"Loading models from {self.model_dir}...")
        
        # Load individual models
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
            
    def _load_feature_schema(self):
        """Load feature schema."""
        schema_path = self.model_dir / "feature_schema.json"
        if schema_path.exists():
            with open(schema_path, 'r') as f:
                self.feature_schema = json.load(f)
            self.feature_names = self.feature_schema['columns']
            print(f"✅ Loaded feature schema with {len(self.feature_names)} features")
        else:
            print(f"❌ Feature schema not found at {schema_path}")
            
    def load_new_data(self, seasons=[2024], weeks=None):
        """Load new TE data from NFL API for inference."""
        print(f"Loading new TE data for seasons: {seasons}")
        
        # Load weekly data
        weekly_data = nfl.import_weekly_data(seasons)
        
        # Filter for TEs only
        te_data = weekly_data[weekly_data['position'] == 'TE'].copy()
        
        # Filter by specific weeks if provided
        if weeks:
            te_data = te_data[te_data['week'].isin(weeks)].copy()
            
        print(f"Loaded {len(te_data)} TE records")
        print(f"Available columns: {list(te_data.columns)}")
        
        return te_data
        
    def prepare_features(self, te_data):
        """Prepare features for inference (same as training)."""
        print("Preparing features for inference...")
        
        # Create derived features (same as training)
        te_data = self._create_derived_features(te_data)
        
        # Encode categorical features
        te_data = self._encode_categorical_features(te_data)
        
        # Select only the features the model expects
        available_features = [col for col in self.feature_names if col in te_data.columns]
        missing_features = [col for col in self.feature_names if col not in te_data.columns]
        
        print(f"Available features: {len(available_features)}/{len(self.feature_names)}")
        if missing_features:
            print(f"Missing features: {missing_features}")
            # Fill missing features with 0
            for col in missing_features:
                te_data[col] = 0
                
        # Create feature matrix
        X = te_data[self.feature_names].fillna(0)
        
        print(f"Feature matrix shape: {X.shape}")
        return X, te_data
        
    def _create_derived_features(self, te_data):
        """Create derived features (same as training)."""
        # Efficiency metrics (TE-specific)
        te_data['catch_rate'] = (te_data['receptions'] / te_data['targets']).fillna(0)
        te_data['yards_per_reception'] = (te_data['receiving_yards'] / te_data['receptions']).fillna(0)
        te_data['yards_per_target'] = (te_data['receiving_yards'] / te_data['targets']).fillna(0)
        te_data['yards_per_rush'] = (te_data['rushing_yards'] / te_data['carries']).fillna(0)
        
        # Total metrics
        te_data['total_yards'] = te_data['receiving_yards'] + te_data['rushing_yards']
        te_data['total_tds'] = te_data['receiving_tds'] + te_data['rushing_tds']
        te_data['total_touches'] = te_data['receptions'] + te_data['carries']
        
        # Season progression
        te_data['early_season'] = (te_data['week'] <= 4).astype(int)
        te_data['mid_season'] = ((te_data['week'] > 4) & (te_data['week'] <= 12)).astype(int)
        te_data['late_season'] = (te_data['week'] > 12).astype(int)
        te_data['week_progression'] = te_data['week'] / 18
        
        # Cap extreme values
        te_data['catch_rate'] = te_data['catch_rate'].clip(0, 1)
        te_data['yards_per_reception'] = te_data['yards_per_reception'].clip(0, 50)
        te_data['yards_per_target'] = te_data['yards_per_target'].clip(0, 50)
        te_data['yards_per_rush'] = te_data['yards_per_rush'].clip(0, 20)
        
        return te_data
        
    def _encode_categorical_features(self, te_data):
        """Encode categorical features using trained encoders."""
        # Encode team
        if 'team' in self.encoders:
            team_encoder = self.encoders['team']
            te_data['team_encoded'] = team_encoder.transform(te_data['recent_team'].fillna('UNK'))
            
        # Encode opponent team
        if 'opponent' in self.encoders:
            opponent_encoder = self.encoders['opponent']
            te_data['opponent_encoded'] = opponent_encoder.transform(te_data['opponent_team'].fillna('UNK'))
            
        # Encode season type
        if 'season_type' in self.encoders:
            season_type_encoder = self.encoders['season_type']
            te_data['season_type_encoded'] = season_type_encoder.transform(te_data['season_type'].fillna('REG'))
            
        return te_data
        
    def predict(self, X, use_ensemble=True):
        """Make predictions using loaded models."""
        print("Making predictions...")
        
        predictions = {}
        
        # Individual model predictions
        for name, model in self.models.items():
            try:
                pred = model.predict(X)
                predictions[name] = pred
                print(f"✅ {name}: Generated {len(pred)} predictions")
            except Exception as e:
                print(f"❌ {name}: Error - {e}")
                
        # Ensemble prediction (average of all models)
        if use_ensemble and len(predictions) > 1:
            ensemble_pred = np.mean([pred for pred in predictions.values()], axis=0)
            predictions['ensemble'] = ensemble_pred
            print(f"✅ ensemble: Generated {len(ensemble_pred)} predictions")
            
        return predictions
        
    def run_inference(self, seasons=[2024], weeks=None, use_ensemble=True, output_file=None):
        """Run complete inference pipeline."""
        print("🚀 TE Advanced Model Inference")
        print("=" * 50)
        
        # Load new data
        te_data = self.load_new_data(seasons, weeks)
        
        if len(te_data) == 0:
            print("❌ No data found for specified seasons/weeks")
            return None
            
        # Prepare features
        X, te_data = self.prepare_features(te_data)
        
        # Make predictions
        predictions = self.predict(X, use_ensemble)
        
        if not predictions:
            print("❌ No predictions generated")
            return None
            
        # Create output
        output = te_data[['player_id', 'player_name', 'recent_team', 'position', 'season', 'week']].copy()
        
        # Add predictions
        for name, pred in predictions.items():
            output[f'predicted_points_{name}'] = pred
            
        # Sort by ensemble predictions if available
        if 'ensemble' in predictions:
            output = output.sort_values('predicted_points_ensemble', ascending=False)
        else:
            # Sort by first available model
            first_model = list(predictions.keys())[0]
            output = output.sort_values(f'predicted_points_{first_model}', ascending=False)
            
        # Save results
        if output_file:
            output.to_csv(output_file, index=False)
            print(f"✅ Results saved to {output_file}")
            
        # Show top predictions
        print(f"\nTop 15 TE predictions:")
        display_cols = ['player_name', 'recent_team', 'season', 'week']
        display_cols.extend([f'predicted_points_{name}' for name in predictions.keys()])
        
        print(output[display_cols].head(15).to_string(index=False))
        
        return output

def main():
    """Main inference function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced TE Model Inference")
    parser.add_argument("--model-dir", default="TEModel_Advanced", help="Model directory")
    parser.add_argument("--seasons", nargs='+', type=int, default=[2024], help="Seasons to predict")
    parser.add_argument("--weeks", nargs='+', type=int, default=None, help="Specific weeks to predict")
    parser.add_argument("--output", default="te_predictions_advanced.csv", help="Output file")
    parser.add_argument("--no-ensemble", action="store_true", help="Don't use ensemble predictions")
    
    args = parser.parse_args()
    
    # Initialize inference engine
    try:
        engine = TEInferenceEngine(args.model_dir)
    except Exception as e:
        print(f"❌ Failed to initialize inference engine: {e}")
        return
        
    # Run inference
    result = engine.run_inference(
        seasons=args.seasons,
        weeks=args.weeks,
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
