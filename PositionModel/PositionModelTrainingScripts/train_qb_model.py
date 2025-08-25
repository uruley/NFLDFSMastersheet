#!/usr/bin/env python3
"""
QB Position Model Training Script
Trains a comprehensive QB model using all available NFL API features
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import nfl_data_py as nfl
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import pickle
import json
from datetime import datetime

def load_qb_data(seasons=[2023, 2024]):
    """Load QB weekly data from NFL API with all available features."""
    print(f"Loading QB data for seasons: {seasons}")
    
    # Load weekly data
    weekly_data = nfl.import_weekly_data(seasons)
    
    # Filter for QBs only
    qb_data = weekly_data[weekly_data['position'] == 'QB'].copy()
    
    print(f"Loaded {len(qb_data)} QB weekly performances")
    print(f"Available columns: {len(qb_data.columns)}")
    
    return qb_data

def prepare_features(qb_data):
    """Prepare all available features for QB modeling."""
    print("Preparing features...")
    
    # Start with numeric features only (exclude categorical columns that will be encoded)
    feature_cols = ['season', 'week']
    
    # Add all passing features
    passing_features = [col for col in qb_data.columns if 'passing' in col.lower()]
    feature_cols.extend(passing_features)
    
    # Add all rushing features
    rushing_features = [col for col in qb_data.columns if 'rushing' in col.lower()]
    feature_cols.extend(rushing_features)
    
    # Add all receiving features (QBs can catch their own passes)
    receiving_features = [col for col in qb_data.columns if 'receiving' in col.lower()]
    feature_cols.extend(receiving_features)
    
    # Add advanced metrics
    advanced_features = [col for col in qb_data.columns if any(x in col.lower() for x in ['epa', 'dakota', 'pacr', 'racr', 'wopr'])]
    feature_cols.extend(advanced_features)
    
    # Add other relevant numeric features
    other_features = [col for col in qb_data.columns if col not in feature_cols and 
                     col not in ['player_id', 'player_display_name', 'headshot_url', 'position', 'position_group',
                                'player_name', 'recent_team', 'opponent_team', 'season_type'] and
                     qb_data[col].dtype in ['int64', 'float64']]
    feature_cols.extend(other_features)
    
    # Remove duplicates
    feature_cols = list(set(feature_cols))
    
    print(f"Selected {len(feature_cols)} feature columns")
    print("Feature categories:")
    print(f"  - Passing: {len(passing_features)}")
    print(f"  - Rushing: {len(rushing_features)}")
    print(f"  - Receiving: {len(receiving_features)}")
    print(f"  - Advanced: {len(advanced_features)}")
    print(f"  - Other: {len(other_features)}")
    
    return feature_cols

def create_derived_features(qb_data):
    """Create additional derived features for QB modeling."""
    print("Creating derived features...")
    
    # Efficiency metrics
    qb_data['completion_rate'] = np.where(qb_data['attempts'] > 0, 
                                         qb_data['completions'] / qb_data['attempts'], 0)
    
    qb_data['yards_per_attempt'] = np.where(qb_data['attempts'] > 0, 
                                           qb_data['passing_yards'] / qb_data['attempts'], 0)
    
    qb_data['yards_per_rush'] = np.where(qb_data['carries'] > 0, 
                                        qb_data['rushing_yards'] / qb_data['carries'], 0)
    
    qb_data['total_yards'] = qb_data['passing_yards'] + qb_data['rushing_yards'] + qb_data['receiving_yards']
    
    qb_data['total_tds'] = qb_data['passing_tds'] + qb_data['rushing_tds'] + qb_data['receiving_tds']
    
    qb_data['total_touches'] = qb_data['attempts'] + qb_data['carries'] + qb_data['receptions']
    
    # Week of season patterns
    qb_data['early_season'] = (qb_data['week'] <= 4).astype(int)
    qb_data['mid_season'] = ((qb_data['week'] > 4) & (qb_data['week'] <= 12)).astype(int)
    qb_data['late_season'] = (qb_data['week'] > 12).astype(int)
    
    # Season progression
    qb_data['week_progression'] = qb_data['week'] / 18  # Normalize to 0-1
    
    print("Created derived features:")
    print("  - Efficiency: completion_rate, yards_per_attempt, yards_per_rush")
    print("  - Totals: total_yards, total_tds, total_touches")
    print("  - Season: early_season, mid_season, late_season, week_progression")
    
    return qb_data

def encode_categorical_features(qb_data, feature_cols):
    """Encode categorical features for modeling."""
    print("Encoding categorical features...")
    
    # Create a copy for encoding
    qb_encoded = qb_data.copy()
    
    # Encode team names
    team_encoder = LabelEncoder()
    qb_encoded['team_encoded'] = team_encoder.fit_transform(qb_encoded['recent_team'].fillna('UNK'))
    
    # Encode opponent teams
    opponent_encoder = LabelEncoder()
    qb_encoded['opponent_encoded'] = opponent_encoder.fit_transform(qb_encoded['opponent_team'].fillna('UNK'))
    
    # Encode season type
    season_type_encoder = LabelEncoder()
    qb_encoded['season_type_encoded'] = season_type_encoder.fit_transform(qb_encoded['season_type'].fillna('UNK'))
    
    # Add encoded columns to feature list
    feature_cols.extend(['team_encoded', 'opponent_encoded', 'season_type_encoded'])
    
    # Save encoders for later use
    encoders = {
        'team': team_encoder,
        'opponent': opponent_encoder,
        'season_type': season_type_encoder
    }
    
    print("Encoded categorical features:")
    print(f"  - Teams: {len(team_encoder.classes_)} unique teams")
    print(f"  - Opponents: {len(opponent_encoder.classes_)} unique opponents")
    print(f"  - Season types: {len(season_type_encoder.classes_)} types")
    
    return qb_encoded, feature_cols, encoders

def prepare_modeling_data(qb_encoded, feature_cols, target_col='fantasy_points'):
    """Prepare final dataset for modeling."""
    print("Preparing final modeling dataset...")
    
    # Select features and target
    X = qb_encoded[feature_cols].copy()
    y = qb_encoded[target_col].copy()
    
    # Remove rows with missing target
    valid_mask = y.notna()
    X = X[valid_mask]
    y = y[valid_mask]
    
    print(f"Final dataset: {len(X)} samples, {len(X.columns)} features")
    print(f"Target range: {y.min():.2f} to {y.max():.2f}")
    print(f"Target mean: {y.mean():.2f}")
    
    # Handle missing values in features
    missing_counts = X.isnull().sum()
    if missing_counts.sum() > 0:
        print(f"Missing values found in {missing_counts[missing_counts > 0].count()} features")
        print("Filling missing values with 0...")
        X = X.fillna(0)
    
    return X, y

def train_qb_model(X, y, test_size=0.2, random_state=42):
    """Train the QB model with cross-validation."""
    print("Training QB model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=None
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Initialize LightGBM model
    model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        random_state=random_state,
        verbose=-1
    )
    
    # Train model
    print("Training LightGBM model...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=100)]
    )
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    print("\n=== MODEL PERFORMANCE ===")
    print(f"Training RMSE: {train_rmse:.2f}")
    print(f"Test RMSE: {test_rmse:.2f}")
    print(f"Training MAE: {train_mae:.2f}")
    print(f"Test MAE: {test_mae:.2f}")
    print(f"Training R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n=== TOP 10 FEATURES ===")
    print(feature_importance.head(10).to_string(index=False))
    
    return model, {
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'feature_importance': feature_importance
    }

def save_model_and_artifacts(model, metrics, encoders, feature_cols, output_dir):
    """Save the trained model and all artifacts."""
    print(f"Saving model and artifacts to {output_dir}")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, 'qb_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved: {model_path}")
    
    # Save metrics
    metrics_path = os.path.join(output_dir, 'model_metrics.json')
    metrics_to_save = {k: v for k, v in metrics.items() if k != 'feature_importance'}
    with open(metrics_path, 'w') as f:
        json.dump(metrics_to_save, f, indent=2)
    print(f"Metrics saved: {metrics_path}")
    
    # Save feature importance
    importance_path = os.path.join(output_dir, 'feature_importance.csv')
    metrics['feature_importance'].to_csv(importance_path, index=False)
    print(f"Feature importance saved: {importance_path}")
    
    # Save encoders
    encoders_path = os.path.join(output_dir, 'encoders.pkl')
    with open(encoders_path, 'wb') as f:
        pickle.dump(encoders, f)
    print(f"Encoders saved: {encoders_path}")
    
    # Save feature list
    features_path = os.path.join(output_dir, 'feature_columns.json')
    with open(features_path, 'w') as f:
        json.dump(feature_cols, f, indent=2)
    print(f"Feature columns saved: {features_path}")
    
    # Save training summary
    summary_path = os.path.join(output_dir, 'training_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"QB Model Training Summary\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Model Performance:\n")
        f.write(f"  Test RMSE: {metrics['test_rmse']:.2f}\n")
        f.write(f"  Test MAE: {metrics['test_mae']:.2f}\n")
        f.write(f"  Test R²: {metrics['test_r2']:.4f}\n\n")
        f.write(f"Features: {len(feature_cols)}\n")
        f.write(f"Training samples: {len(feature_cols)}\n")
    
    print(f"Training summary saved: {summary_path}")
    print(f"All artifacts saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Train QB Position Model')
    parser.add_argument('--seasons', nargs='+', type=int, default=[2023, 2024],
                       help='Seasons to train on (default: 2023 2024)')
    parser.add_argument('--output-dir', type=str, default='PositionModel/QB_Model',
                       help='Output directory for model artifacts')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set size (default: 0.2)')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random state for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    print("=== QB POSITION MODEL TRAINING ===")
    print(f"Seasons: {args.seasons}")
    print(f"Output directory: {args.output_dir}")
    print(f"Test size: {args.test_size}")
    print(f"Random state: {args.random_state}")
    print()
    
    try:
        # Load data
        qb_data = load_qb_data(args.seasons)
        
        # Prepare features
        feature_cols = prepare_features(qb_data)
        
        # Create derived features
        qb_data = create_derived_features(qb_data)
        
        # Encode categorical features
        qb_encoded, feature_cols, encoders = encode_categorical_features(qb_data, feature_cols)
        
        # Prepare modeling data
        X, y = prepare_modeling_data(qb_encoded, feature_cols)
        
        # Train model
        model, metrics = train_qb_model(X, y, args.test_size, args.random_state)
        
        # Save everything
        save_model_and_artifacts(model, metrics, encoders, feature_cols, args.output_dir)
        
        print("\n=== TRAINING COMPLETE ===")
        print(f"QB model successfully trained and saved to: {args.output_dir}")
        print("You can now use this model for QB fantasy point predictions!")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
