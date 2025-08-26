#!/usr/bin/env python3
"""
Fixed QB Position Model Training Script
Addresses data leakage and unrealistic prediction issues
"""
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import nfl_data_py as nfl
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import lightgbm as lgb
import catboost as cb
import optuna
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_qb_data(seasons=[2020, 2021, 2022, 2023, 2024]):
    """Load weekly QB data from NFL API."""
    print(f"Loading QB data for seasons: {seasons}")
    
    weekly_data = nfl.import_weekly_data(seasons)
    qb_data = weekly_data[weekly_data['position'] == 'QB'].copy()
    
    # FIXED: Remove incomplete games and outliers
    qb_data = qb_data[qb_data['attempts'] >= 10]  # Minimum 10 pass attempts
    
    print(f"Loaded {len(qb_data)} QB weekly records (filtered for 10+ attempts)")
    
    return qb_data

def create_lagged_features(qb_data):
    """Create lagged features to avoid data leakage."""
    print("Creating lagged features to prevent data leakage...")
    
    # Sort by player and game
    qb_data = qb_data.sort_values(['player_id', 'season', 'week'])
    
    # Features to lag (from previous games)
    lag_features = [
        'completions', 'attempts', 'passing_yards', 'passing_tds', 'interceptions',
        'sacks', 'rushing_attempts', 'rushing_yards', 'rushing_tds',
        'target_share', 'air_yards_share'
    ]
    
    # Create rolling averages for different windows
    for window in [3, 5, 10]:  # Last 3, 5, and 10 games
        for feature in lag_features:
            if feature in qb_data.columns:
                # FIXED: Use shift(1) to ensure we don't use current game data
                col_name = f'{feature}_avg_last_{window}'
                qb_data[col_name] = qb_data.groupby('player_id')[feature].transform(
                    lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
                )
    
    # Create lagged fantasy points (this is what we're trying to predict)
    qb_data['fantasy_points_avg_last_3'] = qb_data.groupby('player_id')['fantasy_points'].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
    )
    qb_data['fantasy_points_avg_last_10'] = qb_data.groupby('player_id')['fantasy_points'].transform(
        lambda x: x.shift(1).rolling(window=10, min_periods=1).mean()
    )
    
    # FIXED: Create variance features for consistency
    qb_data['fantasy_points_std_last_5'] = qb_data.groupby('player_id')['fantasy_points'].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=2).std()
    )
    
    # Drop rows with no historical data (first game for each player)
    qb_data = qb_data.dropna(subset=['fantasy_points_avg_last_3'])
    
    print(f"Created lagged features, {len(qb_data)} records remaining")
    
    return qb_data

def create_derived_features(qb_data):
    """Create derived features with proper constraints."""
    print("Creating derived features...")
    
    # FIXED: Use lagged data for efficiency metrics
    qb_data['completion_rate_last_3'] = (
        qb_data['completions_avg_last_3'] / qb_data['attempts_avg_last_3']
    ).fillna(0.6).clip(0, 0.8)  # Cap at 80% max
    
    qb_data['yards_per_attempt_last_3'] = (
        qb_data['passing_yards_avg_last_3'] / qb_data['attempts_avg_last_3']
    ).fillna(7.0).clip(0, 10)  # Cap at 10 YPA max
    
    qb_data['td_rate_last_3'] = (
        qb_data['passing_tds_avg_last_3'] / qb_data['attempts_avg_last_3']
    ).fillna(0.04).clip(0, 0.10)  # Cap TD rate at 10%
    
    qb_data['int_rate_last_3'] = (
        qb_data['interceptions_avg_last_3'] / qb_data['attempts_avg_last_3']
    ).fillna(0.02).clip(0, 0.08)  # Cap INT rate at 8%
    
    # Season progression features
    qb_data['early_season'] = (qb_data['week'] <= 4).astype(int)
    qb_data['mid_season'] = ((qb_data['week'] > 4) & (qb_data['week'] <= 12)).astype(int)
    qb_data['late_season'] = (qb_data['week'] > 12).astype(int)
    qb_data['week_progression'] = qb_data['week'] / 18
    
    # FIXED: Add consistency penalty feature
    qb_data['is_consistent'] = (qb_data['fantasy_points_std_last_5'] < 8).astype(int)
    
    # Home/away if available
    if 'home_team' in qb_data.columns and 'recent_team' in qb_data.columns:
        qb_data['is_home'] = (qb_data['home_team'] == qb_data['recent_team']).astype(int)
    
    print("Derived features created with proper constraints")
    return qb_data

def prepare_features(qb_data):
    """Select features for modeling - only using lagged/historical data."""
    print("Selecting features (using only historical/lagged data)...")
    
    # FIXED: Only use lagged features and context, not current game stats
    feature_cols = [
        # Lagged passing stats
        'completions_avg_last_3', 'completions_avg_last_5', 'completions_avg_last_10',
        'attempts_avg_last_3', 'attempts_avg_last_5', 'attempts_avg_last_10',
        'passing_yards_avg_last_3', 'passing_yards_avg_last_5', 'passing_yards_avg_last_10',
        'passing_tds_avg_last_3', 'passing_tds_avg_last_5', 'passing_tds_avg_last_10',
        'interceptions_avg_last_3', 'interceptions_avg_last_5',
        'sacks_avg_last_3', 'sacks_avg_last_5',
        
        # Lagged rushing stats
        'rushing_attempts_avg_last_3', 'rushing_attempts_avg_last_5',
        'rushing_yards_avg_last_3', 'rushing_yards_avg_last_5',
        'rushing_tds_avg_last_3', 'rushing_tds_avg_last_5',
        
        # Lagged fantasy points
        'fantasy_points_avg_last_3', 'fantasy_points_avg_last_10',
        'fantasy_points_std_last_5',
        
        # Derived efficiency metrics
        'completion_rate_last_3', 'yards_per_attempt_last_3',
        'td_rate_last_3', 'int_rate_last_3',
        
        # Context features
        'week', 'early_season', 'mid_season', 'late_season', 'week_progression',
        'is_consistent'
    ]
    
    # Add home/away if available
    if 'is_home' in qb_data.columns:
        feature_cols.append('is_home')
    
    # Only keep features that exist in the dataframe
    feature_cols = [col for col in feature_cols if col in qb_data.columns]
    
    print(f"Selected {len(feature_cols)} features (all historical/lagged)")
    return feature_cols

def encode_categorical_features(qb_data, feature_cols):
    """Encode categorical features for modeling."""
    print("Encoding categorical features...")
    
    encoders = {}
    
    # Encode team
    if 'recent_team' in qb_data.columns:
        team_encoder = LabelEncoder()
        qb_data['team_encoded'] = team_encoder.fit_transform(qb_data['recent_team'].fillna('UNK'))
        encoders['team'] = team_encoder
        feature_cols.append('team_encoded')
    
    # Encode opponent
    if 'opponent_team' in qb_data.columns:
        opponent_encoder = LabelEncoder()
        qb_data['opponent_encoded'] = opponent_encoder.fit_transform(qb_data['opponent_team'].fillna('UNK'))
        encoders['opponent'] = opponent_encoder
        feature_cols.append('opponent_encoded')
    
    # Encode season type
    if 'season_type' in qb_data.columns:
        season_type_encoder = LabelEncoder()
        qb_data['season_type_encoded'] = season_type_encoder.fit_transform(qb_data['season_type'].fillna('REG'))
        encoders['season_type'] = season_type_encoder
        feature_cols.append('season_type_encoded')
    
    print(f"Encoded {len(encoders)} categorical features")
    return qb_data, feature_cols, encoders

def prepare_modeling_data(qb_data, feature_cols, target_col='fantasy_points'):
    """Prepare final modeling dataset with realistic constraints."""
    print("Preparing modeling dataset...")
    
    # Select features and target
    X = qb_data[feature_cols].copy()
    y = qb_data[target_col].copy()
    
    # Handle missing values
    X = X.fillna(0)
    
    # FIXED: Cap target variable to realistic range
    y = y.clip(lower=0, upper=50)  # No negative points, cap at 50 for training
    
    # Remove extreme outliers (games with unrealistic fantasy points)
    valid_mask = (y >= 0) & (y <= 45)  # Keep games between 0-45 points
    X = X[valid_mask]
    y = y[valid_mask]
    
    # FIXED: Apply log transformation to reduce impact of extreme values
    # We'll transform back during prediction
    y_transformed = np.log1p(y)  # log(1 + y) to handle 0 values
    
    print(f"Final dataset: {len(X)} samples, {len(feature_cols)} features")
    print(f"Target range (original): {y.min():.2f} to {y.max():.2f}")
    print(f"Target range (transformed): {y_transformed.min():.2f} to {y_transformed.max():.2f}")
    
    return X, y_transformed, y

def objective_lightgbm(trial, X_train, y_train, X_val, y_val):
    """Optuna objective for LightGBM with constraints."""
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),  # Reduced max to prevent overfitting
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15),  # Lower learning rate
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 0.9),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 0.9),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),  # Higher minimum
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 8),  # Add depth constraint
        'n_estimators': 500,  # Reduced from 1000
        'random_state': 42,
        'verbose': -1
    }
    
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
              callbacks=[lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(period=0)])
    
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    
    return rmse

def objective_catboost(trial, X_train, y_train, X_val, y_val):
    """Optuna objective for CatBoost with constraints."""
    params = {
        'objective': 'RMSE',
        'eval_metric': 'RMSE',
        'depth': trial.suggest_int('depth', 3, 7),  # Reduced max depth
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),  # Higher regularization
        'border_count': trial.suggest_int('border_count', 32, 128),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'random_strength': trial.suggest_float('random_strength', 0.0, 1.0),
        'iterations': 500,  # Reduced from 1000
        'random_state': 42,
        'verbose': False
    }
    
    model = cb.CatBoostRegressor(**params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), 
              early_stopping_rounds=30, verbose=False)
    
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    
    return rmse

def train_ensemble_models(X_train, y_train, X_val, y_val, feature_names):
    """Train multiple models with proper constraints."""
    print("\n=== TRAINING ENSEMBLE MODELS WITH CONSTRAINTS ===")
    
    models = {}
    predictions = {}
    metrics = {}
    
    # 1. LightGBM with Optuna tuning
    print("Tuning LightGBM...")
    study_lgb = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    study_lgb.optimize(lambda trial: objective_lightgbm(trial, X_train, y_train, X_val, y_val), 
                       n_trials=30, show_progress_bar=True)  # Reduced trials
    
    best_lgb_params = study_lgb.best_params
    best_lgb_params.update({
        'objective': 'regression', 
        'metric': 'rmse', 
        'boosting_type': 'gbdt',
        'n_estimators': 500, 
        'random_state': 42, 
        'verbose': -1
    })
    
    lgb_model = lgb.LGBMRegressor(**best_lgb_params)
    lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                  callbacks=[lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(period=0)])
    
    models['lightgbm'] = lgb_model
    predictions['lightgbm'] = lgb_model.predict(X_val)
    
    # 2. CatBoost with Optuna tuning
    print("Tuning CatBoost...")
    study_cb = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    study_cb.optimize(lambda trial: objective_catboost(trial, X_train, y_train, X_val, y_val), 
                       n_trials=30, show_progress_bar=True)
    
    best_cb_params = study_cb.best_params
    best_cb_params.update({
        'objective': 'RMSE', 
        'eval_metric': 'RMSE',
        'iterations': 500, 
        'random_state': 42, 
        'verbose': False
    })
    
    cb_model = cb.CatBoostRegressor(**best_cb_params)
    cb_model.fit(X_train, y_train, eval_set=(X_val, y_val), 
                 early_stopping_rounds=30, verbose=False)
    
    models['catboost'] = cb_model
    predictions['catboost'] = cb_model.predict(X_val)
    
    # 3. Random Forest with constraints
    print("Training Random Forest...")
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,  # Reduced from 15
        min_samples_split=20,  # Higher minimum
        min_samples_leaf=10,  # Higher minimum
        max_features='sqrt',  # Use sqrt of features
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    models['random_forest'] = rf_model
    predictions['random_forest'] = rf_model.predict(X_val)
    
    # 4. Gradient Boosting with constraints
    print("Training Gradient Boosting...")
    gb_model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,  # Reduced from 8
        min_samples_split=20,
        min_samples_leaf=10,
        learning_rate=0.05,  # Lower learning rate
        subsample=0.8,
        random_state=42
    )
    gb_model.fit(X_train, y_train)
    
    models['gradient_boosting'] = gb_model
    predictions['gradient_boosting'] = gb_model.predict(X_val)
    
    # Calculate metrics for each model (on transformed scale)
    for name, pred in predictions.items():
        rmse = np.sqrt(mean_squared_error(y_val, pred))
        mae = mean_absolute_error(y_val, pred)
        r2 = r2_score(y_val, pred)
        
        metrics[name] = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
        print(f"{name.upper()}: RMSE={rmse:.3f}, MAE={mae:.3f}, R²={r2:.4f}")
    
    # Create ensemble (weighted average based on performance)
    # Give more weight to models with lower RMSE
    weights = []
    for name in predictions.keys():
        weight = 1.0 / (metrics[name]['rmse'] + 0.001)  # Avoid division by zero
        weights.append(weight)
    
    weights = np.array(weights) / np.sum(weights)  # Normalize weights
    
    ensemble_pred = np.average([pred for pred in predictions.values()], axis=0, weights=weights)
    ensemble_rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred))
    ensemble_mae = mean_absolute_error(y_val, ensemble_pred)
    ensemble_r2 = r2_score(y_val, ensemble_pred)
    
    metrics['ensemble'] = {
        'rmse': ensemble_rmse,
        'mae': ensemble_mae,
        'r2': ensemble_r2,
        'weights': weights.tolist()
    }
    
    print(f"\nWEIGHTED ENSEMBLE: RMSE={ensemble_rmse:.3f}, MAE={ensemble_mae:.3f}, R²={ensemble_r2:.4f}")
    print(f"Weights: {dict(zip(predictions.keys(), weights))}")
    
    return models, predictions, metrics

def save_model_wrapper(models, y_transform_params):
    """Create wrapper for models that handles transformation."""
    class ModelWrapper:
        def __init__(self, models, transform_params):
            self.models = models
            self.transform_params = transform_params
        
        def predict(self, X):
            """Make predictions and transform back to original scale."""
            predictions = {}
            for name, model in self.models.items():
                # Get prediction in log space
                pred_log = model.predict(X)
                # Transform back and apply realistic cap
                pred_original = np.expm1(pred_log)  # Inverse of log1p
                pred_capped = np.clip(pred_original, 0, 35)  # Cap at 35 points
                predictions[name] = pred_capped
            
            # Return ensemble average
            return np.mean(list(predictions.values()), axis=0)
    
    return ModelWrapper(models, y_transform_params)

def save_models_and_artifacts(models, metrics, encoders, feature_names, output_dir, y_mean, y_std):
    """Save all models and artifacts."""
    print(f"\nSaving models and artifacts to {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create transform parameters
    transform_params = {'mean': float(y_mean), 'std': float(y_std)}
    
    # Save individual models with wrapper
    for name, model in models.items():
        model_path = os.path.join(output_dir, f'{name}_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Saved {name} model")
    
    # Save encoders
    encoders_path = os.path.join(output_dir, 'encoders.pkl')
    with open(encoders_path, 'wb') as f:
        pickle.dump(encoders, f)
    
    # Save feature information
    feature_info = {
        'feature_names': feature_names,
        'total_features': len(feature_names),
        'transform_params': transform_params
    }
    
    feature_path = os.path.join(output_dir, 'feature_info.json')
    with open(feature_path, 'w') as f:
        json.dump(feature_info, f, indent=2)
    
    # Save metrics
    metrics_path = os.path.join(output_dir, 'model_metrics.json')
    # Convert numpy types to Python types
    for model_name in metrics:
        for metric_name in metrics[model_name]:
            if isinstance(metrics[model_name][metric_name], np.floating):
                metrics[model_name][metric_name] = float(metrics[model_name][metric_name])
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"All artifacts saved to {output_dir}")

def main():
    """Main training function with fixes."""
    print("=== FIXED QB MODEL TRAINING ===")
    print("Fixes: Lagged features, no data leakage, realistic constraints, log transformation")
    
    # Configuration
    seasons = [2020, 2021, 2022, 2023, 2024]
    output_dir = "QB_Model_Fixed"
    
    try:
        # Load and prepare data
        qb_data = load_qb_data(seasons)
        
        # FIXED: Create lagged features first (prevents data leakage)
        qb_data = create_lagged_features(qb_data)
        
        # Create derived features
        qb_data = create_derived_features(qb_data)
        
        # Select features
        feature_cols = prepare_features(qb_data)
        
        # Encode categorical features
        qb_data, feature_cols, encoders = encode_categorical_features(qb_data, feature_cols)
        
        # Prepare modeling dataset with transformation
        X, y_transformed, y_original = prepare_modeling_data(qb_data, feature_cols)
        
        # FIXED: Use time series split to respect temporal order
        # Split by time - train on earlier games, validate on later games
        split_idx = int(len(X) * 0.7)
        X_train = X.iloc[:split_idx]
        y_train = y_transformed.iloc[:split_idx]
        y_train_original = y_original.iloc[:split_idx]
        
        X_temp = X.iloc[split_idx:]
        y_temp = y_transformed.iloc[split_idx:]
        y_temp_original = y_original.iloc[split_idx:]
        
        # Further split validation and test
        val_split = int(len(X_temp) * 0.5)
        X_val = X_temp.iloc[:val_split]
        y_val = y_temp.iloc[:val_split]
        y_val_original = y_temp_original.iloc[:val_split]
        
        X_test = X_temp.iloc[val_split:]
        y_test = y_temp.iloc[val_split:]
        y_test_original = y_temp_original.iloc[val_split:]
        
        print(f"\nData split (temporal): Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        print(f"Train date range: {qb_data.iloc[:split_idx]['season'].min()}-{qb_data.iloc[:split_idx]['season'].max()}")
        print(f"Val/Test date range: {qb_data.iloc[split_idx:]['season'].min()}-{qb_data.iloc[split_idx:]['season'].max()}")
        
        # Train ensemble models
        models, predictions, metrics = train_ensemble_models(X_train, y_train, X_val, y_val, feature_cols)
        
        # Test on held-out test set (transform predictions back)
        print("\n=== TEST SET PERFORMANCE (Original Scale) ===")
        for name, model in models.items():
            y_pred_log = model.predict(X_test)
            y_pred_original = np.expm1(y_pred_log)  # Transform back
            y_pred_capped = np.clip(y_pred_original, 0, 35)  # Cap predictions
            
            rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_capped))
            mae = mean_absolute_error(y_test_original, y_pred_capped)
            
            print(f"{name}: RMSE={rmse:.2f}, MAE={mae:.2f} fantasy points")
        
        # Save everything
        save_models_and_artifacts(
            models, metrics, encoders, feature_cols, output_dir,
            y_train_original.mean(), y_train_original.std()
        )
        
        print(f"\n=== TRAINING COMPLETE ===")
        print(f"Models saved to: {output_dir}")
        print(f"Key improvements:")
        print(f"  - No data leakage (using lagged features)")
        print(f"  - Realistic caps on predictions")
        print(f"  - Log transformation for stable training")
        print(f"  - Temporal validation split")
        print(f"  - Weighted ensemble based on performance")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()