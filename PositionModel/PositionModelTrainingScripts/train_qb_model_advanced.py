#!/usr/bin/env python3
"""
Advanced QB Position Model Training Script
Includes hyperparameter tuning, feature selection, and multiple model types
"""
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import nfl_data_py as nfl
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import lightgbm as lgb
# import xgboost as xgb  # Removed due to compatibility issues
import catboost as cb
import optuna
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_qb_data(seasons=[2023, 2024]):
    """Load weekly QB data from NFL API."""
    print(f"Loading QB data for seasons: {seasons}")
    
    weekly_data = nfl.import_weekly_data(seasons)
    qb_data = weekly_data[weekly_data['position'] == 'QB'].copy()
    
    print(f"Loaded {len(qb_data)} QB weekly records")
    print(f"Available columns: {list(qb_data.columns)}")
    
    return qb_data

def prepare_features(qb_data):
    """Prepare features for modeling, excluding non-numeric columns."""
    print("Preparing features...")
    
    # Get numeric columns only
    numeric_cols = qb_data.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove columns that are not useful for prediction
    exclude_cols = ['player_id', 'season', 'week', 'fantasy_points', 'fantasy_points_ppr']
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    print(f"Selected {len(feature_cols)} numeric features")
    return feature_cols

def create_derived_features(qb_data):
    """Create derived features for QB modeling."""
    print("Creating derived features...")
    
    # Efficiency metrics
    qb_data['completion_rate'] = (qb_data['completions'] / qb_data['attempts']).fillna(0)
    qb_data['yards_per_attempt'] = (qb_data['passing_yards'] / qb_data['attempts']).fillna(0)
    qb_data['yards_per_rush'] = (qb_data['rushing_yards'] / qb_data['carries']).fillna(0)
    
    # Total metrics
    qb_data['total_yards'] = qb_data['passing_yards'] + qb_data['rushing_yards'] + qb_data['receiving_yards']
    qb_data['total_tds'] = qb_data['passing_tds'] + qb_data['rushing_tds'] + qb_data['receiving_tds']
    qb_data['total_touches'] = qb_data['attempts'] + qb_data['carries'] + qb_data['receptions']
    
    # Season progression
    qb_data['early_season'] = (qb_data['week'] <= 4).astype(int)
    qb_data['mid_season'] = ((qb_data['week'] > 4) & (qb_data['week'] <= 12)).astype(int)
    qb_data['late_season'] = (qb_data['week'] > 12).astype(int)
    qb_data['week_progression'] = qb_data['week'] / 18
    
    # Cap extreme values to prevent model issues
    qb_data['completion_rate'] = qb_data['completion_rate'].clip(0, 1)
    qb_data['yards_per_attempt'] = qb_data['yards_per_attempt'].clip(0, 20)
    qb_data['yards_per_rush'] = qb_data['yards_per_rush'].clip(0, 15)
    
    print("Derived features created")
    return qb_data

def encode_categorical_features(qb_data, feature_cols):
    """Encode categorical features for modeling."""
    print("Encoding categorical features...")
    
    # Create encoders dictionary
    encoders = {}
    
    # Encode team
    team_encoder = LabelEncoder()
    qb_data['team_encoded'] = team_encoder.fit_transform(qb_data['recent_team'].fillna('UNK'))
    encoders['team'] = team_encoder
    
    # Encode opponent team
    opponent_encoder = LabelEncoder()
    qb_data['opponent_encoded'] = opponent_encoder.fit_transform(qb_data['opponent_team'].fillna('UNK'))
    encoders['opponent'] = opponent_encoder
    
    # Encode season type
    season_type_encoder = LabelEncoder()
    qb_data['season_type_encoded'] = season_type_encoder.fit_transform(qb_data['season_type'].fillna('REG'))
    encoders['season_type'] = season_type_encoder
    
    # Add encoded columns to feature list
    feature_cols.extend(['team_encoded', 'opponent_encoded', 'season_type_encoded'])
    
    print(f"Encoded {len(encoders)} categorical features")
    return qb_data, feature_cols, encoders

def prepare_modeling_data(qb_encoded, feature_cols, target_col='fantasy_points'):
    """Prepare final modeling dataset."""
    print("Preparing modeling dataset...")
    
    # Select features and target
    X = qb_encoded[feature_cols].copy()
    y = qb_encoded[target_col].copy()
    
    # Handle missing values
    X = X.fillna(0)
    y = y.fillna(0)
    
    # Remove rows with missing target or extreme outliers
    valid_mask = (y.notna()) & (y >= -10) & (y <= 100)  # Reasonable fantasy point range
    X = X[valid_mask]
    y = y[valid_mask]
    
    print(f"Final dataset: {len(X)} samples, {len(feature_cols)} features")
    print(f"Target range: {y.min():.2f} to {y.max():.2f}")
    
    return X, y

def objective_lightgbm(trial, X_train, y_train, X_val, y_val):
    """Optuna objective function for LightGBM hyperparameter tuning."""
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'n_estimators': 1000,
        'random_state': 42,
        'verbose': -1
    }
    
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
              callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=0)])
    
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    
    return rmse

# XGBoost removed due to compatibility issues

def objective_catboost(trial, X_train, y_train, X_val, y_val):
    """Optuna objective function for CatBoost hyperparameter tuning."""
    params = {
        'objective': 'RMSE',
        'eval_metric': 'RMSE',
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'random_strength': trial.suggest_float('random_strength', 1e-8, 10.0, log=True),
        'iterations': 1000,
        'random_state': 42,
        'verbose': False
    }
    
    model = cb.CatBoostRegressor(**params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), 
              early_stopping_rounds=50, verbose=False)
    
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    
    return rmse

def feature_selection_analysis(X, y, feature_names, k_best=20):
    """Perform feature selection analysis."""
    print(f"\n=== FEATURE SELECTION ANALYSIS ===")
    
    # Method 1: SelectKBest with f_regression
    selector_kbest = SelectKBest(score_func=f_regression, k=k_best)
    X_selected_kbest = selector_kbest.fit_transform(X, y)
    selected_features_kbest = [feature_names[i] for i in selector_kbest.get_support(indices=True)]
    
    # Method 2: RFE with Random Forest
    rfe_selector = RFE(estimator=RandomForestRegressor(n_estimators=100, random_state=42), n_features_to_select=k_best)
    X_selected_rfe = rfe_selector.fit_transform(X, y)
    selected_features_rfe = [feature_names[i] for i in rfe_selector.get_support(indices=True)]
    
    # Method 3: Feature importance from Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"Top {k_best} features by f_regression:")
    print(selected_features_kbest)
    
    print(f"\nTop {k_best} features by RFE:")
    print(selected_features_rfe)
    
    print(f"\nTop 15 features by Random Forest importance:")
    print(feature_importance.head(15))
    
    return selected_features_kbest, selected_features_rfe, feature_importance

def train_ensemble_models(X_train, y_train, X_val, y_val, feature_names):
    """Train multiple models and create ensemble."""
    print("\n=== TRAINING ENSEMBLE MODELS ===")
    
    models = {}
    predictions = {}
    metrics = {}
    
    # 1. LightGBM with Optuna tuning
    print("Tuning LightGBM...")
    study_lgb = optuna.create_study(direction='minimize')
    study_lgb.optimize(lambda trial: objective_lightgbm(trial, X_train, y_train, X_val, y_val), 
                       n_trials=50, show_progress_bar=True)
    
    best_lgb_params = study_lgb.best_params
    best_lgb_params.update({'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt', 
                           'n_estimators': 1000, 'random_state': 42, 'verbose': -1})
    
    lgb_model = lgb.LGBMRegressor(**best_lgb_params)
    lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                  callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=0)])
    
    models['lightgbm'] = lgb_model
    predictions['lightgbm'] = lgb_model.predict(X_val)
    
    # 2. CatBoost with Optuna tuning
    print("Tuning CatBoost...")
    study_cb = optuna.create_study(direction='minimize')
    study_cb.optimize(lambda trial: objective_catboost(trial, X_train, y_train, X_val, y_val), 
                       n_trials=50, show_progress_bar=True)
    
    best_cb_params = study_cb.best_params
    best_cb_params.update({'objective': 'RMSE', 'eval_metric': 'RMSE', 
                          'iterations': 1000, 'random_state': 42, 'verbose': False})
    
    cb_model = cb.CatBoostRegressor(**best_cb_params)
    cb_model.fit(X_train, y_train, eval_set=(X_val, y_val), 
                 early_stopping_rounds=50, verbose=False)
    
    models['catboost'] = cb_model
    predictions['catboost'] = cb_model.predict(X_val)
    
    # 3. Random Forest
    print("Training Random Forest...")
    rf_model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    
    models['random_forest'] = rf_model
    predictions['random_forest'] = rf_model.predict(X_val)
    
    # 4. Gradient Boosting
    print("Training Gradient Boosting...")
    gb_model = GradientBoostingRegressor(n_estimators=200, max_depth=8, random_state=42)
    gb_model.fit(X_train, y_train)
    
    models['gradient_boosting'] = gb_model
    predictions['gradient_boosting'] = gb_model.predict(X_val)
    
    # Calculate metrics for each model
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
    
    # Create ensemble (simple average)
    ensemble_pred = np.mean([pred for pred in predictions.values()], axis=0)
    ensemble_rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred))
    ensemble_mae = mean_absolute_error(y_val, ensemble_pred)
    ensemble_r2 = r2_score(y_val, ensemble_pred)
    
    metrics['ensemble'] = {
        'rmse': ensemble_rmse,
        'mae': ensemble_mae,
        'r2': ensemble_r2
    }
    
    print(f"\nENSEMBLE: RMSE={ensemble_rmse:.3f}, MAE={ensemble_mae:.3f}, R²={ensemble_r2:.4f}")
    
    return models, predictions, metrics

def convert_to_serializable(obj):
    """Convert NumPy types to Python native types for JSON serialization."""
    if isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj

def save_advanced_model_and_artifacts(models, metrics, encoders, feature_names, output_dir, 
                                    feature_importance_df, selected_features):
    """Save all models and artifacts."""
    print(f"\nSaving models and artifacts to {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save individual models
    for name, model in models.items():
        model_path = os.path.join(output_dir, f'{name}_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Saved {name} model: {model_path}")
    
    # Save encoders
    encoders_path = os.path.join(output_dir, 'encoders.pkl')
    with open(encoders_path, 'wb') as f:
        pickle.dump(encoders, f)
    
    # Save feature information
    feature_info = {
        'feature_names': feature_names,
        'selected_features_kbest': selected_features[0],
        'selected_features_rfe': selected_features[1],
        'total_features': len(feature_names)
    }
    
    feature_path = os.path.join(output_dir, 'feature_info.json')
    serializable_feature_info = convert_to_serializable(feature_info)
    with open(feature_path, 'w') as f:
        json.dump(serializable_feature_info, f, indent=2)
    
    # Save metrics
    metrics_path = os.path.join(output_dir, 'model_metrics.json')
    serializable_metrics = convert_to_serializable(metrics)
    with open(metrics_path, 'w') as f:
        json.dump(serializable_metrics, f, indent=2)
    
    # Save feature importance
    feature_importance_path = os.path.join(output_dir, 'feature_importance.csv')
    feature_importance_df.to_csv(feature_importance_path, index=False)
    
    # Save training summary
    summary = {
        'training_date': datetime.now().isoformat(),
        'models_trained': list(models.keys()),
        'best_ensemble_rmse': metrics['ensemble']['rmse'],
        'best_ensemble_r2': metrics['ensemble']['r2'],
        'feature_count': len(feature_names),
        'training_samples': len(feature_names)  # This will be updated with actual count
    }
    
    summary_path = os.path.join(output_dir, 'training_summary.json')
    serializable_summary = convert_to_serializable(summary)
    with open(summary_path, 'w') as f:
        json.dump(serializable_summary, f, indent=2)
    
    print(f"All artifacts saved to {output_dir}")

def main():
    """Main training function."""
    print("=== ADVANCED QB MODEL TRAINING ===")
    print("Includes: Hyperparameter Tuning, Feature Selection, Multiple Models, Ensemble")
    
    # Configuration
    seasons = [2020, 2021, 2022, 2023, 2024]  # Expanded to 5 seasons
    output_dir = "PositionModel/QB_Model_Advanced"
    
    try:
        # Load and prepare data
        qb_data = load_qb_data(seasons)
        feature_cols = prepare_features(qb_data)
        qb_data = create_derived_features(qb_data)
        qb_encoded, feature_cols, encoders = encode_categorical_features(qb_data, feature_cols)
        
        # Prepare modeling dataset
        X, y = prepare_modeling_data(qb_encoded, feature_cols)
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        
        print(f"\nData split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        
        # Feature selection analysis
        selected_features = feature_selection_analysis(X_train, y_train, feature_cols)
        
        # Train ensemble models
        models, predictions, metrics = train_ensemble_models(X_train, y_train, X_val, y_val, feature_cols)
        
        # Save everything
        save_advanced_model_and_artifacts(models, metrics, encoders, feature_cols, output_dir, 
                                        selected_features[2], selected_features)
        
        print(f"\n=== TRAINING COMPLETE ===")
        print(f"Best ensemble RMSE: {metrics['ensemble']['rmse']:.3f}")
        print(f"Best ensemble R²: {metrics['ensemble']['r2']:.4f}")
        print(f"Models saved to: {output_dir}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

