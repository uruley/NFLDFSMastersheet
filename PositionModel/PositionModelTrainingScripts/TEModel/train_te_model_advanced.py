#!/usr/bin/env python3
"""
Advanced TE Position Model Training Script
Follows the same pattern as your successful WR model - uses nfl_data_py, multiple models, Optuna tuning
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
import catboost as cb
import optuna
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_te_data(seasons=[2020, 2021, 2022, 2023, 2024]):
    """Load weekly TE data from NFL API."""
    print(f"Loading TE data for seasons: {seasons}")
    
    weekly_data = nfl.import_weekly_data(seasons)
    te_data = weekly_data[weekly_data['position'] == 'TE'].copy()
    
    print(f"Loaded {len(te_data)} TE weekly records")
    print(f"Available columns: {list(te_data.columns)}")
    
    return te_data

def prepare_features(te_data):
    """Prepare features for modeling, excluding non-numeric columns."""
    print("Preparing features...")
    
    # Get numeric columns only
    numeric_cols = te_data.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove columns that are not useful for prediction
    exclude_cols = ['player_id', 'season', 'week', 'fantasy_points', 'fantasy_points_ppr']
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    print(f"Selected {len(feature_cols)} numeric features")
    return feature_cols

def create_derived_features(te_data):
    """Create derived features for TE modeling."""
    print("Creating derived features...")
    
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
    
    # Cap extreme values to prevent model issues
    te_data['catch_rate'] = te_data['catch_rate'].clip(0, 1)
    te_data['yards_per_reception'] = te_data['yards_per_reception'].clip(0, 50)
    te_data['yards_per_target'] = te_data['yards_per_target'].clip(0, 50)
    te_data['yards_per_rush'] = te_data['yards_per_rush'].clip(0, 20)
    
    print("Derived features created")
    return te_data

def encode_categorical_features(te_data, feature_cols):
    """Encode categorical features for modeling."""
    print("Encoding categorical features...")
    
    # Create encoders dictionary
    encoders = {}
    
    # Encode team
    team_encoder = LabelEncoder()
    te_data['team_encoded'] = team_encoder.fit_transform(te_data['recent_team'].fillna('UNK'))
    encoders['team'] = team_encoder
    
    # Encode opponent team
    opponent_encoder = LabelEncoder()
    te_data['opponent_encoded'] = opponent_encoder.fit_transform(te_data['opponent_team'].fillna('UNK'))
    encoders['opponent'] = opponent_encoder
    
    # Encode season type
    season_type_encoder = LabelEncoder()
    te_data['season_type_encoded'] = season_type_encoder.fit_transform(te_data['season_type'].fillna('REG'))
    encoders['season_type'] = season_type_encoder
    
    # Add encoded columns to feature list
    feature_cols.extend(['team_encoded', 'opponent_encoded', 'season_type_encoded'])
    
    print("Categorical features encoded")
    return te_data, feature_cols, encoders

def feature_selection(X, y, feature_names, k_best=20):
    """Perform feature selection using multiple methods."""
    print("\n=== FEATURE SELECTION ===")
    
    # Method 1: SelectKBest with f_regression
    selector_kbest = SelectKBest(score_func=f_regression, k=k_best)
    X_kbest = selector_kbest.fit_transform(X, y)
    selected_features_kbest = [feature_names[i] for i in selector_kbest.get_support(indices=True)]
    
    # Method 2: RFE with Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    selector_rfe = RFE(estimator=rf, n_features_to_select=k_best)
    X_rfe = selector_rfe.fit_transform(X, y)
    selected_features_rfe = [feature_names[i] for i in selector_rfe.get_support(indices=True)]
    
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

def objective_lightgbm(trial, X_train, y_train, X_val, y_val):
    """Optuna objective for LightGBM hyperparameter tuning."""
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'num_leaves': trial.suggest_int('num_leaves', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
    }
    
    model = lgb.LGBMRegressor(**params, random_state=42, verbose=-1)
    model.fit(X_train, y_train)
    pred = model.predict(X_val)
    return np.sqrt(mean_squared_error(y_val, pred))

def objective_catboost(trial, X_train, y_train, X_val, y_val):
    """Optuna objective for CatBoost hyperparameter tuning."""
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
    }
    
    model = cb.CatBoostRegressor(**params, random_state=42, verbose=False)
    model.fit(X_train, y_train)
    pred = model.predict(X_val)
    return np.sqrt(mean_squared_error(y_val, pred))

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

def save_models_and_artifacts(models, encoders, feature_names, metrics, output_dir):
    """Save all models and artifacts."""
    print(f"\nSaving models and artifacts to {output_dir}...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save individual models
    for name, model in models.items():
        model_path = output_path / f"{name}_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Saved {name} model to {model_path}")
    
    # Save encoders
    encoders_path = output_path / "encoders.pkl"
    with open(encoders_path, 'wb') as f:
        pickle.dump(encoders, f)
    print(f"Saved encoders to {encoders_path}")
    
    # Save feature schema
    feature_schema = {
        "columns": feature_names,
        "types": {col: "numeric" for col in feature_names}
    }
    schema_path = output_path / "feature_schema.json"
    with open(schema_path, 'w') as f:
        json.dump(feature_schema, f, indent=2)
    print(f"Saved feature schema to {schema_path}")
    
    # Save metrics
    metrics_path = output_path / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")
    
    # Save feature importance for LightGBM (most important model)
    if 'lightgbm' in models:
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': models['lightgbm'].feature_importances_
        }).sort_values('importance', ascending=False)
        
        importance_path = output_path / "feature_importance.csv"
        importance_df.to_csv(importance_path, index=False)
        print(f"Saved feature importance to {importance_path}")
    
    # Save summary
    summary = {
        "model_type": "Ensemble (LightGBM, CatBoost, Random Forest, Gradient Boosting)",
        "features_used": feature_names,
        "n_features": len(feature_names),
        "metrics": metrics,
        "training_date": datetime.now().isoformat(),
        "nfl_api_seasons": [2020, 2021, 2022, 2023, 2024]
    }
    
    summary_path = output_path / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")

def main():
    """Main training function."""
    print("🚀 Advanced TE Model Training")
    print("=" * 50)
    
    # Load data from NFL API
    te_data = load_te_data()
    
    # Create derived features
    te_data = create_derived_features(te_data)
    
    # Prepare features
    feature_cols = prepare_features(te_data)
    
    # Encode categorical features
    te_data, feature_cols, encoders = encode_categorical_features(te_data, feature_cols)
    
    # Create target variable (next week's fantasy points)
    te_data = te_data.sort_values(["player_id", "season", "week"])
    te_data["fantasy_points_next"] = te_data.groupby("player_id")["fantasy_points"].shift(-1)
    
    # Filter out rows without next week's fantasy points
    te_data = te_data.dropna(subset=["fantasy_points_next"])
    
    # Prepare feature matrix and target AFTER filtering
    X = te_data[feature_cols].fillna(0)
    y = te_data["fantasy_points_next"]
    
    print(f"Records with next week data: {len(y)}")
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target range: {y.min():.1f} to {y.max():.1f}")
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    
    # Feature selection
    selected_features, _, feature_importance = feature_selection(X_train, y_train, feature_cols)
    
    # Use selected features
    X_train_selected = X_train[selected_features]
    X_val_selected = X_val[selected_features]
    X_test_selected = X_test[selected_features]
    
    print(f"\nUsing {len(selected_features)} selected features")
    
    # Train ensemble models
    models, predictions, metrics = train_ensemble_models(
        X_train_selected, y_train, X_val_selected, y_val, selected_features
    )
    
    # Save everything
    output_dir = "TEModel_Advanced"
    save_models_and_artifacts(models, encoders, selected_features, metrics, output_dir)
    
    print(f"\n🎯 TRAINING COMPLETE!")
    print(f"Models saved to: {output_dir}")
    print(f"Best ensemble R²: {metrics['ensemble']['r2']:.4f}")

if __name__ == "__main__":
    main()
