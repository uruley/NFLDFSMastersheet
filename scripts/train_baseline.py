#!/usr/bin/env python3
"""
Baseline model training script for NFL DFS prediction.
Trains a regression model on historical data with season-based train/valid/test splits.
"""

import argparse
import pandas as pd
import numpy as np
import os
import json
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

# Try to import LightGBM, fallback to XGBoost
try:
    import lightgbm as lgb
    MODEL_TYPE = "lightgbm"
    print("Using LightGBM for training")
except ImportError:
    try:
        import xgboost as xgb
        MODEL_TYPE = "xgboost"
        print("LightGBM not available, using XGBoost for training")
    except ImportError:
        raise ImportError("Neither LightGBM nor XGBoost available. Please install one of them.")

# Import unified feature engineering
from feature_engineering import calculate_derived_features, add_team_features, get_all_teams_from_training_data


def load_and_prepare_data(data_path, valid_season, test_season):
    """Load data and prepare train/valid/test splits by season with NO player overlap"""
    print(f"Loading data from: {data_path}")
    
    # Load data
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} total rows")
    
    # Keep only rows with fantasy_points_dk not null
    df = df[df['fantasy_points_dk'].notna()].copy()
    print(f"After filtering null fantasy_points_dk: {len(df):,} rows")
    
    # Convert season to int for comparison
    df['season'] = df['season'].astype(int)
    
    # STRICT TEMPORAL SPLITTING - No player overlap between sets
    print(f"\nImplementing STRICT temporal splitting (no player overlap)...")
    
    # Get all unique players
    all_players = set(df['player_id'].unique())
    print(f"Total unique players: {len(all_players)}")
    
    # Split players by their FIRST appearance season
    player_first_season = df.groupby('player_id')['season'].min()
    
    # Train: Players who first appeared before valid_season
    train_players = set(player_first_season[player_first_season < valid_season].index)
    
    # Valid: Players who first appeared in valid_season
    valid_players = set(player_first_season[player_first_season == valid_season].index)
    
    # Test: Players who first appeared in test_season
    test_players = set(player_first_season[player_first_season == test_season].index)
    
    # Create splits using player_id filtering
    train_df = df[df['player_id'].isin(train_players)].copy()
    valid_df = df[df['player_id'].isin(valid_players)].copy()
    test_df = df[df['player_id'].isin(test_players)].copy()
    
    print(f"\nStrict temporal splits (by player debut season):")
    print(f"  Train (debut < {valid_season}): {len(train_df):,} rows, {len(train_players)} players")
    print(f"  Valid (debut = {valid_season}): {len(valid_df):,} rows, {len(valid_players)} players")
    print(f"  Test (debut = {test_season}): {len(test_df):,} rows, {len(test_players)} players")
    
    # Verify no overlap
    train_player_set = set(train_df['player_id'].unique())
    valid_player_set = set(valid_df['player_id'].unique())
    test_player_set = set(test_df['player_id'].unique())
    
    overlap_train_valid = train_player_set.intersection(valid_player_set)
    overlap_train_test = train_player_set.intersection(test_player_set)
    overlap_valid_test = valid_player_set.intersection(test_player_set)
    
    print(f"\nOverlap verification:")
    print(f"  Train-Valid overlap: {len(overlap_train_valid)} players")
    print(f"  Train-Test overlap: {len(overlap_train_test)} players")
    print(f"  Valid-Test overlap: {len(overlap_valid_test)} players")
    
    if len(overlap_train_valid) > 0 or len(overlap_train_test) > 0 or len(overlap_valid_test) > 0:
        print(f"  ⚠️  WARNING: Player overlap detected! This will cause overfitting!")
        return None, None, None
    
    print(f"  ✅ No player overlap - clean temporal split!")
    
    return train_df, valid_df, test_df


def prepare_features(train_df, valid_df, test_df):
    """Prepare features for all datasets"""
    print("\nPreparing features...")
    
    # Define feature columns - REMOVE ALL BROKEN EFFICIENCY FEATURES
    # These efficiency features are causing the model to predict unrealistic values
    # because they can be infinite or extremely high for players with few touches
    
    # REMOVED: touchdown_rate, yards_per_touch, target_efficiency, carry_efficiency
    # These features are fundamentally flawed for fantasy point prediction
    
    base_features = ['total_touches']  # Only keep total touches - this is safe
    
    categorical_features = ['position_QB', 'position_RB', 'position_WR', 'position_TE',
                           'is_qb', 'is_skill', 'conference_afc']
    
    optional_features = ['completion_rate', 'air_yards_per_attempt', 'breakaway_rate', 
                        'catch_rate', 'week_early', 'week_mid', 'week_late', 
                        'week_progression', 'weeks_since_bye']
    
    # Check which features are available
    available_base = [f for f in base_features if f in train_df.columns]
    available_categorical = [f for f in categorical_features if f in train_df.columns]
    available_optional = [f for f in optional_features if f in train_df.columns]
    
    print(f"Base features: {len(available_base)}")
    print(f"Categorical features: {len(available_categorical)}")
    print(f"Optional features: {len(available_optional)}")
    
    # Get team dummy columns
    team_columns = [col for col in train_df.columns if col.startswith('team_')]
    print(f"Team features: {len(team_columns)}")
    
    # Combine all feature columns
    all_features = available_base + available_categorical + available_optional + team_columns
    
    # Transform all datasets
    def transform_features(df):
        # Select available features
        X = df[all_features].copy()
        
        # Fill any missing values with 0
        X = X.fillna(0)
        
        return X
    
    X_train = transform_features(train_df)
    X_valid = transform_features(valid_df)
    X_test = transform_features(test_df)
    
    # Get actual feature names
    actual_feature_names = list(X_train.columns)
    print(f"Actual feature matrix columns: {len(actual_feature_names)}")
    
    # Calculate feature scaling statistics for inference
    feature_stats = {
        'means': X_train.mean().to_dict(),
        'stds': X_train.std().to_dict(),
        'mins': X_train.min().to_dict(),
        'maxs': X_train.max().to_dict()
    }
    print(f"Feature scaling stats calculated for {len(feature_stats['means'])} features")
    
    # Prepare targets
    y_train = train_df['fantasy_points_dk'].values
    y_valid = valid_df['fantasy_points_dk'].values
    y_test = test_df['fantasy_points_dk'].values
    
    print(f"Feature matrix shapes:")
    print(f"  Train: {X_train.shape}")
    print(f"  Valid: {X_valid.shape}")
    print(f"  Test: {X_test.shape}")
    
    return X_train, X_valid, X_test, y_train, y_valid, y_test, actual_feature_names, feature_stats


def train_model(X_train, y_train, X_valid, y_valid):
    """Train the model"""
    print(f"\nTraining {MODEL_TYPE.upper()} model...")
    
    if MODEL_TYPE == "lightgbm":
        # LightGBM parameters
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }
        
        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)
        
        # Train model
        model = lgb.train(
            params,
            train_data,
            valid_sets=[valid_data],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
        )
        
    else:  # XGBoost
        # XGBoost parameters
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.9,
            'verbosity': 0
        }
        
        # Train model
        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            early_stopping_rounds=50,
            verbose=False
        )
    
    print("Training completed!")
    return model


def run_cross_validation(X_train_full, y_train_full, train_df, cv_folds):
    """Run cross-validation with GroupKFold grouped by season and week"""
    print(f"\nRunning {cv_folds}-fold cross-validation grouped by season and week...")
    
    # Create groups based on season and week for GroupKFold
    groups = train_df['season'].astype(str) + '_' + train_df['week'].astype(str)
    unique_groups = groups.unique()
    print(f"Total unique season-week groups in training data: {len(unique_groups)}")
    
    # Adjust cv_folds if we don't have enough groups
    if len(unique_groups) < cv_folds:
        print(f"Warning: Reducing CV folds from {cv_folds} to {len(unique_groups)} due to insufficient unique groups")
        cv_folds = len(unique_groups)
    
    # Initialize GroupKFold
    gkf = GroupKFold(n_splits=cv_folds)
    
    # Track CV results
    cv_scores = []
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_train_full, y_train_full, groups)):
        print(f"\n--- Fold {fold + 1}/{cv_folds} ---")
        
        # Split data for this fold
        X_fold_train = X_train_full.iloc[train_idx]
        X_fold_val = X_train_full.iloc[val_idx]
        y_fold_train = y_train_full[train_idx]
        y_fold_val = y_train_full[val_idx]
        
        # Show fold info
        train_groups = groups.iloc[train_idx].unique()
        val_groups = groups.iloc[val_idx].unique()
        print(f"  Train groups: {sorted(train_groups)[:5]}...")
        print(f"  Val groups: {sorted(val_groups)[:5]}...")
        print(f"  Train samples: {len(X_fold_train)}")
        print(f"  Val samples: {len(X_fold_val)}")
        
        # Train model on fold
        if MODEL_TYPE == "lightgbm":
            # LightGBM parameters
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1
            }
            
            # Create datasets
            fold_train_data = lgb.Dataset(X_fold_train, label=y_fold_train)
            fold_val_data = lgb.Dataset(X_fold_val, label=y_fold_val, reference=fold_train_data)
            
            # Train model
            fold_model = lgb.train(
                params,
                fold_train_data,
                valid_sets=[fold_val_data],
                num_boost_round=1000,
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]  # Silent training
            )
            
        else:  # XGBoost
            # XGBoost parameters
            params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.9,
                'verbosity': 0
            }
            
            # Train model
            fold_model = xgb.XGBRegressor(**params)
            fold_model.fit(
                X_fold_train, y_fold_train,
                eval_set=[(X_fold_val, y_fold_val)],
                early_stopping_rounds=50,
                verbose=False
            )
        
        # Evaluate fold
        y_fold_pred = fold_model.predict(X_fold_val)
        fold_rmse = np.sqrt(mean_squared_error(y_fold_val, y_fold_pred))
        
        cv_scores.append(fold_rmse)
        fold_results.append({
            'fold': fold + 1,
            'train_samples': len(X_fold_train),
            'val_samples': len(X_fold_val),
            'rmse': fold_rmse
        })
        
        print(f"  Fold RMSE: {fold_rmse:.3f}")
    
    # Calculate CV statistics
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    
    print(f"\n=== CROSS-VALIDATION RESULTS ===")
    print(f"CV RMSE: {cv_mean:.3f} ± {cv_std:.3f}")
    print(f"Individual fold scores: {[f'{score:.3f}' for score in cv_scores]}")
    
    return cv_mean, cv_std, fold_results


def evaluate_model(model, X_train, X_valid, X_test, y_train, y_valid, y_test):
    """Evaluate model performance"""
    print("\nEvaluating model performance...")
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_valid_pred = model.predict(X_valid)
    y_test_pred = model.predict(X_test)
    
    # Calculate RMSE
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    valid_rmse = np.sqrt(mean_squared_error(y_valid, y_valid_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    # Calculate Spearman correlation
    valid_spearman, _ = spearmanr(y_valid, y_valid_pred)
    test_spearman, _ = spearmanr(y_test, y_test_pred)
    
    metrics = {
        'train_rmse': float(train_rmse),
        'valid_rmse': float(valid_rmse),
        'test_rmse': float(test_rmse),
        'valid_spearman': float(valid_spearman),
        'test_spearman': float(test_spearman)
    }
    
    print(f"Metrics:")
    print(f"  Train RMSE: {train_rmse:.3f}")
    print(f"  Valid RMSE: {valid_rmse:.3f}")
    print(f"  Test RMSE: {test_rmse:.3f}")
    print(f"  Valid Spearman: {valid_spearman:.3f}")
    print(f"  Test Spearman: {test_spearman:.3f}")
    
    return metrics, y_train_pred, y_valid_pred, y_test_pred


def save_results(model, metrics, feature_names, feature_stats, train_df, valid_df, test_df,
                y_train_pred, y_valid_pred, y_test_pred, out_dir):
    """Save all results"""
    print(f"\nSaving results to: {out_dir}")
    
    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(out_dir, f"baseline_model_{MODEL_TYPE}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"  Model saved: {model_path}")
    
    # Save feature importance
    if MODEL_TYPE == "lightgbm":
        # Get feature importance and ensure it matches feature names
        feature_importance = model.feature_importance('gain')
        print(f"Feature importance length: {len(feature_importance)}")
        print(f"Feature names length: {len(feature_names)}")
        
        # Ensure lengths match
        if len(feature_importance) != len(feature_names):
            print(f"Warning: Feature importance length ({len(feature_importance)}) doesn't match feature names length ({len(feature_names)})")
            # Truncate or extend as needed
            if len(feature_importance) > len(feature_names):
                feature_importance = feature_importance[:len(feature_names)]
            else:
                feature_names = feature_names[:len(feature_importance)]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        })
    else:  # XGBoost
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        })
    
    importance_df = importance_df.sort_values('importance', ascending=False)
    importance_path = os.path.join(out_dir, "feature_importance.csv")
    importance_df.to_csv(importance_path, index=False)
    print(f"  Feature importance saved: {importance_path}")
    
    # Save metrics
    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved: {metrics_path}")
    
    # Save feature scaling statistics for inference
    feature_stats_path = os.path.join(out_dir, "feature_stats.json")
    with open(feature_stats_path, 'w') as f:
        json.dump(feature_stats, f, indent=2)
    print(f"  Feature scaling stats saved: {feature_stats_path}")
    
    # Create predictions dataframe
    def create_predictions_df(df, y_true, y_pred, season_name):
        # Check which columns are available
        available_cols = ['season', 'week', 'player_id', 'player_name', 'fantasy_points_dk', 'position']
        existing_cols = [col for col in available_cols if col in df.columns]
        
        pred_df = df[existing_cols].copy()
        pred_df['pred'] = y_pred
        pred_df['split'] = season_name
        return pred_df
    
    # Combine all predictions
    all_predictions = []
    
    if len(train_df) > 0:
        all_predictions.append(create_predictions_df(train_df, None, y_train_pred, 'train'))
    if len(valid_df) > 0:
        all_predictions.append(create_predictions_df(valid_df, None, y_valid_pred, 'valid'))
    if len(test_df) > 0:
        all_predictions.append(create_predictions_df(test_df, None, y_test_pred, 'test'))
    
    if all_predictions:
        all_predictions = pd.concat(all_predictions, ignore_index=True)
    else:
        all_predictions = pd.DataFrame()
    
    # Save predictions
    predictions_path = os.path.join(out_dir, "predictions.csv")
    all_predictions.to_csv(predictions_path, index=False)
    print(f"  Predictions saved: {predictions_path}")
    
    return all_predictions


def main():
    parser = argparse.ArgumentParser(description='Train baseline model for NFL DFS prediction')
    parser.add_argument('--data', required=True, help='Path to training data CSV')
    parser.add_argument('--out-dir', required=True, help='Output directory for model and results')
    parser.add_argument('--valid-season', type=int, default=2024, help='Validation season (default: 2024)')
    parser.add_argument('--test-season', type=int, default=2025, help='Test season (default: 2025)')
    parser.add_argument('--cv-folds', type=int, help='Number of cross-validation folds (optional)')
    
    args = parser.parse_args()
    
    try:
        # Load and prepare data
        train_df, valid_df, test_df = load_and_prepare_data(
            args.data, args.valid_season, args.test_season
        )
        
        if train_df is None:
            print("Data preparation failed due to player overlap. Exiting.")
            return 1

        # Prepare features
        X_train, X_valid, X_test, y_train, y_valid, y_test, feature_names, feature_stats = prepare_features(
            train_df, valid_df, test_df
        )
        
        # Run cross-validation if requested
        cv_mean = cv_std = None
        if args.cv_folds:
            cv_mean, cv_std, fold_results = run_cross_validation(
                X_train, y_train, train_df, args.cv_folds
            )
        
        # Train model on full training data
        model = train_model(X_train, y_train, X_valid, y_valid)
        
        # Evaluate model
        metrics, y_train_pred, y_valid_pred, y_test_pred = evaluate_model(
            model, X_train, X_valid, X_test, y_train, y_valid, y_test
        )
        
        # Add CV results to metrics if available
        if cv_mean is not None:
            metrics['cv_rmse_mean'] = float(cv_mean)
            metrics['cv_rmse_std'] = float(cv_std)
        
        # Save results
        predictions = save_results(
            model, metrics, feature_names, feature_stats, train_df, valid_df, test_df,
            y_train_pred, y_valid_pred, y_test_pred, args.out_dir
        )
        
        # Print final report
        print(f"\n{'='*50}")
        print(f"BASELINE MODEL TRAINING COMPLETED")
        print(f"{'='*50}")
        print(f"Model type: {MODEL_TYPE.upper()}")
        print(f"Total training samples: {len(train_df):,}")
        if cv_mean is not None:
            print(f"CV RMSE: {cv_mean:.3f} ± {cv_std:.3f}")
        print(f"Validation RMSE: {metrics['valid_rmse']:.3f}")
        print(f"Test RMSE: {metrics['test_rmse']:.3f}")
        print(f"Validation Spearman: {metrics['valid_spearman']:.3f}")
        print(f"Test Spearman: {metrics['test_spearman']:.3f}")
        print(f"Results saved to: {args.out_dir}")
        print(f"{'='*50}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
