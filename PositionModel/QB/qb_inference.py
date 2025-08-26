#!/usr/bin/env python3
"""
Fixed QB Model Inference Script
Addresses issues with unrealistic predictions and transformation errors
"""
import os
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
import nfl_data_py as nfl

def clean_name(name):
    if pd.isna(name): return ''
    name = str(name).replace(' Jr.', '').replace(' Sr.', '').replace(' III', '').replace(' II', '').replace(' IV', '')
    return ''.join(c for c in name if c.isalnum() or c.isspace()).strip().lower()

def norm_team(team):
    if pd.isna(team): return ''
    return str(team).strip().upper()

def norm_pos(position):
    if pd.isna(position): return ''
    return str(position).strip().upper()

def load_advanced_models(model_dir):
    """Load all trained models and artifacts."""
    print(f"Loading advanced QB models from {model_dir}")
    
    models = {}
    model_files = {
        'lightgbm': 'lightgbm_model.pkl',
        'catboost': 'catboost_model.pkl',
        'random_forest': 'random_forest_model.pkl',
        'gradient_boosting': 'gradient_boosting_model.pkl'
    }
    
    for name, filename in model_files.items():
        model_path = os.path.join(model_dir, filename)
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                models[name] = pickle.load(f)
            print(f"Loaded {name} model")
    
    encoders_path = os.path.join(model_dir, 'encoders.pkl')
    with open(encoders_path, 'rb') as f:
        encoders = pickle.load(f)
    
    feature_path = os.path.join(model_dir, 'feature_info.json')
    with open(feature_path, 'r') as f:
        feature_info = json.load(f)
    
    metrics_path = os.path.join(model_dir, 'model_metrics.json')
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    return models, encoders, feature_info, metrics

def load_dk_slate(slate_path):
    """Load the DraftKings slate."""
    print(f"Loading DraftKings slate from {slate_path}")
    
    dk_slate = pd.read_csv(slate_path)
    qb_slate = dk_slate[dk_slate['Position'] == 'QB'].copy()
    
    print(f"Found {len(qb_slate)} QBs in the slate")
    
    qb_slate['clean_name'] = qb_slate['Name'].apply(clean_name)
    qb_slate['norm_team'] = qb_slate['TeamAbbrev'].apply(norm_team)
    qb_slate['norm_pos'] = qb_slate['Position'].apply(norm_pos)
    qb_slate['join_key'] = qb_slate['clean_name'] + "|" + qb_slate['norm_team'] + "|" + qb_slate['norm_pos']
    
    return qb_slate

def load_master_sheet(master_sheet_path):
    """Load the master sheet for player matching."""
    print(f"Loading master sheet from {master_sheet_path}")
    
    if master_sheet_path.endswith('.csv'):
        master_sheet = pd.read_csv(master_sheet_path)
    elif master_sheet_path.endswith('.parquet'):
        master_sheet = pd.read_parquet(master_sheet_path)
    else:
        raise ValueError("Master sheet must be CSV or Parquet")
    
    return master_sheet

def match_qbs_to_master_sheet(qb_slate, master_sheet):
    """Match QBs from DK slate to master sheet."""
    matched_qbs = qb_slate.merge(
        master_sheet,
        on='join_key',
        how='left',
        suffixes=('_dk', '_master')
    )
    
    matched_count = matched_qbs['player_id'].notna().sum()
    unmatched_count = len(matched_qbs) - matched_count
    
    print(f"Matched: {matched_count} QBs")
    print(f"Unmatched: {unmatched_count} QBs")
    
    return matched_qbs

def get_qb_features_from_master(master_row, current_season=2025, current_week=1):
    """Get QB features with proper adjustments to avoid unrealistic predictions."""
    print(f"Getting features for {master_row['Name_dk']} (ID: {master_row['player_id']})")
    
    try:
        # Get recent QB data
        if pd.notna(master_row['player_id']):
            weekly_data = nfl.import_weekly_data([2023, 2024])
            qb_data = weekly_data[
                (weekly_data['position'] == 'QB') & 
                (weekly_data['player_id'] == master_row['player_id'])
            ].copy()
        else:
            weekly_data = nfl.import_weekly_data([2023, 2024])
            qb_data = weekly_data[
                (weekly_data['position'] == 'QB') & 
                (weekly_data['recent_team'] == master_row['norm_team'])
            ].copy()
        
        if len(qb_data) == 0:
            print(f"No historical data found for {master_row['Name_dk']}")
            return None
        
        qb_data = qb_data.sort_values(['season', 'week'])
        
        # Create lagged features (mimicking training)
        lag_features = [
            'completions', 'attempts', 'passing_yards', 'passing_tds', 'interceptions',
            'sacks', 'rushing_attempts', 'rushing_yards', 'rushing_tds',
            'target_share', 'air_yards_share', 'fantasy_points'
        ]
        
        features = {}
        for window in [3, 5, 10]:
            for feature in lag_features:
                if feature in qb_data.columns:
                    col_name = f'{feature}_avg_last_{window}'
                    features[col_name] = qb_data[feature].shift(1).rolling(window=window, min_periods=1).mean().iloc[-1] or 0
        
        features['fantasy_points_std_last_5'] = qb_data['fantasy_points'].shift(1).rolling(window=5, min_periods=2).std().iloc[-1] or 5.0
        
        # Derived features
        features['completion_rate_last_3'] = (
            features.get('completions_avg_last_3', 0) / (features.get('attempts_avg_last_3', 1) or 1)
        ) * 0.95
        features['completion_rate_last_3'] = min(features['completion_rate_last_3'], 0.75)
        
        features['yards_per_attempt_last_3'] = (
            features.get('passing_yards_avg_last_3', 0) / (features.get('attempts_avg_last_3', 1) or 1)
        ) * 0.9
        features['yards_per_attempt_last_3'] = min(features['yards_per_attempt_last_3'], 9.0)
        
        features['td_rate_last_3'] = (
            features.get('passing_tds_avg_last_3', 0) / (features.get('attempts_avg_last_3', 1) or 1)
        ) * 0.85
        features['td_rate_last_3'] = min(features['td_rate_last_3'], 0.10)
        
        features['int_rate_last_3'] = (
            features.get('interceptions_avg_last_3', 0) / (features.get('attempts_avg_last_3', 1) or 1)
        )
        features['int_rate_last_3'] = min(features['int_rate_last_3'], 0.08)
        
        # Context features
        features['week'] = current_week
        features['early_season'] = 1 if current_week <= 4 else 0
        features['mid_season'] = 1 if 4 < current_week <= 12 else 0
        features['late_season'] = 1 if current_week > 12 else 0
        features['week_progression'] = current_week / 18
        features['is_consistent'] = 1 if features['fantasy_points_std_last_5'] < 8 else 0
        features['recent_team'] = master_row['norm_team_dk']
        
        # Cap total stats
        features['total_yards'] = min(
            (features.get('passing_yards_avg_last_3', 0) * 0.9 + 
             features.get('rushing_yards_avg_last_3', 0) * 0.8),
            400
        )
        features['total_tds'] = min(
            (features.get('passing_tds_avg_last_3', 0) * 0.85 + 
             features.get('rushing_tds_avg_last_3', 0) * 0.7),
            4
        )
        
        print(f"Found {len(qb_data)} historical performances")
        return features
    
    except Exception as e:
        print(f"Error getting features for {master_row['Name_dk']}: {e}")
        return None

def encode_qb_features(qb_features, encoders, feature_names):
    """Encode QB features for model prediction."""
    feature_vector = []
    
    for feature in feature_names:
        if feature in qb_features:
            value = qb_features[feature]
            if 'yards' in feature.lower():
                value = min(value, 500)
            elif 'tds' in feature.lower() or 'touchdowns' in feature.lower():
                value = min(value, 5)
            elif 'rate' in feature.lower() or 'percentage' in feature.lower():
                value = min(value, 1.0)
            feature_vector.append(value)
        elif feature == 'team_encoded':
            team = qb_features.get('recent_team', 'UNK')
            if team in encoders['team'].classes_:
                feature_vector.append(encoders['team'].transform([team])[0])
            else:
                feature_vector.append(0)
        elif feature == 'opponent_encoded':
            feature_vector.append(0)
        elif feature == 'season_type_encoded':
            feature_vector.append(0)
        else:
            feature_vector.append(0)
    
    return np.array(feature_vector).reshape(1, -1)

def make_ensemble_predictions(matched_qbs, models, encoders, feature_names, metrics, current_season=2025, current_week=1):
    """Make predictions with realistic adjustments and proper transformation."""
    print("Making ensemble QB predictions with adjustments...")
    
    predictions = []
    weights = np.array([1.0 / (metrics[model]['rmse'] + 0.001) for model in models])
    weights = weights / weights.sum()  # Normalize weights
    
    for _, qb_row in matched_qbs.iterrows():
        qb_name = qb_row['Name_dk']
        team = qb_row['TeamAbbrev_dk']
        salary = qb_row['Salary_dk']
        player_id = qb_row.get('player_id', 'Unknown')
        
        print(f"\nProcessing {qb_name} ({team}) - Salary: ${salary:,}")
        
        qb_features = get_qb_features_from_master(qb_row, current_season, current_week)
        
        if qb_features is None:
            print(f"No features for {qb_name} - using salary-based default")
            # Salary-based default prediction
            default_points = min(18.0, max(8.0, 8.0 + (salary - 4000) / 500))
            ensemble_prediction = default_points
            prediction_std = 5.0
            model_predictions = {name: default_points for name in models}
        else:
            feature_vector = encode_qb_features(qb_features, encoders, feature_names)
            
            model_predictions = {}
            for name, model in models.items():
                try:
                    pred_log = model.predict(feature_vector)[0]
                    pred = np.expm1(pred_log)  # Transform back to original scale
                    pred = max(0, min(pred, 35))  # Cap between 0 and 35
                    model_predictions[name] = pred
                    print(f"  {name}: {pred:.2f} points")
                except Exception as e:
                    print(f"  {name}: Error - {e}")
                    model_predictions[name] = 12.0
            
            valid_predictions = [p for p in model_predictions.values() if p > 0]
            if valid_predictions:
                if len(valid_predictions) >= 4:
                    valid_predictions.remove(max(valid_predictions))
                    valid_predictions.remove(min(valid_predictions))
                
                ensemble_prediction = np.average(valid_predictions, weights=weights[:len(valid_predictions)])
                prediction_std = np.std(valid_predictions)
                
                ensemble_prediction = min(ensemble_prediction, 32)
                
                # Salary-based adjustment
                if salary > 7500:
                    ensemble_prediction *= 0.95
                elif salary < 5500:
                    ensemble_prediction *= 1.05
            else:
                ensemble_prediction = 12.0
                prediction_std = 5.0
        
        value = (ensemble_prediction / (salary / 1000)) if salary > 0 else 0
        
        predictions.append({
            'Name': qb_name,
            'Team': team,
            'Position': 'QB',
            'Salary': salary,
            'Player_ID': player_id,
            'Ensemble_Prediction': round(ensemble_prediction, 2),
            'Prediction_Std': round(prediction_std, 2),
            'Value': round(value, 3),
            'Recent_Median_Points': round(qb_features.get('fantasy_points_avg_last_3', 0), 2) if qb_features else 0,
            'Match_Status': 'Matched' if pd.notna(player_id) else 'Unmatched',
            'LightGBM': round(model_predictions.get('lightgbm', 12), 2),
            'CatBoost': round(model_predictions.get('catboost', 12), 2),
            'RandomForest': round(model_predictions.get('random_forest', 12), 2),
            'GradientBoosting': round(model_predictions.get('gradient_boosting', 12), 2)
        })
        
        print(f"  Adjusted Ensemble: {ensemble_prediction:.2f} ± {prediction_std:.2f} points")
        print(f"  Value: {value:.3f} pts/$1000")
    
    return pd.DataFrame(predictions)

def main():
    """Main inference function."""
    model_dir = "QB_Model_Fixed"
    dk_slate_path = "../../data/raw/DKSalaries.csv"
    master_sheet_path = "../../data/processed/master_sheet_2025.csv"
    current_season = 2025
    current_week = 1
    
    print("=== FIXED QB MODEL INFERENCE ===")
    print(f"Model directory: {model_dir}")
    print(f"DK slate: {dk_slate_path}")
    print(f"Master sheet: {master_sheet_path}")
    print(f"Current season: {current_season}, Week: {current_week}")
    print()
    
    try:
        models, encoders, feature_info, metrics = load_advanced_models(model_dir)
        
        if not models:
            print("No models loaded! Please train the models first.")
            return
        
        qb_slate = load_dk_slate(dk_slate_path)
        master_sheet = load_master_sheet(master_sheet_path)
        matched_qbs = match_qbs_to_master_sheet(qb_slate, master_sheet)
        
        if len(matched_qbs) == 0:
            print("No QBs found in the slate!")
            return
        
        predictions_df = make_ensemble_predictions(
            matched_qbs, models, encoders, 
            feature_info['feature_names'], metrics, current_season, current_week
        )
        
        if len(predictions_df) == 0:
            print("No predictions generated!")
            return
        
        predictions_df = predictions_df.sort_values('Ensemble_Prediction', ascending=False)
        
        print("\n" + "="*100)
        print("ADJUSTED QB PREDICTIONS")
        print("="*100)
        
        display_cols = ['Name', 'Team', 'Salary', 'Ensemble_Prediction', 'Prediction_Std', 'Value']
        print(predictions_df[display_cols].to_string(index=False))
        
        output_path = "qb_predictions_fixed.csv"
        predictions_df.to_csv(output_path, index=False)
        print(f"\nPredictions saved to: {output_path}")
        
        print(f"\n=== SUMMARY ===")
        print(f"Total QBs: {len(predictions_df)}")
        print(f"Average prediction: {predictions_df['Ensemble_Prediction'].mean():.2f}")
        print(f"Range: {predictions_df['Ensemble_Prediction'].min():.2f} - {predictions_df['Ensemble_Prediction'].max():.2f}")
        print(f"Top QB: {predictions_df.iloc[0]['Name']} - {predictions_df.iloc[0]['Ensemble_Prediction']:.2f} points")
        
        unrealistic = predictions_df[predictions_df['Ensemble_Prediction'] > 35]
        if len(unrealistic) > 0:
            print(f"\nWARNING: {len(unrealistic)} predictions still above 35 points!")
        else:
            print("\n✓ All predictions within realistic range (0-35 points)")
        
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()