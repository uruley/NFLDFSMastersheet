#!/usr/bin/env python3
"""
RB Model Inference Script
Uses the advanced RB models to make predictions on DraftKings slate
"""

import pandas as pd
import numpy as np
import nfl_data_py as nfl
import pickle
import os
import json

def clean_name(name):
    """Clean player name for matching."""
    if pd.isna(name):
        return ''
    # Remove Jr., Sr., III, etc.
    name = str(name).replace(' Jr.', '').replace(' Sr.', '').replace(' III', '').replace(' II', '').replace(' IV', '')
    # Remove special characters and convert to lowercase
    name = ''.join(c for c in name if c.isalnum() or c.isspace()).strip().lower()
    return name

def norm_team(team):
    """Normalize team abbreviation."""
    if pd.isna(team):
        return ''
    return str(team).strip().upper()

def norm_pos(position):
    """Normalize position."""
    if pd.isna(position):
        return ''
    return str(position).strip().upper()

def load_dk_slate(slate_path):
    """Load the DraftKings slate and prepare for master sheet matching."""
    print(f"Loading DraftKings slate from {slate_path}")
    
    dk_slate = pd.read_csv(slate_path)
    
    # Filter for RBs only
    rb_slate = dk_slate[dk_slate['Position'] == 'RB'].copy()
    
    # Define valid NFL team abbreviations (based on nfl_data_py and DK slate)
    valid_teams = {
        'ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE', 'DAL', 'DEN',
        'DET', 'GB', 'HOU', 'IND', 'JAX', 'KC', 'LA', 'LAR', 'LAC', 'LV', 'MIA',
        'MIN', 'NE', 'NO', 'NYG', 'NYJ', 'PHI', 'PIT', 'SEA', 'SF', 'TB',
        'TEN', 'WAS'
    }
    
    def extract_opponent(game_info, team_abbrev):
        """Extract opponent from Game Info, avoiding self-matches."""
        try:
            # Split Game Info (e.g., CIN@CLE 09/07/2025 01:00PM ET)
            teams = game_info.split(' ')[0].split('@')
            if len(teams) != 2:
                print(f"Warning: Malformed Game Info '{game_info}', defaulting to 'UNK'")
                return 'UNK'
            
            team1, team2 = teams
            
            # Return the team that isn't the player's team
            if team1 == team_abbrev:
                opponent = team2
            elif team2 == team_abbrev:
                opponent = team1
            else:
                print(f"Warning: Player team '{team_abbrev}' not found in Game Info '{game_info}', defaulting to 'UNK'")
                return 'UNK'
            
            # Validate opponent
            if opponent not in valid_teams:
                print(f"Warning: Opponent '{opponent}' not in valid teams, defaulting to 'UNK'")
                return 'UNK'
            
            return opponent
            
        except Exception as e:
            print(f"Error parsing Game Info '{game_info}': {e}, defaulting to 'UNK'")
            return 'UNK'
    
    # Extract opponent using the improved function
    if 'Game Info' in rb_slate.columns:
        rb_slate['Opponent'] = rb_slate.apply(
            lambda row: extract_opponent(row['Game Info'], row['TeamAbbrev']), axis=1
        )
    elif 'Opponent' in rb_slate.columns:
        # If Opponent column already exists, validate it
        rb_slate['Opponent'] = rb_slate['Opponent'].apply(
            lambda x: x if x in valid_teams else 'UNK'
        )
    else:
        print("Warning: No Game Info or Opponent column found, setting default opponent")
        rb_slate['Opponent'] = 'UNK'
    
    print(f"Found {len(rb_slate)} RBs in the slate")
    
    # Prepare for master sheet matching - handle different name column formats
    if 'Name' in rb_slate.columns:
        name_col = 'Name'
    elif 'Name + ID' in rb_slate.columns:
        # Extract just the name part from "Name + ID" column
        rb_slate['Name'] = rb_slate['Name + ID'].str.split(' (')[0]
        name_col = 'Name'
    else:
        print("Warning: No Name column found")
        name_col = 'Name'
    
    rb_slate['clean_name'] = rb_slate[name_col].apply(clean_name)
    rb_slate['norm_team'] = rb_slate['TeamAbbrev'].apply(norm_team)
    rb_slate['norm_pos'] = rb_slate['Position'].apply(norm_pos)
    rb_slate['join_key'] = rb_slate['clean_name'] + "|" + rb_slate['norm_team'] + "|" + rb_slate['norm_pos']
    
    print("Prepared join keys for master sheet matching")
    print("Sample join keys:", rb_slate['join_key'].head().tolist())
    print("Sample opponents:", rb_slate['Opponent'].head().tolist())
    
    return rb_slate

def get_rb_features_from_master(master_row, current_season=2025, current_week=1, weekly_data=None):
    """Get RB features from master sheet data for prediction."""
    # Handle different column name formats from merged data
    rb_name = master_row.get('Name_dk', master_row.get('Name', 'Unknown'))
    player_id = master_row.get('player_id', None)  # This is the NFL API player_id from master sheet
    
    print(f"Getting features for {rb_name} (NFL API ID: {player_id})")
    
    try:
        # Get recent RB data from NFL API using player_id if available
        if pd.notna(player_id):
            print(f"  Using player_id: {player_id}")
            
            # Use passed weekly_data if available, otherwise load it
            if weekly_data is None:
                weekly_data = nfl.import_weekly_data([2023, 2024])
                print(f"  Loaded {len(weekly_data)} weekly records from NFL API")
            else:
                print(f"  Using cached weekly data with {len(weekly_data)} records")
            
            print(f"  Available columns: {list(weekly_data.columns)}")
            
            # Check if player_id column exists
            if 'player_id' in weekly_data.columns:
                rb_data = weekly_data[
                    (weekly_data['position'] == 'RB') & 
                    (weekly_data['player_id'] == player_id)
                ].copy()
                print(f"  Found {len(rb_data)} records for player_id {player_id}")
                
                # If no records found by player_id, try name/team matching as fallback
                if len(rb_data) == 0:
                    print(f"  No records found for player_id {player_id}, trying name/team matching...")
                    rb_data = weekly_data[
                        (weekly_data['position'] == 'RB') & 
                        (weekly_data['recent_team'] == master_row.get('norm_team_master', 'UNK'))
                    ].copy()
                    
                    # Filter by similar names if multiple matches
                    if len(rb_data) > 1:
                        rb_data = rb_data[
                            rb_data['player_name'].str.contains(
                                master_row.get('clean_name_master', '').split()[0], 
                                case=False, 
                                na=False
                            )
                        ]
                        print(f"  After name filtering: {len(rb_data)} records")
            else:
                print("  Warning: 'player_id' column not found in NFL API data")
                rb_data = pd.DataFrame()
        else:
            print(f"  No player_id available, trying name/team matching")
            # Fallback: try to match by name and team
            if weekly_data is None:
                weekly_data = nfl.import_weekly_data([2023, 2024])
            
            rb_data = weekly_data[
                (weekly_data['position'] == 'RB') & 
                (weekly_data['recent_team'] == master_row.get('norm_team_master', 'UNK'))
            ].copy()
            
            # Filter by similar names if multiple matches
            if len(rb_data) > 1:
                rb_data = rb_data[
                    rb_data['player_name'].str.contains(
                        master_row.get('clean_name_master', '').split()[0], 
                        case=False, 
                        na=False
                    )
                ]
        
        if len(rb_data) == 0:
            print(f"No historical data found for {rb_name}, using league average features as fallback")
            # Use league average for RBs as fallback
            rb_data = weekly_data[weekly_data['position'] == 'RB'].copy()
            avg_features = rb_data.mean(numeric_only=True)
            print(f"Using league average features for {rb_name}")
        else:
            # Use player's recent data
            recent_data = rb_data.sort_values(['season', 'week'], ascending=False).head(3)
            avg_features = recent_data.mean(numeric_only=True)
        
        # Add current context
        avg_features['season'] = current_season
        avg_features['week'] = current_week
        avg_features['opponent_team'] = master_row.get('Opponent', 'UNK')  # Get opponent from DK slate
        
        # Create derived features for RBs
        avg_features['yards_per_carry'] = (avg_features['rushing_yards'] / avg_features['carries']) if avg_features['carries'] > 0 else 0
        avg_features['yards_per_reception'] = (avg_features['receiving_yards'] / avg_features['receptions']) if avg_features['receptions'] > 0 else 0
        avg_features['yards_per_touch'] = ((avg_features['rushing_yards'] + avg_features['receiving_yards']) / 
                                          (avg_features['carries'] + avg_features['receptions'])) if (avg_features['carries'] + avg_features['receptions']) > 0 else 0
        avg_features['total_yards'] = avg_features['rushing_yards'] + avg_features['receiving_yards']
        avg_features['total_tds'] = avg_features['rushing_tds'] + avg_features['receiving_tds']
        avg_features['total_touches'] = avg_features['carries'] + avg_features['receptions']
        avg_features['touchdown_rate'] = (avg_features['total_tds'] / avg_features['total_touches']) if avg_features['total_touches'] > 0 else 0
        
        # Week of season patterns
        avg_features['early_season'] = 1 if current_week <= 4 else 0
        avg_features['mid_season'] = 1 if (current_week > 4 and current_week <= 12) else 0
        avg_features['late_season'] = 1 if current_week > 12 else 0
        avg_features['week_progression'] = current_week / 18
        
        # Add team context
        avg_features['recent_team'] = master_row.get('norm_team_master', 'UNK')
        
        # Cap extreme values to prevent model issues
        avg_features['yards_per_carry'] = min(max(avg_features['yards_per_carry'], 0), 15)
        avg_features['yards_per_reception'] = min(max(avg_features['yards_per_reception'], 0), 25)
        avg_features['yards_per_touch'] = min(max(avg_features['yards_per_touch'], 0), 20)
        avg_features['touchdown_rate'] = min(max(avg_features['touchdown_rate'], 0), 1)
        
        print(f"Found {len(rb_data)} historical performances")
        return avg_features
        
    except Exception as e:
        print(f"Error getting features for {rb_name}: {e}")
        return None

def encode_rb_features(rb_features, encoders, feature_cols):
    """Encode RB features for model prediction."""
    print("Encoding features for model input...")
    
    # Mapping for DK slate to nfl_data_py team abbreviations
    team_mapping = {
        'LAR': 'LA',  # DraftKings uses LAR, nfl_data_py uses LA
        'LVR': 'LV',  # DraftKings might use LVR for Las Vegas
        # Add other mappings as needed
    }
    
    feature_vector = []
    
    for feature in feature_cols:
        if feature in rb_features:
            feature_vector.append(rb_features[feature])
        elif feature == 'team_encoded':
            team = rb_features.get('recent_team', 'UNK')
            # Apply team mapping if needed
            team = team_mapping.get(team, team)
            if 'team' in encoders and team in encoders['team'].classes_:
                feature_vector.append(encoders['team'].transform([team])[0])
            else:
                print(f"Team {team} not in training data or encoder missing, using default encoding")
                feature_vector.append(0)
        elif feature == 'opponent_encoded':
            opponent = rb_features.get('opponent_team', 'UNK')
            # Apply team mapping if needed
            opponent = team_mapping.get(opponent, opponent)
            if 'opponent' in encoders and opponent in encoders['opponent'].classes_:
                feature_vector.append(encoders['opponent'].transform([opponent])[0])
            else:
                print(f"Opponent {opponent} not in training data or encoder missing, using default encoding")
                feature_vector.append(0)
        elif feature == 'season_type_encoded':
            feature_vector.append(0)  # Assume regular season
        else:
            feature_vector.append(0)
    
    return np.array(feature_vector).reshape(1, -1)

def make_rb_predictions(matched_rbs, models, encoders, feature_cols, current_season=2025, current_week=1, weekly_data=None):
    """Make predictions for all matched RBs using ensemble models."""
    print("Making RB predictions...")
    
    predictions = []
    
    for _, rb_row in matched_rbs.iterrows():
        # Handle different column name formats from merged data
        rb_name = rb_row.get('Name_dk', 'Unknown')
        team = rb_row.get('TeamAbbrev_dk', 'UNK')
        salary = rb_row.get('Salary_dk', 0)
        opponent = rb_row.get('Opponent', 'UNK')  # Get opponent from DK slate
        player_id = rb_row.get('player_id', 'Unknown')  # NFL API player_id from master sheet
        
        print(f"\nProcessing {rb_name} ({team}) vs {opponent} - Salary: ${salary:,} - ID: {player_id}")
        
        # Get RB features
        rb_features = get_rb_features_from_master(rb_row, current_season, current_week, weekly_data)
        
        if rb_features is None:
            print(f"Could not get features for {rb_name} - skipping")
            continue
        
        # Encode features
        feature_vector = encode_rb_features(rb_features, encoders, feature_cols)
        
        # Make predictions with all models
        model_predictions = {}
        for model_name, model in models.items():
            try:
                pred = model.predict(feature_vector)[0]
                model_predictions[model_name] = pred
                print(f"  {model_name}: {pred:.2f} points")
            except Exception as e:
                print(f"  Error with {model_name}: {e}")
                model_predictions[model_name] = 0
        
        # Calculate ensemble prediction (average of all models)
        valid_predictions = [p for p in model_predictions.values() if p > 0]
        if valid_predictions:
            ensemble_prediction = np.mean(valid_predictions)
            predicted_points = np.clip(ensemble_prediction, 0, 50)
        else:
            predicted_points = 0
        
        # Calculate value
        value = (predicted_points / (salary / 1000)) if salary > 0 else 0
        
        predictions.append({
            'Name': rb_name,
            'Team': team,
            'Opponent': opponent,
            'Position': 'RB',
            'Salary': salary,
            'Player_ID': player_id,
            'Predicted_Points': round(predicted_points, 2),
            'Value': round(value, 3),
            'Recent_Avg_Points': round(rb_features.get('fantasy_points', 0), 2),
            'Recent_Avg_Yards': round(rb_features.get('total_yards', 0), 1),
            'Recent_Avg_TDs': round(rb_features.get('total_tds', 0), 1),
            'Match_Status': 'Matched' if pd.notna(player_id) else 'Unmatched',
            'LightGBM': round(model_predictions.get('lightgbm', 0), 2),
            'CatBoost': round(model_predictions.get('catboost', 0), 2),
            'RandomForest': round(model_predictions.get('random_forest', 0), 2),
            'GradientBoosting': round(model_predictions.get('gradient_boosting', 0), 2)
        })
        
        print(f"  Ensemble Predicted: {predicted_points:.2f} points")
        print(f"  Value: {value:.3f} pts/$1000")
    
    return pd.DataFrame(predictions)

def load_models_and_encoders(models_dir):
    """Load the trained models and encoders."""
    print(f"Loading models from {models_dir}")
    
    models = {}
    encoders = None
    
    try:
        # Load encoders
        encoders_path = os.path.join(models_dir, "encoders.pkl")
        with open(encoders_path, 'rb') as f:
            encoders = pickle.load(f)
        print("Encoders loaded successfully")
        
        # Load all models
        model_files = {
            'lightgbm': 'lightgbm_model.pkl',
            'catboost': 'catboost_model.pkl',
            'random_forest': 'random_forest_model.pkl',
            'gradient_boosting': 'gradient_boosting_model.pkl'
        }
        
        for model_name, filename in model_files.items():
            model_path = os.path.join(models_dir, filename)
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    models[model_name] = pickle.load(f)
                print(f"  {model_name} model loaded")
            else:
                print(f"  Warning: {filename} not found")
        
        if not models:
            print("No models loaded!")
            return None, None
            
        print(f"Loaded {len(models)} models successfully")
        return models, encoders
        
    except Exception as e:
        print(f"Error loading models/encoders: {e}")
        return None, None

def main():
    """Main function to run RB predictions."""
    print("RB Model Inference Script (Advanced Ensemble)")
    print("=" * 60)
    
    # Configuration
    models_dir = "PositionModel/RB_Model_Advanced"
    feature_cols_path = os.path.join(models_dir, "feature_info.json")
    dk_slate_path = "../../data/raw/dk_slates/2025/DKSalaries_20250824.csv"  # Use 2025 DK slate to match master sheet
    master_sheet_path = "../../data/processed/master_sheet_2025.csv"  # Use master sheet for matching
    current_season = 2025
    current_week = 1
    
    # Check if files exist
    if not os.path.exists(models_dir):
        print(f"Models directory not found: {models_dir}")
        return
    
    if not os.path.exists(feature_cols_path):
        print(f"Feature info file not found: {feature_cols_path}")
        return
    
    if not os.path.exists(master_sheet_path):
        print(f"Master sheet not found: {master_sheet_path}")
        return
    
    # Load models and encoders
    models, encoders = load_models_and_encoders(models_dir)
    if models is None or encoders is None:
        return
    
    # Load feature columns
    with open(feature_cols_path, 'r') as f:
        feature_info = json.load(f)
        feature_cols = feature_info['feature_names']
    
    print(f"Loaded {len(feature_cols)} features for prediction")
    
    # Load master sheet
    print(f"Loading master sheet from {master_sheet_path}")
    master_sheet = pd.read_csv(master_sheet_path)
    print(f"Loaded {len(master_sheet)} players from master sheet")
    
    # Load NFL weekly data once
    print("Loading NFL weekly data (this may take a moment)...")
    weekly_data = nfl.import_weekly_data([2023, 2024])
    print(f"Loaded {len(weekly_data)} weekly records from NFL API")
    
    # Load DK slate
    try:
        dk_slate = load_dk_slate(dk_slate_path)
    except Exception as e:
        print(f"Error loading DK slate: {e}")
        return
    
    # Match DK slate with master sheet
    print("Matching DK slate with master sheet...")
    matched_rbs = pd.merge(
        dk_slate, 
        master_sheet[master_sheet['Position'] == 'RB'], 
        on='join_key', 
        how='inner',
        suffixes=('_dk', '_master')
    )
    
    print(f"Found {len(matched_rbs)} matched RBs")
    
    if len(matched_rbs) == 0:
        print("No RBs matched between DK slate and master sheet")
        return
    
    # Make predictions with cached weekly data
    predictions = make_rb_predictions(matched_rbs, models, encoders, feature_cols, current_season, current_week, weekly_data)
    
    if predictions is not None and len(predictions) > 0:
        print("\n" + "=" * 60)
        print("PREDICTIONS COMPLETE")
        print("=" * 60)
        print(predictions.to_string(index=False))
        
        # Save predictions
        output_path = "rb_predictions.csv"
        predictions.to_csv(output_path, index=False)
        print(f"\nPredictions saved to {output_path}")
        
        # Show top predictions by value
        print("\n" + "=" * 60)
        print("TOP 10 RBs BY VALUE (pts/$1000)")
        print("=" * 60)
        top_by_value = predictions.nlargest(10, 'Value')[['Name', 'Team', 'Opponent', 'Salary', 'Predicted_Points', 'Value']]
        print(top_by_value.to_string(index=False))
        
        # Show top predictions by points
        print("\n" + "=" * 60)
        print("TOP 10 RBs BY PREDICTED POINTS")
        print("=" * 60)
        top_by_points = predictions.nlargest(10, 'Predicted_Points')[['Name', 'Team', 'Opponent', 'Salary', 'Predicted_Points', 'Value']]
        print(top_by_points.to_string(index=False))
        
    else:
        print("No predictions generated")

if __name__ == "__main__":
    main()
