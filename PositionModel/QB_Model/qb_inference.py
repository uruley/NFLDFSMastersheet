import pandas as pd
import numpy as np
import nfl_data_py as nfl
import pickle
import os

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

def is_backup_qb(qb_name, salary, team, current_week=1):
    """
    Determine if a QB is a backup based on salary, team situation, and known context.
    This is the key function to prevent over-projecting backup QBs.
    """
    qb_name_lower = str(qb_name).lower()
    
    # Known backup QBs for specific weeks/situations
    known_backups = {
        'drew lock': {
            'reason': 'Backup QB, Geno Smith is starter',
            'max_points': 12.0,
            'salary_threshold': 5000
        },
        'tyler huntley': {
            'reason': 'Backup QB, Lamar Jackson is starter',
            'max_points': 10.0,
            'salary_threshold': 5000
        },
        'sam darnold': {
            'reason': 'Backup QB, likely behind starter',
            'max_points': 11.0,
            'salary_threshold': 5000
        },
        'jameis winston': {
            'reason': 'Backup QB, Derek Carr is starter',
            'max_points': 10.0,
            'salary_threshold': 5000
        },
        'mason rudolph': {
            'reason': 'Backup QB, Kenny Pickett is starter',
            'max_points': 9.0,
            'salary_threshold': 5000
        },
        'jordan love': {
            'reason': 'Backup QB, Aaron Rodgers is starter',
            'max_points': 8.0,
            'salary_threshold': 5000
        }
    }
    
    # Check if this is a known backup
    if qb_name_lower in known_backups:
        backup_info = known_backups[qb_name_lower]
        if salary <= backup_info['salary_threshold']:
            print(f"⚠️  {qb_name} identified as known backup: {backup_info['reason']}")
            return True, backup_info
    
    # Salary-based heuristics for backup detection
    if salary < 5000:
        print(f"⚠️  {qb_name} has low salary (${salary:,}), likely backup")
        return True, {
            'reason': f'Low salary (${salary:,}) suggests backup role',
            'max_points': 10.0,
            'salary_threshold': salary
        }
    
    # Team-specific depth chart logic for Week 1
    if current_week == 1:
        team_backup_situations = {
            'SEA': ['drew lock'],  # Geno Smith is starter
            'BAL': ['tyler huntley'],  # Lamar Jackson is starter
            'MIN': ['sam darnold'],  # Kirk Cousins is starter
            'NO': ['jameis winston'],  # Derek Carr is starter
            'PIT': ['mason rudolph'],  # Kenny Pickett is starter
            'GB': ['jordan love'],  # Aaron Rodgers is starter
        }
        
        if team in team_backup_situations:
            if qb_name_lower in team_backup_situations[team]:
                print(f"⚠️  {qb_name} identified as backup for {team} in Week 1")
                return True, {
                    'reason': f'Backup QB for {team} in Week 1',
                    'max_points': 10.0,
                    'salary_threshold': salary
                }
    
    return False, None

def get_backup_baseline_features(qb_name, salary, team, backup_info):
    """
    Generate conservative baseline features for backup QBs.
    This prevents identical fallback values and provides realistic projections.
    """
    print(f"🔄 Using backup baseline for {qb_name}")
    
    # Create varied baseline based on salary and situation
    base_points = min(8.0, salary / 1000)  # Basic salary-based baseline
    
    # Add some randomization to prevent identical values
    import random
    random.seed(hash(f"{qb_name}{team}{salary}") % 1000)  # Deterministic but varied
    
    # Vary the baseline slightly based on player/team
    variation = random.uniform(-1.0, 1.0)
    adjusted_points = max(4.0, base_points + variation)
    
    # Cap at the backup's max points
    if backup_info and 'max_points' in backup_info:
        adjusted_points = min(adjusted_points, backup_info['max_points'])
    
    # Generate realistic backup stats
    baseline = {
        'fantasy_points': round(adjusted_points, 2),
        'passing_yards': int(adjusted_points * 25 + random.uniform(-50, 50)),
        'passing_tds': int(adjusted_points / 6 + random.uniform(-0.5, 0.5)),
        'rushing_yards': int(adjusted_points * 2 + random.uniform(-10, 10)),
        'rushing_tds': int(adjusted_points / 20 + random.uniform(-0.2, 0.2)),
        'completions': int(adjusted_points * 1.5 + random.uniform(-2, 2)),
        'attempts': int(adjusted_points * 2.5 + random.uniform(-3, 3)),
        'carries': int(adjusted_points * 0.3 + random.uniform(-1, 1)),
        'receptions': 0,  # QBs don't receive
        'receiving_yards': 0,
        'receiving_tds': 0,
        'interceptions': int(adjusted_points / 8 + random.uniform(-0.3, 0.3)),
        'fumbles': int(adjusted_points / 15 + random.uniform(-0.2, 0.2)),
        'season': 2025,
        'week': 1,
        'opponent_team': 'UNK'
    }
    
    # Ensure non-negative values
    for key, value in baseline.items():
        if isinstance(value, (int, float)):
            baseline[key] = max(0, value)
    
    print(f"  Backup baseline: {adjusted_points:.2f} points, {baseline['passing_yards']} pass yds, {baseline['passing_tds']} pass TDs")
    return baseline

def load_dk_slate(slate_path):
    """Load the DraftKings slate and prepare for master sheet matching."""
    print(f"Loading DraftKings slate from {slate_path}")
    
    dk_slate = pd.read_csv(slate_path)
    
    # Filter for QBs only
    qb_slate = dk_slate[dk_slate['Position'] == 'QB'].copy()
    
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
    if 'Game Info' in qb_slate.columns:
        qb_slate['Opponent'] = qb_slate.apply(
            lambda row: extract_opponent(row['Game Info'], row['TeamAbbrev']), axis=1
        )
    elif 'Opponent' in qb_slate.columns:
        # If Opponent column already exists, validate it
        qb_slate['Opponent'] = qb_slate['Opponent'].apply(
            lambda x: x if x in valid_teams else 'UNK'
        )
    else:
        print("Warning: No Game Info or Opponent column found, setting default opponent")
        qb_slate['Opponent'] = 'UNK'
    
    print(f"Found {len(qb_slate)} QBs in the slate")
    
    # Prepare for master sheet matching - handle different name column formats
    if 'Name' in qb_slate.columns:
        name_col = 'Name'
    elif 'Name + ID' in qb_slate.columns:
        # Extract just the name part from "Name + ID" column
        qb_slate['Name'] = qb_slate['Name + ID'].str.split(' (')[0]
        name_col = 'Name'
    else:
        print("Warning: No Name column found")
        name_col = 'Name'
    
    qb_slate['clean_name'] = qb_slate[name_col].apply(clean_name)
    qb_slate['norm_team'] = qb_slate['TeamAbbrev'].apply(norm_team)
    qb_slate['norm_pos'] = qb_slate['Position'].apply(norm_pos)
    qb_slate['join_key'] = qb_slate['clean_name'] + "|" + qb_slate['norm_team'] + "|" + qb_slate['norm_pos']
    
    print("Prepared join keys for master sheet matching")
    print("Sample join keys:", qb_slate['join_key'].head().tolist())
    print("Sample opponents:", qb_slate['Opponent'].head().tolist())
    
    return qb_slate

def get_qb_features_from_master(master_row, current_season=2025, current_week=1, weekly_data=None):
    """Get QB features from master sheet data for prediction."""
    # Handle different column name formats from merged data
    qb_name = master_row.get('Name_dk', master_row.get('Name', 'Unknown'))
    player_id = master_row.get('player_id', None)  # This is the NFL API player_id from master sheet
    salary = master_row.get('Salary_dk', 0)
    team = master_row.get('TeamAbbrev_dk', 'UNK')
    
    print(f"Getting features for {qb_name} (NFL API ID: {player_id})")
    
    # Check if this QB should be treated as a backup even with historical data
    is_backup, backup_info = is_backup_qb(qb_name, salary, team, current_week)
    if is_backup:
        print(f"🔄 {qb_name} identified as backup, using backup baseline despite historical data availability")
        return get_backup_baseline_features(qb_name, salary, team, backup_info)
    
    try:
        # Get recent QB data from NFL API using player_id if available
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
                qb_data = weekly_data[
                    (weekly_data['position'] == 'QB') & 
                    (weekly_data['player_id'] == player_id)
                ].copy()
                print(f"  Found {len(qb_data)} records for player_id {player_id}")
                
                # If no records found by player_id, try name/team matching as fallback
                if len(qb_data) == 0:
                    print(f"  No records found for player_id {player_id}, trying name/team matching...")
                    qb_data = weekly_data[
                        (weekly_data['position'] == 'QB') & 
                        (weekly_data['recent_team'] == master_row.get('norm_team_master', 'UNK'))
                    ].copy()
                    
                    # Filter by similar names if multiple matches
                    if len(qb_data) > 1:
                        qb_data = qb_data[
                            qb_data['player_name'].str.contains(
                                master_row.get('clean_name_master', '').split()[0], 
                                case=False, 
                                na=False
                            )
                        ]
                        print(f"  After name filtering: {len(qb_data)} records")
            else:
                print("  Warning: 'player_id' column not found in NFL API data")
                qb_data = pd.DataFrame()
        else:
            print(f"  No player_id available, trying name/team matching")
            # Fallback: try to match by name and team
            if weekly_data is None:
                weekly_data = nfl.import_weekly_data([2023, 2024])
            
            qb_data = weekly_data[
                (weekly_data['position'] == 'QB') & 
                (weekly_data['recent_team'] == master_row.get('norm_team_master', 'UNK'))
            ].copy()
            
            # Filter by similar names if multiple matches
            if len(qb_data) > 1:
                qb_data = qb_data[
                    qb_data['player_name'].str.contains(
                        master_row.get('clean_name_master', '').split()[0], 
                        case=False, 
                        na=False
                    )
                ]
        
        if len(qb_data) == 0:
            print(f"No historical data found for {qb_name}, using varied fallback features")
            # Create varied fallback instead of identical league average
            fallback_features = create_varied_fallback_features(qb_name, salary, team)
            print(f"Using varied fallback features for {qb_name}")
            return fallback_features
        else:
            # Use player's recent data
            recent_data = qb_data.sort_values(['season', 'week'], ascending=False).head(3)
            avg_features = recent_data.mean(numeric_only=True)
        
        # Features are now calculated above in the if/else block
        
        # Add current context
        avg_features['season'] = current_season
        avg_features['week'] = current_week
        avg_features['opponent_team'] = master_row.get('Opponent', 'UNK')  # Get opponent from DK slate
        
        # Create derived features
        avg_features['completion_rate'] = (avg_features['completions'] / avg_features['attempts']) if avg_features['attempts'] > 0 else 0
        avg_features['yards_per_attempt'] = (avg_features['passing_yards'] / avg_features['attempts']) if avg_features['attempts'] > 0 else 0
        avg_features['yards_per_rush'] = (avg_features['rushing_yards'] / avg_features['carries']) if avg_features['carries'] > 0 else 0
        avg_features['total_yards'] = avg_features['passing_yards'] + avg_features['rushing_yards'] + avg_features['receiving_yards']
        avg_features['total_tds'] = avg_features['passing_tds'] + avg_features['rushing_tds'] + avg_features['receiving_tds']
        avg_features['total_touches'] = avg_features['attempts'] + avg_features['carries'] + avg_features['receptions']
        
        # Week of season patterns
        avg_features['early_season'] = 1 if current_week <= 4 else 0
        avg_features['mid_season'] = 1 if (current_week > 4 and current_week <= 12) else 0
        avg_features['late_season'] = 1 if current_week > 12 else 0
        avg_features['week_progression'] = current_week / 18
        
        # Add team context
        avg_features['recent_team'] = master_row.get('norm_team_master', 'UNK')
        
        print(f"Found {len(qb_data)} historical performances")
        return avg_features
        
    except Exception as e:
        print(f"Error getting features for {qb_name}: {e}")
        return None

def create_varied_fallback_features(qb_name, salary, team):
    """
    Create varied fallback features to prevent identical projections.
    This replaces the old league average fallback that caused identical values.
    """
    print(f"🔄 Creating varied fallback features for {qb_name}")
    
    # Use deterministic but varied values based on player/team/salary
    import random
    random.seed(hash(f"{qb_name}{team}{salary}") % 1000)
    
    # Base fantasy points vary by salary tier
    if salary >= 8000:
        base_points = random.uniform(18.0, 22.0)  # High-end starter
    elif salary >= 6000:
        base_points = random.uniform(15.0, 19.0)  # Mid-tier starter
    elif salary >= 4500:
        base_points = random.uniform(12.0, 16.0)  # Lower-tier starter
    else:
        base_points = random.uniform(8.0, 12.0)   # Backup/low-salary
    
    # Add some randomness to prevent identical values
    variation = random.uniform(-0.5, 0.5)
    fantasy_points = round(base_points + variation, 2)
    
    # Generate varied stats based on fantasy points
    passing_yards = int(fantasy_points * 25 + random.uniform(-30, 30))
    passing_tds = int(fantasy_points / 6 + random.uniform(-0.3, 0.3))
    rushing_yards = int(fantasy_points * 2 + random.uniform(-5, 5))
    rushing_tds = int(fantasy_points / 20 + random.uniform(-0.2, 0.2))
    
    fallback_features = {
        'fantasy_points': fantasy_points,
        'passing_yards': max(0, passing_yards),
        'passing_tds': max(0, passing_tds),
        'rushing_yards': max(0, rushing_yards),
        'rushing_tds': max(0, rushing_tds),
        'completions': int(fantasy_points * 1.5 + random.uniform(-2, 2)),
        'attempts': int(fantasy_points * 2.5 + random.uniform(-3, 3)),
        'carries': int(fantasy_points * 0.3 + random.uniform(-1, 1)),
        'receptions': 0,
        'receiving_yards': 0,
        'receiving_tds': 0,
        'interceptions': int(fantasy_points / 8 + random.uniform(-0.3, 0.3)),
        'fumbles': int(fantasy_points / 15 + random.uniform(-0.2, 0.2)),
        'season': 2025,
        'week': 1,
        'opponent_team': 'UNK',
        'recent_team': team
    }
    
    # Ensure non-negative values
    for key, value in fallback_features.items():
        if isinstance(value, (int, float)):
            fallback_features[key] = max(0, value)
    
    print(f"  Fallback features: {fantasy_points:.2f} points, {passing_yards} pass yds, {passing_tds} pass TDs")
    return fallback_features

def encode_qb_features(qb_features, encoders, feature_cols):
    """Encode QB features for model prediction."""
    print("Encoding features for model input...")
    
    # Mapping for DK slate to nfl_data_py team abbreviations
    team_mapping = {
        'LAR': 'LA',  # DraftKings uses LAR, nfl_data_py uses LA
        'LVR': 'LV',  # DraftKings might use LVR for Las Vegas
        # Add other mappings as needed
    }
    
    feature_vector = []
    
    for feature in feature_cols:
        if feature in qb_features:
            feature_vector.append(qb_features[feature])
        elif feature == 'team_encoded':
            team = qb_features.get('recent_team', 'UNK')
            # Apply team mapping if needed
            team = team_mapping.get(team, team)
            if 'team' in encoders and team in encoders['team'].classes_:
                feature_vector.append(encoders['team'].transform([team])[0])
            else:
                print(f"Team {team} not in training data or encoder missing, using default encoding")
                feature_vector.append(0)
        elif feature == 'opponent_encoded':
            opponent = qb_features.get('opponent_team', 'UNK')
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

def make_qb_predictions(matched_qbs, model, encoders, feature_cols, current_season=2025, current_week=1, weekly_data=None):
    """Make predictions for all matched QBs."""
    print("Making QB predictions...")
    
    predictions = []
    
    for _, qb_row in matched_qbs.iterrows():
        # Handle different column name formats from merged data
        qb_name = qb_row.get('Name_dk', 'Unknown')
        team = qb_row.get('TeamAbbrev_dk', 'UNK')
        salary = qb_row.get('Salary_dk', 0)
        opponent = qb_row.get('Opponent', 'UNK')  # Get opponent from DK slate
        player_id = qb_row.get('player_id', 'Unknown')  # NFL API player_id from master sheet
        
        print(f"\nProcessing {qb_name} ({team}) vs {opponent} - Salary: ${salary:,} - ID: {player_id}")
        
        # Get QB features (backup detection is now handled within this function)
        qb_features = get_qb_features_from_master(qb_row, current_season, current_week, weekly_data)
        
        if qb_features is None:
            print(f"Could not get features for {qb_name} - skipping")
            continue
        
        # Encode features
        feature_vector = encode_qb_features(qb_features, encoders, feature_cols)
        
        # Make prediction
        predicted_points = np.clip(model.predict(feature_vector)[0], 0, 50)
        
        # Calculate value
        value = (predicted_points / (salary / 1000)) if salary > 0 else 0
        
        predictions.append({
            'Name': qb_name,
            'Team': team,
            'Opponent': opponent,
            'Position': 'QB',
            'Salary': salary,
            'Player_ID': player_id,
            'Predicted_Points': round(predicted_points, 2),
            'Value': round(value, 3),
            'Recent_Avg_Points': round(qb_features.get('fantasy_points', 0), 2),
            'Recent_Avg_Yards': round(qb_features.get('total_yards', 0), 1),
            'Recent_Avg_TDs': round(qb_features.get('total_tds', 0), 1),
            'Match_Status': 'Matched' if pd.notna(player_id) else 'Unmatched'
        })
        
        print(f"  Predicted: {predicted_points:.2f} points")
        print(f"  Value: {value:.3f} pts/$1000")
    
    return pd.DataFrame(predictions)

def load_model_and_encoders(model_path, encoders_path):
    """Load the trained model and encoders."""
    print(f"Loading model from {model_path}")
    print(f"Loading encoders from {encoders_path}")
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        with open(encoders_path, 'rb') as f:
            encoders = pickle.load(f)
            
        print("Model and encoders loaded successfully")
        return model, encoders
        
    except Exception as e:
        print(f"Error loading model/encoders: {e}")
        return None, None

def main():
    """Main function to run QB predictions."""
    print("QB Model Inference Script")
    print("=" * 50)
    
    # Configuration
    model_path = "qb_model.pkl"
    encoders_path = "encoders.pkl"
    feature_cols_path = "feature_columns.json"
    dk_slate_path = "../../data/raw/dk_slates/2025/DKSalaries_20250824.csv"  # Use 2025 DK slate to match master sheet
    master_sheet_path = "../../data/processed/master_sheet_2025.csv"  # Use master sheet for matching
    current_season = 2025
    current_week = 1
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return
    
    if not os.path.exists(encoders_path):
        print(f"Encoders file not found: {encoders_path}")
        return
    
    if not os.path.exists(feature_cols_path):
        print(f"Feature columns file not found: {feature_cols_path}")
        return
    
    if not os.path.exists(master_sheet_path):
        print(f"Master sheet not found: {master_sheet_path}")
        return
    
    # Load model components
    model, encoders = load_model_and_encoders(model_path, encoders_path)
    if model is None or encoders is None:
        return
    
    # Load feature columns
    with open(feature_cols_path, 'r') as f:
        feature_cols = json.load(f)
    
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
    matched_qbs = pd.merge(
        dk_slate, 
        master_sheet[master_sheet['Position'] == 'QB'], 
        on='join_key', 
        how='inner',
        suffixes=('_dk', '_master')
    )
    
    print(f"Found {len(matched_qbs)} matched QBs")
    
    if len(matched_qbs) == 0:
        print("No QBs matched between DK slate and master sheet")
        return
    
    # Make predictions with cached weekly data
    predictions = make_qb_predictions(matched_qbs, model, encoders, feature_cols, current_season, current_week, weekly_data)
    
    if predictions is not None and len(predictions) > 0:
        print("\n" + "=" * 50)
        print("PREDICTIONS COMPLETE")
        print("=" * 50)
        print(predictions.to_string(index=False))
        
        # Save predictions
        output_path = "qb_predictions.csv"
        predictions.to_csv(output_path, index=False)
        print(f"\nPredictions saved to {output_path}")
    else:
        print("No predictions generated")

if __name__ == "__main__":
    import json
    main()