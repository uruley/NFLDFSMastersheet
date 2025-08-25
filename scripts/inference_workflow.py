#!/usr/bin/env python3
"""
Inference workflow for NFL DFS prediction.
Loads trained model and makes predictions on new DraftKings slates.
"""

import argparse
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
import nfl_data_py as nfl
from typing import Dict, List, Tuple

# Import unified feature engineering
from feature_engineering import (
    calculate_derived_features, 
    add_team_features, 
    scale_features, 
    create_feature_vector as fe_create_feature_vector,
    get_all_teams_from_training_data
)


def load_trained_model(model_path: str):
    """Load the trained model from disk."""
    print(f"Loading trained model from: {model_path}")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"Model loaded successfully")
    return model


def load_model_metadata(model_dir: str) -> Dict:
    """Load model metadata (feature names, etc.) from the model directory."""
    print(f"Loading model metadata from: {model_dir}")
    
    metadata = {}
    
    # Load feature importance to get feature names
    feature_importance_path = Path(model_dir) / "feature_importance.csv"
    if feature_importance_path.exists():
        feature_df = pd.read_csv(feature_importance_path)
        metadata['feature_names'] = feature_df['feature'].tolist()
        print(f"  Loaded {len(metadata['feature_names'])} feature names")
    
    # Load metrics
    metrics_path = Path(model_dir) / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            metadata['metrics'] = json.load(f)
        print(f"  Loaded model metrics")
    
    return metadata


def load_feature_stats(model_dir: str) -> Dict:
    """Load feature scaling statistics from training."""
    feature_stats_path = Path(model_dir) / "feature_stats.json"
    if feature_stats_path.exists():
        with open(feature_stats_path, 'r') as f:
            return json.load(f)
    else:
        print("Warning: No feature stats found - predictions may be incorrect")
        return {}


def load_dk_slate(slate_path: str) -> pd.DataFrame:
    """Load DraftKings slate CSV file."""
    print(f"Loading DraftKings slate from: {slate_path}")
    
    df = pd.read_csv(slate_path)
    print(f"  Loaded {len(df)} players")
    print(f"  Columns: {list(df.columns)}")
    
    return df


def clean_name(name):
    """Apply name cleaning rules: lowercase → remove punctuation → collapse spaces → drop suffix"""
    if pd.isna(name):
        return ""
    
    import re
    
    # Convert to string and lowercase
    name = str(name).lower()
    
    # Remove punctuation
    name = re.sub(r'[^\w\s]', '', name)
    
    # Collapse multiple spaces to single space
    name = re.sub(r'\s+', ' ', name)
    
    # Strip leading/trailing spaces
    name = name.strip()
    
    # Drop common suffixes at the end
    suffixes = [' jr', ' sr', ' ii', ' iii', ' iv', ' v']
    for suffix in suffixes:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break
    
    return name


def norm_team(team):
    """Normalize team abbreviations"""
    if pd.isna(team):
        return ""
    
    team = str(team).strip()
    
    # Team mapping rules
    team_map = {
        'JAX': 'JAC',
        'WSH': 'WAS', 
        'OAK': 'LV',
        'SD': 'LAC',
        'LA': 'LAR',
        'STL': 'LAR'
    }
    
    return team_map.get(team, team)


def norm_pos(position):
    """Normalize position: D/ST or DEF → DST, else UPPERCASE"""
    if pd.isna(position):
        return ""
    
    pos = str(position).strip()
    
    if pos in ['D/ST', 'DEF']:
        return 'DST'
    else:
        return pos.upper()


def compute_join_key(row, is_dk=True):
    """Compute join key based on data source"""
    if is_dk:
        # For DK data
        if row['Position'] == 'DST':
            # Special handling for DST: TEAM-<abbr>|<abbr>|DST
            team = norm_team(row['TeamAbbrev'])
            return f"{team}-{team}|{team}|DST"
        else:
            # Regular players: clean_name|norm_team|norm_pos
            clean = clean_name(row['Name'])
            team = norm_team(row['TeamAbbrev'])
            pos = norm_pos(row['Position'])
            return f"{clean}|{team}|{pos}"
    else:
        # For roster data
        if row['position'] == 'DST':
            # DST teams
            team = norm_team(row['team'])
            return f"{team}-{team}|{team}|DST"
        else:
            # Regular players
            # Try different possible name columns
            name_cols = ['player_name', 'full_name', 'player_name']
            name = None
            for col in name_cols:
                if col in row and pd.notna(row[col]):
                    name = row[col]
                    break
            
            if name is None:
                return ""
                
            clean = clean_name(name)
            team = norm_team(row['team'])
            pos = norm_pos(row['position'])
            return f"{clean}|{team}|{pos}"


def load_aliases(aliases_path: str) -> Dict[str, str]:
    """Load player name aliases for matching."""
    print(f"Loading aliases from: {aliases_path}")
    
    alias_lookup = {}
    
    if not Path(aliases_path).exists():
        print(f"  Warning: Aliases file not found, using empty mapping")
        return alias_lookup
    
    try:
        aliases_df = pd.read_csv(aliases_path)
        print(f"  Aliases: {len(aliases_df)} rows")
        for _, row in aliases_df.iterrows():
            if pd.notna(row['dk_name']) and pd.notna(row['roster_full_name']):
                alias_lookup[row['dk_name']] = row['roster_full_name']
                print(f"    Alias: {row['dk_name']} -> {row['roster_full_name']}")
    except Exception as e:
        print(f"  Error loading aliases: {e}")
    
    return alias_lookup


def load_roster_data(season: int) -> Tuple[pd.DataFrame, Dict]:
    """Load roster data and create lookup dictionary."""
    print(f"Loading roster data for season {season}...")
    
    try:
        # Try to load from master sheet first (most complete)
        master_sheet_path = f"data/processed/master_sheet_{season}.csv"
        if Path(master_sheet_path).exists():
            print(f"  Loading from master sheet: {master_sheet_path}")
            roster_df = pd.read_csv(master_sheet_path)
            
            # Check if master sheet has the expected columns
            if 'player_id' in roster_df.columns and 'player_name' in roster_df.columns:
                # Extract unique roster entries from master sheet
                roster_data = roster_df[['player_id', 'player_name', 'TeamAbbrev', 'Position']].drop_duplicates()
                roster_data = roster_data.rename(columns={'TeamAbbrev': 'team', 'Position': 'position'})
                print(f"  Loaded {len(roster_data)} unique roster entries from master sheet")
            else:
                # Master sheet doesn't have expected columns, fall back to NFL API
                print(f"  Master sheet missing expected columns, falling back to NFL API...")
                raise ValueError("Master sheet missing expected columns")
        else:
            # Fallback to NFL API roster
            print(f"  Master sheet not found, loading from NFL API...")
            raise ValueError("Master sheet not found")
            
    except Exception as e:
        print(f"  Falling back to NFL API roster: {e}")
        try:
            import nfl_data_py as nfl
            roster_df = nfl.import_seasonal_rosters([season])
            
            # Normalize columns
            roster_df = roster_df.rename(columns={
                'player_name': 'player_name',
                'recent_team': 'team',
                'position': 'position',
                'player_id': 'player_id'
            })
            
            # Filter to only keep the columns we need
            roster_data = roster_df[['player_name', 'team', 'position', 'player_id']].copy()
            print(f"  Loaded {len(roster_data)} roster entries from NFL API")
        except Exception as nfl_error:
            print(f"  Error loading from NFL API: {nfl_error}")
            # Final fallback to local roster file
            print(f"  Trying local roster file...")
            local_roster_path = f"data/raw/rosters_{season}_season.csv"
            if Path(local_roster_path).exists():
                roster_df = pd.read_csv(local_roster_path)
                roster_data = roster_df[['player_name', 'team', 'position', 'player_id']].copy()
                print(f"  Loaded {len(roster_data)} roster entries from local file")
            else:
                print(f"  No roster data available for season {season}")
                return pd.DataFrame(), {}
    
    # Apply cleaning rules to roster data
    print(f"  Applying cleaning rules to roster data...")
    roster_data['clean_name'] = roster_data['player_name'].apply(clean_name)
    roster_data['norm_team'] = roster_data['team'].apply(norm_team)
    roster_data['norm_pos'] = roster_data['position'].apply(norm_pos)
    roster_data['join_key'] = roster_data.apply(compute_join_key, axis=1, args=(False,))
    
    # Create lookup dictionary for roster data
    roster_lookup = {}
    for _, row in roster_data.iterrows():
        if row['join_key']:  # Skip empty join keys
            roster_lookup[row['join_key']] = row['player_id']
    
    # Apply manual overrides from xwalk_manual.csv (last priority)
    print(f"  Reading manual crosswalk overrides...")
    try:
        manual_df = pd.read_csv("data/xwalk/xwalk_manual.csv")
        print(f"    Manual overrides: {len(manual_df)} rows")
        for _, row in manual_df.iterrows():
            if row['join_key'] and row['player_id']:
                roster_lookup[row['join_key']] = row['player_id']
                print(f"      Override: {row['join_key']} -> {row['player_id']} ({row['reason']})")
    except FileNotFoundError:
        print(f"    No manual crosswalk file found")
    
    return roster_data, roster_lookup


def match_players_to_roster(dk_slate: pd.DataFrame, aliases: Dict[str, str], roster_lookup: Dict) -> pd.DataFrame:
    """Match DraftKings players to roster using master sheet logic."""
    print("Matching players to roster using master sheet logic...")
    
    # Apply aliases first, then clean
    dk_slate['name_for_cleaning'] = dk_slate['Name'].map(aliases).fillna(dk_slate['Name'])
    dk_slate['clean_name'] = dk_slate['name_for_cleaning'].apply(clean_name)
    dk_slate['norm_team'] = dk_slate['TeamAbbrev'].apply(norm_team)
    dk_slate['norm_pos'] = dk_slate['Position'].apply(norm_pos)
    dk_slate['join_key'] = dk_slate.apply(compute_join_key, axis=1, args=(True,))
    
    # Merge DK data with roster data
    print("  Merging data using join keys...")
    dk_slate['player_id'] = dk_slate['join_key'].map(roster_lookup)
    
    # Build name_team_key for fallback matching
    print("  Building name_team_key for fallback matching...")
    dk_slate['name_team_key'] = dk_slate['clean_name'] + "|" + dk_slate['norm_team']
    
    # Count matches
    total_dk = len(dk_slate)
    matched_count = len(dk_slate[dk_slate['player_id'].notna()])
    match_pct = matched_count/total_dk*100
    
    print(f"  Match summary: {matched_count}/{total_dk} ({match_pct:.1f}%)")
    
    # Show sample matches
    if matched_count > 0:
        print(f"  Sample matches:")
        sample_matches = dk_slate[dk_slate['player_id'].notna()].head(3)[['Name', 'TeamAbbrev', 'Position', 'player_id']]
        for _, row in sample_matches.iterrows():
            print(f"    {row['Name']} ({row['TeamAbbrev']}, {row['Position']}) -> {row['player_id']}")
    
    return dk_slate


def load_aliases(aliases_path: str) -> Dict[str, str]:
    """Load player name aliases for matching."""
    print(f"Loading aliases from: {aliases_path}")
    
    if not Path(aliases_path).exists():
        print(f"  Warning: Aliases file not found, using empty mapping")
        return {}
    
    aliases_df = pd.read_csv(aliases_path)
    aliases = {}
    
    for _, row in aliases_df.iterrows():
        dk_name = clean_name(row.get('dk_name', ''))
        roster_name = clean_name(row.get('roster_name', ''))
        if dk_name and roster_name:
            aliases[dk_name] = roster_name
    
    print(f"  Loaded {len(aliases)} aliases")
    return aliases





def get_player_features(player_id: str, current_season: int, current_week: int) -> Dict:
    """Get player features from NFL API data using player_id."""
    try:
        # Load recent weekly data for the player
        weekly_data = nfl.import_weekly_data([current_season])
        
        # Filter to player by player_id
        player_data = weekly_data[weekly_data['player_id'] == player_id].copy()
        
        print(f"    Found {len(player_data)} rows for player_id {player_id}")
        
        if len(player_data) == 0:
            return {}
        
        # Get most recent data (up to current week)
        recent_data = player_data[player_data['week'] <= current_week].copy()
        print(f"    Recent data (week <= {current_week}): {len(recent_data)} rows")
        
        if len(recent_data) == 0:
            # Use all available data if no recent data
            recent_data = player_data.copy()
            print(f"    Using all available data: {len(recent_data)} rows")
        
        # Aggregate recent performance - use available columns
        features = {}
        
        # Basic stats
        for stat in ['passing_yards', 'passing_tds', 'interceptions', 'rushing_yards', 
                    'rushing_tds', 'receptions', 'receiving_yards', 'receiving_tds']:
            if stat in recent_data.columns:
                features[stat] = recent_data[stat].mean()
            else:
                features[stat] = 0.0
        
        # Handle fumbles - combine different fumble types
        fumbles_lost = 0.0
        for fumble_col in ['sack_fumbles_lost', 'rushing_fumbles_lost', 'receiving_fumbles_lost']:
            if fumble_col in recent_data.columns:
                fumbles_lost += recent_data[fumble_col].mean()
        features['fumbles_lost'] = fumbles_lost
        
        # Usage stats
        for stat in ['targets', 'carries']:
            if stat in recent_data.columns:
                features[stat] = recent_data[stat].mean()
            else:
                features[stat] = 0.0
        
        # Fill NaN values
        for key in features:
            if pd.isna(features[key]):
                features[key] = 0.0
        
        return features
        
    except Exception as e:
        print(f"  Error getting features for player_id {player_id}: {e}")
        return {}


def create_feature_vector(player_features: Dict, position: str, team: str, salary: float, 
                         current_week: int, feature_names: List[str], feature_stats: Dict) -> np.ndarray:
    """Create feature vector for model prediction."""
    # Initialize feature vector with zeros
    feature_vector = np.zeros(len(feature_names))
    
    # Map features to their positions in the feature vector
    feature_map = {name: i for i, name in enumerate(feature_names)}
    
    # Add base features (only the ones the corrected model expects)
    if 'total_touches' in feature_map:
        touches = player_features.get('carries', 0) + player_features.get('targets', 0)
        feature_vector[feature_map['total_touches']] = touches
    
    if 'touchdown_rate' in feature_map:
        tds = player_features.get('passing_tds', 0) + player_features.get('rushing_tds', 0) + player_features.get('receiving_tds', 0)
        touches = max(player_features.get('carries', 0) + player_features.get('targets', 0), 1)
        feature_vector[feature_map['touchdown_rate']] = tds / touches
    
    if 'yards_per_touch' in feature_map:
        yards = player_features.get('passing_yards', 0) + player_features.get('rushing_yards', 0) + player_features.get('receiving_yards', 0)
        touches = max(player_features.get('carries', 0) + player_features.get('targets', 0), 1)
        feature_vector[feature_map['yards_per_touch']] = yards / touches
    
    if 'carry_efficiency' in feature_map:
        carries = player_features.get('carries', 0)
        if carries > 0:
            feature_vector[feature_map['carry_efficiency']] = player_features.get('rushing_yards', 0) / carries
        else:
            feature_vector[feature_map['carry_efficiency']] = 0.0
    
    if 'target_efficiency' in feature_map:
        targets = player_features.get('targets', 0)
        if targets > 0:
            feature_vector[feature_map['target_efficiency']] = player_features.get('receiving_yards', 0) / targets
        else:
            feature_vector[feature_map['target_efficiency']] = 0.0
    
    # Add position features
    if 'position_QB' in feature_map:
        feature_vector[feature_map['position_QB']] = 1 if position == 'QB' else 0
    
    if 'position_RB' in feature_map:
        feature_vector[feature_map['position_RB']] = 1 if position == 'RB' else 0
    
    if 'position_WR' in feature_map:
        feature_vector[feature_map['position_WR']] = 1 if position == 'WR' else 0
    
    if 'position_TE' in feature_map:
        feature_vector[feature_map['position_TE']] = 1 if position == 'TE' else 0
    
    if 'is_qb' in feature_map:
        feature_vector[feature_map['is_qb']] = 1 if position == 'QB' else 0
    
    if 'is_skill' in feature_map:
        feature_vector[feature_map['is_skill']] = 1 if position in ['RB', 'WR', 'TE'] else 0
    
    # Add conference feature (placeholder - you might want to load actual conference data)
    if 'conference_afc' in feature_map:
        # Simple heuristic: odd team indices are AFC, even are NFC (this is just a placeholder)
        feature_vector[feature_map['conference_afc']] = 0.5  # Neutral value
    
    # Add team features
    for feature_name in feature_names:
        if feature_name.startswith('team_'):
            team_abbr = feature_name.replace('team_', '')
            if team_abbr == team:
                feature_vector[feature_map[feature_name]] = 1
    
    # Add temporal features
    if 'week_early' in feature_map:
        feature_vector[feature_map['week_early']] = 1 if current_week <= 4 else 0
    
    if 'week_mid' in feature_map:
        feature_vector[feature_map['week_mid']] = 1 if 5 <= current_week <= 12 else 0
    
    if 'week_late' in feature_map:
        feature_vector[feature_map['week_late']] = 1 if current_week >= 13 else 0
    
    if 'week_progression' in feature_map:
        feature_vector[feature_map['week_progression']] = current_week / 18.0  # Normalize to 0-1
    
    if 'weeks_since_bye' in feature_map:
        feature_vector[feature_map['weeks_since_bye']] = 0.0  # Placeholder - you'd need to calculate this
    
    # Add optional features (only if the model expects them)
    if 'breakaway_rate' in feature_map:
        # Placeholder - you'd need to calculate this from actual data
        feature_vector[feature_map['breakaway_rate']] = 0.0
    
    if 'catch_rate' in feature_map:
        targets = player_features.get('targets', 0)
        receptions = player_features.get('receptions', 0)
        if targets > 0:
            feature_vector[feature_map['catch_rate']] = receptions / targets
        else:
            feature_vector[feature_map['catch_rate']] = 0.0
    
    if 'completion_rate' in feature_map:
        attempts = player_features.get('passing_attempts', 0)
        completions = player_features.get('passing_completions', 0)
        if attempts > 0:
            feature_vector[feature_map['completion_rate']] = completions / attempts
        else:
            feature_vector[feature_map['completion_rate']] = 0.0
    
    if 'air_yards_per_attempt' in feature_map:
        # Placeholder - you'd need actual air yards data
        feature_vector[feature_map['air_yards_per_attempt']] = 0.0
    
    # CRITICAL: Scale features to match training distribution
    if feature_stats:
        for i, feature_name in enumerate(feature_names):
            if feature_name in feature_stats.get('means', {}):
                mean = feature_stats['means'][feature_name]
                std = feature_stats['stds'][feature_name]
                if std > 0:  # Avoid division by zero
                    feature_vector[i] = (feature_vector[i] - mean) / std
        print(f"    Applied feature scaling to match training distribution")
    else:
        print(f"    Warning: No feature stats available - predictions may be incorrect")
    
    return feature_vector


def create_feature_vector_unified(player_features: Dict, position: str, team: str, salary: float, 
                                 current_week: int, feature_names: List[str], feature_stats: Dict,
                                 all_teams: List[str]) -> np.ndarray:
    """Create feature vector using unified feature engineering module."""
    
    # Calculate all derived features consistently
    features = calculate_derived_features(player_features, position, team, current_week)
    
    # Add team features
    features = add_team_features(features, team, all_teams)
    
    # Scale features to match training distribution
    features = scale_features(features, feature_stats)
    
    # Create feature vector in correct order using the module function
    feature_vector = fe_create_feature_vector(features, feature_names)
    
    print(f"    Applied unified feature engineering with scaling")
    return feature_vector


def make_predictions_unified(dk_slate: pd.DataFrame, model, feature_names: List[str], 
                            current_season: int, current_week: int, feature_stats: Dict,
                            all_teams: List[str]) -> pd.DataFrame:
    """Make predictions for all players in the slate."""
    print("Making predictions...")
    
    predictions = []
    
    for _, player in dk_slate.iterrows():
        player_name = player['Name']
        position = player['Position']
        team = player['TeamAbbrev']
        salary = player['Salary']
        player_id = player.get('player_id')
        
        print(f"  Processing {player_name} ({position}, {team})...")
        
        # Check if we have a player_id match
        if pd.isna(player_id):
            print(f"    Warning: No player_id match found for {player_name}")
            predictions.append({
                'Name': player_name,
                'Position': position,
                'TeamAbbrev': team,
                'Salary': salary,
                'predicted_points': 0.0,
                'confidence': 'low',
                'match_status': 'unmatched'
            })
            continue
        
        print(f"    Matched to player_id: {player_id}")
        
        # Get player features using player_id
        player_features = get_player_features(player_id, current_season, current_week)
        
        if not player_features:
            print(f"    Warning: No features found for player_id {player_id}")
            predictions.append({
                'Name': player_name,
                'Position': position,
                'TeamAbbrev': team,
                'Salary': salary,
                'predicted_points': 0.0,
                'confidence': 'low',
                'match_status': 'matched_no_data'
            })
            continue
        
        # Create feature vector using unified feature engineering
        feature_vector = create_feature_vector_unified(
            player_features, position, team, salary, current_week, feature_names, feature_stats, all_teams
        )
        
        # Make prediction
        try:
            predicted_points = model.predict([feature_vector])[0]
            confidence = 'high' if len(player_features) > 5 else 'medium'
        except Exception as e:
            print(f"    Error making prediction: {e}")
            predicted_points = 0.0
            confidence = 'low'
        
        predictions.append({
            'Name': player_name,
            'Position': position,
            'TeamAbbrev': team,
            'Salary': salary,
            'predicted_points': round(predicted_points, 2),
            'confidence': confidence,
            'recent_games': len(player_features) if player_features else 0,
            'match_status': 'matched_with_data'
        })
    
    return pd.DataFrame(predictions)


def save_predictions(predictions: pd.DataFrame, output_path: str):
    """Save predictions to CSV."""
    print(f"Saving predictions to: {output_path}")
    
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Sort by predicted points (descending)
    predictions_sorted = predictions.sort_values('predicted_points', ascending=False)
    
    # Save to CSV
    predictions_sorted.to_csv(output_path, index=False)
    
    print(f"  Saved {len(predictions)} predictions")
    
    # Show top predictions
    print(f"\nTop 10 predictions:")
    for _, row in predictions_sorted.head(10).iterrows():
        print(f"  {row['Name']} ({row['Position']}): {row['predicted_points']} pts (${row['Salary']:,})")


def main():
    parser = argparse.ArgumentParser(description='Make predictions on new DraftKings slate')
    parser.add_argument('--model-dir', required=True, help='Directory containing trained model')
    parser.add_argument('--dk-slate', required=True, help='Path to DraftKings slate CSV')
    parser.add_argument('--aliases', default='data/xwalk/aliases.csv', help='Path to aliases file')
    parser.add_argument('--current-season', type=int, default=2024, help='Current NFL season')
    parser.add_argument('--current-week', type=int, default=1, help='Current NFL week')
    parser.add_argument('--output', default='predictions.csv', help='Output file for predictions')
    
    args = parser.parse_args()
    
    try:
        print("=" * 60)
        print("NFL DFS INFERENCE WORKFLOW")
        print("=" * 60)
        
        # Load model and metadata
        model_path = Path(args.model_dir) / "baseline_model_lightgbm.pkl"
        model = load_trained_model(str(model_path))
        metadata = load_model_metadata(args.model_dir)
        
        if 'feature_names' not in metadata:
            raise ValueError("Feature names not found in model metadata")
        
        # Load DraftKings slate
        dk_slate = load_dk_slate(args.dk_slate)
        
        # Load aliases
        aliases = load_aliases(args.aliases)
        
        # Load roster data and create lookup
        roster_data, roster_lookup = load_roster_data(args.current_season)
        
        if roster_data.empty or not roster_lookup:
            raise ValueError("Failed to load roster data")
        
        # Match players to roster using master sheet logic
        dk_slate = match_players_to_roster(dk_slate, aliases, roster_lookup)
        
        # Load feature stats for normalization
        feature_stats = load_feature_stats(args.model_dir)
        
        # Get team list for consistent team encoding
        training_data_path = "data/processed/train_table_all.csv"
        all_teams = get_all_teams_from_training_data(training_data_path)
        print(f"Loaded {len(all_teams)} teams for consistent encoding")
        
        # Make predictions using unified feature engineering
        predictions = make_predictions_unified(
            dk_slate, model, metadata['feature_names'], 
            args.current_season, args.current_week, feature_stats, all_teams
        )
        
        # Save predictions
        save_predictions(predictions, args.output)
        
        print(f"\n✅ Inference workflow completed successfully!")
        print(f"Predictions saved to: {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
