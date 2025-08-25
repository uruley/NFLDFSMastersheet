#!/usr/bin/env python3
"""
Build training tables for machine learning with features and targets.
Creates season-specific training tables and a combined dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import glob


def load_api_training_data(api_file):
    """Load the API training data with fantasy points."""
    print(f"Loading API training data from: {api_file}")
    
    df = pd.read_csv(api_file)
    print(f"  Loaded {len(df)} rows")
    print(f"  Columns: {list(df.columns)}")
    
    # Verify required columns exist
    required_cols = ['season', 'week', 'player_id', 'player_name', 'recent_team', 'position', 'fantasy_points_dk']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    return df


def add_salary_features(data):
    """Add salary features (placeholder values for now)."""
    print("Adding salary features...")
    
    # Create placeholder salary based on position and fantasy points
    # This is a simplified approach - in production you'd load actual DK salaries
    data['salary'] = 0  # Placeholder
    
    # Add salary tiers based on position and performance
    data['salary_tier'] = 'mid'
    
    # High performers get higher tier
    high_performers = data['fantasy_points_dk'] > data['fantasy_points_dk'].quantile(0.8)
    data.loc[high_performers, 'salary_tier'] = 'high'
    
    # Low performers get lower tier
    low_performers = data['fantasy_points_dk'] < data['fantasy_points_dk'].quantile(0.2)
    data.loc[low_performers, 'salary_tier'] = 'low'
    
    print(f"  Added salary and salary_tier columns")
    
    return data


def add_usage_features(data):
    """Add usage features from available data."""
    print("Adding usage features...")
    
    # Initialize usage columns
    data['snaps'] = 0
    data['targets'] = 0
    data['carries'] = 0
    
    # If we have these columns in the API data, use them
    if 'targets' in data.columns:
        data['targets'] = data['targets'].fillna(0)
        print(f"  Using existing targets data")
    
    if 'carries' in data.columns:
        data['carries'] = data['carries'].fillna(0)
        print(f"  Using existing carries data")
    
    # Create derived usage features
    data['total_touches'] = data['receptions'] + data['carries']
    data['total_tds'] = data['passing_tds'] + data['rushing_tds'] + data['receiving_tds']
    data['touchdown_rate'] = data['total_tds'] / data['total_touches'].replace(0, 1)
    data['yards_per_touch'] = (data['rushing_yards'] + data['receiving_yards']) / data['total_touches'].replace(0, 1)
    
    # Usage efficiency
    data['target_efficiency'] = data['receptions'] / data['targets'].replace(0, 1)
    data['carry_efficiency'] = data['rushing_yards'] / data['carries'].replace(0, 1)
    
    print(f"  Added usage features: total_touches, touchdown_rate, yards_per_touch, target_efficiency, carry_efficiency")
    
    return data


def add_position_features(data):
    """Add position-specific features."""
    print("Adding position features...")
    
    # Position dummies
    positions = ['QB', 'RB', 'WR', 'TE']
    for pos in positions:
        data[f'position_{pos}'] = (data['position'] == pos).astype(int)
    
    # Position-specific stats
    data['is_qb'] = (data['position'] == 'QB').astype(int)
    data['is_skill'] = (data['position'].isin(['RB', 'WR', 'TE'])).astype(int)
    
    # QB-specific features
    qb_mask = data['position'] == 'QB'
    data.loc[qb_mask, 'completion_rate'] = 0.65  # Placeholder
    data.loc[qb_mask, 'air_yards_per_attempt'] = 8.0  # Placeholder
    
    # RB-specific features
    rb_mask = data['position'] == 'RB'
    data.loc[rb_mask, 'breakaway_rate'] = 0.15  # Placeholder
    
    # WR/TE-specific features
    wr_te_mask = data['position'].isin(['WR', 'TE'])
    data.loc[wr_te_mask, 'catch_rate'] = 0.70  # Placeholder
    
    print(f"  Added position features: position dummies, is_qb, is_skill, position-specific stats")
    
    return data


def add_team_features(data):
    """Add team-related features."""
    print("Adding team features...")
    
    # Team dummies (one-hot encoding)
    teams = sorted(data['recent_team'].unique())
    for team in teams:
        data[f'team_{team}'] = (data['recent_team'] == team).astype(int)
    
    # Conference and division (simplified)
    afc_teams = ['BAL', 'CIN', 'CLE', 'PIT', 'BUF', 'MIA', 'NE', 'NYJ', 'HOU', 'IND', 'JAX', 'TEN', 'DEN', 'KC', 'LV', 'LAC']
    data['conference_afc'] = data['recent_team'].isin(afc_teams).astype(int)
    
    print(f"  Added team features: team dummies, conference")
    
    return data


def add_temporal_features(data):
    """Add time-based features."""
    print("Adding temporal features...")
    
    # Week features
    data['week_early'] = (data['week'] <= 4).astype(int)
    data['week_mid'] = ((data['week'] > 4) & (data['week'] <= 12)).astype(int)
    data['week_late'] = (data['week'] > 12).astype(int)
    
    # Season progression
    data['week_progression'] = data['week'] / 18.0
    
    # Bye week effect (simplified)
    data['weeks_since_bye'] = 0  # Placeholder
    
    print(f"  Added temporal features: week_early, week_mid, week_late, week_progression")
    
    return data


def create_training_table(data, season):
    """Create a training table for a specific season."""
    print(f"Creating training table for season {season}...")
    
    # Filter to specific season
    season_data = data[data['season'] == season].copy()
    print(f"  Season {season}: {len(season_data)} rows")
    
    # Add all features
    season_data = add_salary_features(season_data)
    season_data = add_usage_features(season_data)
    season_data = add_position_features(season_data)
    season_data = add_team_features(season_data)
    season_data = add_temporal_features(season_data)
    
    # Select final feature columns
    feature_columns = [
        # Core identifiers
        'season', 'week', 'player_id', 'player_name', 'recent_team', 'position',
        
        # Basic stats
        'passing_yards', 'passing_tds', 'interceptions',
        'rushing_yards', 'rushing_tds',
        'receptions', 'receiving_yards', 'receiving_tds',
        'fumbles_lost',
        
        # Salary features
        'salary', 'salary_tier',
        
        # Usage features
        'snaps', 'targets', 'carries', 'total_touches', 'touchdown_rate', 'yards_per_touch',
        'target_efficiency', 'carry_efficiency',
        
        # Position features
        'position_QB', 'position_RB', 'position_WR', 'position_TE',
        'is_qb', 'is_skill',
        'completion_rate', 'air_yards_per_attempt', 'breakaway_rate', 'catch_rate',
        
        # Team features
        'conference_afc',
        
        # Temporal features
        'week_early', 'week_mid', 'week_late', 'week_progression', 'weeks_since_bye'
    ]
    
    # Add team dummies
    teams = sorted(season_data['recent_team'].unique())
    for team in teams:
        feature_columns.append(f'team_{team}')
    
    # Filter to available columns
    available_features = [col for col in feature_columns if col in season_data.columns]
    missing_features = [col for col in feature_columns if col not in season_data.columns]
    
    if missing_features:
        print(f"  Warning: Missing features: {missing_features}")
    
    # Create final training table
    train_table = season_data[available_features + ['fantasy_points_dk']].copy()
    
    # Fill any remaining NaN values
    train_table = train_table.fillna(0)
    
    print(f"  Final training table: {len(train_table)} rows, {len(train_table.columns)} columns")
    print(f"  Features: {len(available_features)}")
    print(f"  Target: fantasy_points_dk")
    
    return train_table


def save_training_tables(data, output_dir):
    """Save training tables for each season and combined dataset."""
    print(f"Saving training tables to: {output_dir}")
    
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    all_tables = []
    
    # Process each season
    for season in sorted(data['season'].unique()):
        print(f"\nProcessing season {season}...")
        
        # Create training table for this season
        train_table = create_training_table(data, season)
        
        # Save season-specific file
        season_file = Path(output_dir) / f"history/{season}/train_table_{season}.csv"
        season_file.parent.mkdir(parents=True, exist_ok=True)
        train_table.to_csv(season_file, index=False)
        print(f"  Saved: {season_file}")
        
        all_tables.append(train_table)
    
    # Create combined dataset
    print(f"\nCreating combined dataset...")
    combined_table = pd.concat(all_tables, ignore_index=True)
    
    combined_file = Path(output_dir) / "train_table_all.csv"
    combined_table.to_csv(combined_file, index=False)
    print(f"  Saved combined dataset: {combined_file}")
    print(f"  Combined: {len(combined_table)} rows, {len(combined_table.columns)} columns")
    
    return combined_table


def main():
    """Main function to build training tables."""
    parser = argparse.ArgumentParser(
        description="Build training tables with features and targets"
    )
    parser.add_argument(
        "--api-data", 
        required=True,
        help="Path to API training data CSV file"
    )
    parser.add_argument(
        "--output-dir", 
        default="data/processed",
        help="Output directory for training tables (default: data/processed)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("BUILDING TRAINING TABLES")
    print("=" * 60)
    
    try:
        # Load API data
        api_data = load_api_training_data(args.api_data)
        
        # Save training tables
        combined_table = save_training_tables(api_data, args.output_dir)
        
        print(f"\n✅ Successfully created training tables!")
        print(f"Output directory: {args.output_dir}")
        print(f"Combined dataset: {len(combined_table):,} rows")
        
    except Exception as e:
        print(f"\n❌ Error building training tables: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
