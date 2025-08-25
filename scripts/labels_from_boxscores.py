#!/usr/bin/env python3
"""
Build DK points labels from boxscore data.
Loads boxscore CSVs, maps column names, computes DK points, and outputs labels.
"""

import argparse
import pandas as pd
import os
import glob
import yaml
from pathlib import Path


def load_boxscores(boxscores_pattern):
    """Load and stack all boxscore CSV files"""
    print(f"Loading boxscores from pattern: {boxscores_pattern}")
    
    # Find all matching files
    boxscore_files = glob.glob(boxscores_pattern)
    
    if not boxscore_files:
        raise FileNotFoundError(f"No files found matching pattern: {boxscores_pattern}")
    
    print(f"Found {len(boxscore_files)} boxscore files:")
    for file in boxscore_files:
        print(f"  {file}")
    
    # Read and stack all boxscores
    all_boxscores = []
    total_rows = 0
    
    for file_path in boxscore_files:
        print(f"Reading {file_path}...")
        df = pd.read_csv(file_path)
        print(f"  Loaded {len(df)} rows")
        all_boxscores.append(df)
        total_rows += len(df)
    
    # Combine all boxscores
    if len(all_boxscores) == 1:
        combined_df = all_boxscores[0]
    else:
        print(f"Stacking {len(all_boxscores)} boxscore files...")
        combined_df = pd.concat(all_boxscores, ignore_index=True)
    
    print(f"Combined boxscores: {len(combined_df)} total rows")
    return combined_df


def load_column_mapping(map_file):
    """Load column name mapping from YAML file"""
    print(f"Loading column mapping from: {map_file}")
    
    if not os.path.exists(map_file):
        raise FileNotFoundError(f"Column mapping file not found: {map_file}")
    
    with open(map_file, 'r') as f:
        mapping = yaml.safe_load(f)
    
    print(f"Loaded {len(mapping)} column mappings")
    return mapping


def map_columns(df, column_mapping):
    """Map column names using the provided mapping"""
    print("Mapping column names...")
    
    # Create a copy to avoid modifying original
    mapped_df = df.copy()
    
    # Apply column mappings
    for old_name, new_name in column_mapping.items():
        if old_name in mapped_df.columns:
            mapped_df[new_name] = mapped_df[old_name]
            print(f"  Mapped: {old_name} -> {new_name}")
        else:
            print(f"  Warning: Column '{old_name}' not found in boxscore data")
    
    return mapped_df


def validate_required_columns(df):
    """Validate that all required columns exist"""
    required_columns = [
        'game_date', 'player_id', 'pass_yds', 'pass_td', 'pass_int',
        'rush_yds', 'rush_td', 'rec_rec', 'rec_yds', 'rec_td',
        'two_pt', 'fumbles_lost'
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    print(f"✅ All required columns present")
    return True


def compute_dk_points(row):
    """Compute DK points using official scoring rules"""
    points = 0.0
    
    # Passing
    if pd.notna(row.get('pass_yds', 0)):
        pass_yds = float(row['pass_yds'])
        points += pass_yds * 0.04  # 4 points per 100 yards
        if pass_yds >= 300:
            points += 3  # 300+ yard bonus
    
    if pd.notna(row.get('pass_td', 0)):
        points += float(row['pass_td']) * 4  # 4 points per TD
    
    if pd.notna(row.get('pass_int', 0)):
        points -= float(row['pass_int']) * 1  # -1 point per INT
    
    # Rushing
    if pd.notna(row.get('rush_yds', 0)):
        rush_yds = float(row['rush_yds'])
        points += rush_yds * 0.1  # 10 points per 100 yards
        if rush_yds >= 100:
            points += 3  # 100+ yard bonus
    
    if pd.notna(row.get('rush_td', 0)):
        points += float(row['rush_td']) * 6  # 6 points per TD
    
    # Receiving
    if pd.notna(row.get('rec_rec', 0)):
        points += float(row['rec_rec']) * 1  # 1 point per reception
    
    if pd.notna(row.get('rec_yds', 0)):
        rec_yds = float(row['rec_yds'])
        points += rec_yds * 0.1  # 10 points per 100 yards
        if rec_yds >= 100:
            points += 3  # 100+ yard bonus
    
    if pd.notna(row.get('rec_td', 0)):
        points += float(row['rec_td']) * 6  # 6 points per TD
    
    # Two-point conversions
    if pd.notna(row.get('two_pt', 0)):
        points += float(row['two_pt']) * 2  # 2 points per 2PT
    
    # Fumbles lost
    if pd.notna(row.get('fumbles_lost', 0)):
        points -= float(row['fumbles_lost']) * 1  # -1 point per fumble lost
    
    return round(points, 1)


def process_boxscores(df):
    """Process boxscores and compute DK points"""
    print("Processing boxscores and computing DK points...")
    
    # Check for missing player_id or game_date
    missing_player_id = df['player_id'].isna().sum()
    missing_game_date = df['game_date'].isna().sum()
    
    if missing_player_id > 0:
        print(f"⚠️  Warning: {missing_player_id} rows missing player_id")
    
    if missing_game_date > 0:
        print(f"⚠️  Warning: {missing_game_date} rows missing game_date")
    
    # Filter out rows with missing required fields
    original_count = len(df)
    df_clean = df.dropna(subset=['player_id', 'game_date'])
    filtered_count = len(df_clean)
    
    if filtered_count < original_count:
        print(f"Filtered out {original_count - filtered_count} rows with missing player_id or game_date")
    
    # Compute DK points
    print("Computing DK points for each player...")
    df_clean['dk_points'] = df_clean.apply(compute_dk_points, axis=1)
    
    # Select only the required output columns
    output_df = df_clean[['game_date', 'player_id', 'dk_points']].copy()
    
    # Ensure game_date is in YYYYMMDD format
    # Convert to string and ensure it's 8 digits
    output_df['game_date'] = output_df['game_date'].astype(str).str.zfill(8)
    
    print(f"✅ Processed {len(output_df)} rows with DK points")
    return output_df


def main():
    parser = argparse.ArgumentParser(description='Build DK points labels from boxscore data')
    parser.add_argument('--boxscores', required=True, 
                       help='Glob pattern for boxscore CSV files')
    parser.add_argument('--map', required=True,
                       help='YAML file mapping column names')
    parser.add_argument('--out', required=True,
                       help='Output path for labels CSV')
    
    args = parser.parse_args()
    
    try:
        # Load boxscores
        boxscores_df = load_boxscores(args.boxscores)
        
        # Load column mapping
        column_mapping = load_column_mapping(args.map)
        
        # Map column names
        mapped_df = map_columns(boxscores_df, column_mapping)
        
        # Validate required columns
        validate_required_columns(mapped_df)
        
        # Process and compute DK points
        labels_df = process_boxscores(mapped_df)
        
        # Ensure output directory exists
        output_dir = os.path.dirname(args.out)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Write output
        print(f"Writing labels to: {args.out}")
        labels_df.to_csv(args.out, index=False)
        
        # Print summary
        print(f"\n=== LABELS BUILT SUCCESSFULLY ===")
        print(f"Output file: {args.out}")
        print(f"Total rows written: {len(labels_df):,}")
        print(f"Date range: {labels_df['game_date'].min()} to {labels_df['game_date'].max()}")
        print(f"Unique players: {labels_df['player_id'].nunique():,}")
        print(f"DK points range: {labels_df['dk_points'].min():.1f} to {labels_df['dk_points'].max():.1f}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
