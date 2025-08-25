#!/usr/bin/env python3
"""
Build training data directly from NFL API weekly data.
Extracts relevant columns for fantasy sports analysis and machine learning.
"""

import nfl_data_py as nfl
import pandas as pd
from pathlib import Path
import argparse


def load_weekly_data(seasons):
    """Load weekly NFL data for specified seasons."""
    print(f"Loading weekly data for seasons: {seasons}")
    
    all_weekly_data = []
    
    for season in seasons:
        print(f"  Processing season {season}...")
        try:
            weekly = nfl.import_weekly_data([season])
            print(f"    Loaded {len(weekly)} records")
            
            # Add season column if not present
            if 'season' not in weekly.columns:
                weekly['season'] = season
            
            all_weekly_data.append(weekly)
            
        except Exception as e:
            print(f"    Error loading season {season}: {e}")
            continue
    
    if not all_weekly_data:
        raise ValueError("No weekly data could be loaded for any season")
    
    # Combine all seasons
    combined_data = pd.concat(all_weekly_data, ignore_index=True)
    print(f"Combined data: {len(combined_data)} total records")
    
    return combined_data


def extract_relevant_columns(weekly_data):
    """Extract and clean relevant columns for fantasy sports analysis."""
    print("Extracting relevant columns...")
    
    # Define the columns we want to keep
    relevant_columns = [
        'season', 'week', 'player_id', 'player_name', 'recent_team', 'position',
        'passing_yards', 'passing_tds', 'interceptions',
        'rushing_yards', 'rushing_tds',
        'receptions', 'receiving_yards', 'receiving_tds',
        'fumbles_lost'
    ]
    
    # Check which columns are available in the data
    available_columns = [col for col in relevant_columns if col in weekly_data.columns]
    missing_columns = [col for col in relevant_columns if col not in weekly_data.columns]
    
    if missing_columns:
        print(f"  Warning: Missing columns: {missing_columns}")
        print(f"  Available columns: {list(weekly_data.columns)}")
    
    # Extract available columns
    extracted_data = weekly_data[available_columns].copy()
    
    # Fill missing columns with 0
    for col in missing_columns:
        extracted_data[col] = 0
        print(f"  Added missing column '{col}' with default value 0")
    
    # Ensure all columns are in the correct order
    extracted_data = extracted_data[relevant_columns]
    
    print(f"  Final columns: {list(extracted_data.columns)}")
    print(f"  Final shape: {extracted_data.shape}")
    
    return extracted_data


def clean_and_validate_data(data):
    """Clean and validate the extracted data."""
    print("Cleaning and validating data...")
    
    initial_rows = len(data)
    
    # Remove rows with missing player_id
    data = data[data['player_id'].notna()].copy()
    print(f"  Rows with valid player_id: {len(data)} (dropped {initial_rows - len(data)})")
    
    # Remove rows with missing player_name
    data = data[data['player_name'].notna()].copy()
    print(f"  Rows with valid player_name: {len(data)}")
    
    # Remove rows with missing position
    data = data[data['position'].notna()].copy()
    print(f"  Rows with valid position: {len(data)}")
    
    # Convert numeric columns to appropriate types
    numeric_columns = [
        'passing_yards', 'passing_tds', 'interceptions',
        'rushing_yards', 'rushing_tds',
        'receptions', 'receiving_yards', 'receiving_tds',
        'fumbles_lost'
    ]
    
    for col in numeric_columns:
        data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)
    
    # Ensure season and week are integers
    data['season'] = pd.to_numeric(data['season'], errors='coerce').astype('Int64')
    data['week'] = pd.to_numeric(data['week'], errors='coerce').astype('Int64')
    
    # Remove rows with invalid season/week
    data = data[data['season'].notna() & data['week'].notna()].copy()
    print(f"  Rows with valid season/week: {len(data)}")
    
    # Remove rows with invalid week numbers (should be 1-18 for regular season)
    data = data[(data['week'] >= 1) & (data['week'] <= 18)].copy()
    print(f"  Rows with valid week range (1-18): {len(data)}")
    
    print(f"  Final clean data: {len(data)} rows")
    
    return data


def compute_dk_fantasy_points(data):
    """Compute DraftKings fantasy points using official scoring rules."""
    print("Computing DraftKings fantasy points...")
    
    # Initialize fantasy points column
    data['fantasy_points_dk'] = 0.0
    
    # Passing stats
    data['fantasy_points_dk'] += data['passing_yards'] / 25.0  # 1 pt per 25 yards
    data['fantasy_points_dk'] += data['passing_tds'] * 4       # 4 pts per TD
    data['fantasy_points_dk'] -= data['interceptions'] * 1     # -1 pt per INT
    
    # Rushing stats
    data['fantasy_points_dk'] += data['rushing_yards'] / 10.0  # 1 pt per 10 yards
    data['fantasy_points_dk'] += data['rushing_tds'] * 6       # 6 pts per TD
    
    # Receiving stats
    data['fantasy_points_dk'] += data['receptions'] * 1        # 1 pt per reception (PPR)
    data['fantasy_points_dk'] += data['receiving_yards'] / 10.0 # 1 pt per 10 yards
    data['fantasy_points_dk'] += data['receiving_tds'] * 6     # 6 pts per TD
    
    # Fumbles lost
    data['fantasy_points_dk'] -= data['fumbles_lost'] * 1      # -1 pt per fumble lost
    
    # Bonuses
    # +3 pts for 300+ passing yards
    data.loc[data['passing_yards'] >= 300, 'fantasy_points_dk'] += 3
    
    # +3 pts for 100+ rushing yards
    data.loc[data['rushing_yards'] >= 100, 'fantasy_points_dk'] += 3
    
    # +3 pts for 100+ receiving yards
    data.loc[data['receiving_yards'] >= 100, 'fantasy_points_dk'] += 3
    
    # Round to 2 decimal places
    data['fantasy_points_dk'] = data['fantasy_points_dk'].round(2)
    
    print(f"  Added fantasy_points_dk column")
    print(f"  Fantasy points range: {data['fantasy_points_dk'].min():.2f} to {data['fantasy_points_dk'].max():.2f}")
    
    return data


def add_derived_features(data):
    """Add derived features that might be useful for analysis."""
    print("Adding derived features...")
    
    # Total yards
    data['total_yards'] = (
        data['passing_yards'] + 
        data['rushing_yards'] + 
        data['receiving_yards']
    )
    
    # Total touchdowns
    data['total_tds'] = (
        data['passing_tds'] + 
        data['rushing_tds'] + 
        data['receiving_tds']
    )
    
    # Touchdown rate (TDs per game)
    data['td_rate'] = data['total_tds'].astype(float)
    
    # Yards per touch (for non-QBs)
    data['yards_per_touch'] = 0.0
    non_qb_mask = data['position'] != 'QB'
    total_touches = data.loc[non_qb_mask, 'receptions'] + data.loc[non_qb_mask, 'rushing_yards'].apply(lambda x: 1 if x > 0 else 0)
    data.loc[non_qb_mask, 'yards_per_touch'] = (
        data.loc[non_qb_mask, 'rushing_yards'] + 
        data.loc[non_qb_mask, 'receiving_yards']
    ) / total_touches.replace(0, 1)  # Avoid division by zero
    
    print(f"  Added features: total_yards, total_tds, td_rate, yards_per_touch")
    
    return data


def save_training_data(data, output_file):
    """Save the processed training data to CSV."""
    print(f"Saving training data to: {output_file}")
    
    # Ensure output directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    data.to_csv(output_file, index=False)
    
    print(f"  Successfully saved {len(data)} rows")
    
    # Show summary statistics
    print(f"\n=== TRAINING DATA SUMMARY ===")
    print(f"Total rows: {len(data):,}")
    print(f"Seasons: {sorted(data['season'].unique())}")
    print(f"Weeks: {sorted(data['week'].unique())}")
    print(f"Positions: {sorted(data['position'].unique())}")
    print(f"Teams: {len(data['recent_team'].unique())}")
    
    # Show sample of data
    print(f"\nSample data:")
    sample = data.head(3)
    for _, row in sample.iterrows():
        print(f"  {row['player_name']} ({row['position']}, {row['recent_team']}) - "
              f"Pass: {row['passing_yards']}yds/{row['passing_tds']}TD, "
              f"Rush: {row['rushing_yards']}yds/{row['rushing_tds']}TD, "
              f"Rec: {row['receptions']}rec/{row['receiving_yards']}yds/{row['receiving_tds']}TD - "
              f"DK Points: {row['fantasy_points_dk']}")


def main():
    """Main function to build training data from NFL API."""
    parser = argparse.ArgumentParser(
        description="Build training data directly from NFL API weekly data"
    )
    parser.add_argument(
        "--seasons", 
        nargs="+", 
        type=int, 
        default=[2020, 2021, 2022, 2023, 2024],
        help="Seasons to process (default: 2020-2024)"
    )
    parser.add_argument(
        "--output", 
        default="data/processed/training_from_api.csv",
        help="Output file path (default: data/processed/training_from_api.csv)"
    )
    parser.add_argument(
        "--add-features", 
        action="store_true",
        help="Add derived features (total_yards, total_tds, etc.)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("BUILDING TRAINING DATA FROM NFL API")
    print("=" * 60)
    
    try:
        # Load weekly data
        weekly_data = load_weekly_data(args.seasons)
        
        # Extract relevant columns
        extracted_data = extract_relevant_columns(weekly_data)
        
        # Clean and validate data
        clean_data = clean_and_validate_data(extracted_data)
        
        # Compute DraftKings fantasy points (always included)
        clean_data = compute_dk_fantasy_points(clean_data)
        
        # Add derived features if requested
        if args.add_features:
            clean_data = add_derived_features(clean_data)
        
        # Save training data
        save_training_data(clean_data, args.output)
        
        print(f"\n✅ Successfully built training data with {len(clean_data):,} rows!")
        print(f"Output file: {args.output}")
        
    except Exception as e:
        print(f"\n❌ Error building training data: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
