#!/usr/bin/env python3
"""
Build DK points labels from NFL API data.
Loads weekly data and schedules, computes DK points if needed, and outputs per-season CSV files.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def load_nfl_data(seasons):
    """Load NFL weekly data and schedules for specified seasons."""
    import nfl_data_py as nfl
    
    print("Loading NFL weekly data...")
    weekly = nfl.import_weekly_data(seasons)
    print(f"Weekly data: {len(weekly)} rows")
    
    print("Loading NFL schedules...")
    sched = nfl.import_schedules(seasons)
    print(f"Schedule data: {len(sched)} rows")
    
    return weekly, sched


def build_team_date_map(schedules):
    """Build mapping from (season, week, team) to game_date."""
    print("Building team date mapping from schedules...")
    
    team_date_map = {}
    
    for _, row in schedules.iterrows():
        season = row['season']
        week = row['week']
        home_team = row['home_team']
        away_team = row['away_team']
        
        # Convert gameday to YYYYMMDD format
        if pd.notna(row['gameday']):
            game_date = pd.to_datetime(row['gameday']).strftime('%Y%m%d')
            
            # Map both home and away teams to this game date
            team_date_map[(season, week, home_team)] = game_date
            team_date_map[(season, week, away_team)] = game_date
    
    print(f"Team date mapping: {len(team_date_map)} entries")
    return team_date_map


def compute_dk_points(row):
    """Compute DK points from individual stats if fantasy_points_dk is not available."""
    points = 0.0
    
    # Passing stats
    if pd.notna(row.get('passing_yards', 0)):
        pass_yds = float(row['passing_yards'])
        points += pass_yds * 0.04  # 4 points per 100 yards
        if pass_yds >= 300:
            points += 3  # 300+ yard bonus
    
    if pd.notna(row.get('passing_tds', 0)):
        points += float(row['passing_tds']) * 4  # 4 points per TD
    
    if pd.notna(row.get('interceptions', 0)):
        points -= float(row['interceptions']) * 1  # -1 point per INT
    
    # Rushing stats
    if pd.notna(row.get('rushing_yards', 0)):
        rush_yds = float(row['rushing_yards'])
        points += rush_yds * 0.1  # 10 points per 100 yards
        if rush_yds >= 100:
            points += 3  # 100+ yard bonus
    
    if pd.notna(row.get('rushing_tds', 0)):
        points += float(row['rushing_tds']) * 6  # 6 points per TD
    
    # Receiving stats
    if pd.notna(row.get('receptions', 0)):
        points += float(row['receptions']) * 1  # 1 point per reception
    
    if pd.notna(row.get('receiving_yards', 0)):
        rec_yds = float(row['receiving_yards'])
        points += rec_yds * 0.1  # 10 points per 100 yards
        if rec_yds >= 100:
            points += 3  # 100+ yard bonus
    
    if pd.notna(row.get('receiving_tds', 0)):
        points += float(row['receiving_tds']) * 6  # 6 points per TD
    
    # Two-point conversions (sum all types)
    two_pt_total = 0
    for col in ['passing_2pt_conversions', 'rushing_2pt_conversions', 'receiving_2pt_conversions']:
        if pd.notna(row.get(col, 0)):
            two_pt_total += float(row[col])
    points += two_pt_total * 2  # 2 points per conversion
    
    # Fumbles lost (sum all types)
    fumbles_lost_total = 0
    for col in ['sack_fumbles_lost', 'rushing_fumbles_lost', 'receiving_fumbles_lost']:
        if pd.notna(row.get(col, 0)):
            fumbles_lost_total += float(row[col])
    points -= fumbles_lost_total * 1  # -1 point per fumble lost
    
    return round(points, 2)


def prepare_weekly_data(weekly, team_date_map):
    """Prepare weekly data for label generation."""
    print("Preparing weekly data...")
    
    # Determine team column to use
    if 'recent_team' in weekly.columns:
        team_col = 'recent_team'
        print("Using 'recent_team' column for team identification")
    elif 'team' in weekly.columns:
        team_col = 'recent_team'  # Will rename 'team' to 'recent_team'
        print("Using 'team' column for team identification")
    else:
        print("Warning: No team column found in weekly data")
        return pd.DataFrame()
    
    # Rename 'team' to 'recent_team' if needed for consistency
    if 'team' in weekly.columns and team_col == 'recent_team':
        weekly = weekly.rename(columns={'team': 'recent_team'})
    
    # Keep only rows with player_id
    valid_rows = weekly[weekly['player_id'].notna()].copy()
    print(f"Rows with player_id: {len(valid_rows)}")
    
    # Add game_date using team_date_map
    valid_rows['game_date'] = valid_rows.apply(
        lambda row: team_date_map.get((row['season'], row['week'], row[team_col]), None), 
        axis=1
    )
    
    # Drop rows where we couldn't determine game_date
    dated_rows = valid_rows[valid_rows['game_date'].notna()].copy()
    dropped_count = len(valid_rows) - len(dated_rows)
    print(f"Dropped {dropped_count} rows with no game_date")
    
    return dated_rows


def generate_labels(weekly_data, seasons, out_dir):
    """Generate labels CSV for each season."""
    print(f"\nGenerating labels for seasons: {seasons}")
    
    # Ensure output directory exists
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    for season in seasons:
        print(f"\nProcessing season {season}...")
        
        # Filter to season
        season_data = weekly_data[weekly_data['season'] == season].copy()
        
        if len(season_data) == 0:
            print(f"No data for season {season}")
            continue
        
        # Check if fantasy_points_dk exists
        if 'fantasy_points_dk' in season_data.columns:
            print("Using existing fantasy_points_dk column")
            season_data['dk_points'] = season_data['fantasy_points_dk']
        else:
            print("Computing DK points from individual stats")
            season_data['dk_points'] = season_data.apply(compute_dk_points, axis=1)
        
        # Select required columns
        labels_df = season_data[['game_date', 'player_id', 'dk_points']].copy()
        
        # Drop rows with null values
        initial_count = len(labels_df)
        labels_df = labels_df.dropna()
        final_count = len(labels_df)
        dropped_null = initial_count - final_count
        
        print(f"Season {season}: {initial_count} initial rows, {dropped_null} dropped (nulls), {final_count} written")
        
        # Write to CSV
        output_file = Path(out_dir) / f"labels_{season}.csv"
        labels_df.to_csv(output_file, index=False)
        print(f"Labels written to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Build DK points labels from NFL API data')
    parser.add_argument('--seasons', nargs='+', type=int, required=True, 
                       help='Seasons to process (e.g., 2024 2025)')
    parser.add_argument('--out-dir', default='data/raw/labels/', 
                       help='Output directory for label CSV files')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("BUILDING DK POINTS LABELS FROM NFL API")
    print("=" * 60)
    
    try:
        # Load NFL data
        weekly, sched = load_nfl_data(args.seasons)
        
        # Build team date mapping
        team_date_map = build_team_date_map(sched)
        
        # Prepare weekly data
        prepared_data = prepare_weekly_data(weekly, team_date_map)
        
        if len(prepared_data) == 0:
            print("No valid data to process")
            return
        
        # Generate labels for each season
        generate_labels(prepared_data, args.seasons, args.out_dir)
        
        print(f"\nLabel generation complete! Output directory: {args.out_dir}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
