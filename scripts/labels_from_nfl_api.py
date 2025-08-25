#!/usr/bin/env python3
"""
Generate real DraftKings fantasy points from NFL API weekly data.
Creates authentic historical labels for machine learning training.
"""

import nfl_data_py as nfl
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta


def compute_dk_points(row):
    """Compute DraftKings fantasy points using official scoring rules."""
    pts = 0.0
    
    # Passing stats
    pass_yds = row.get("passing_yards", 0)
    pass_tds = row.get("passing_tds", 0)
    interceptions = row.get("interceptions", 0)
    
    pts += pass_yds / 25.0  # 1 pt per 25 yards
    pts += pass_tds * 4     # 4 pts per TD
    pts -= interceptions * 1 # -1 pt per INT
    
    # Rushing stats
    rush_yds = row.get("rushing_yards", 0)
    rush_tds = row.get("rushing_tds", 0)
    
    pts += rush_yds / 10.0  # 1 pt per 10 yards
    pts += rush_tds * 6     # 6 pts per TD
    
    # Receiving stats
    receptions = row.get("receptions", 0)
    rec_yds = row.get("receiving_yards", 0)
    rec_tds = row.get("receiving_tds", 0)
    
    pts += receptions * 1   # 1 pt per reception (PPR)
    pts += rec_yds / 10.0   # 1 pt per 10 yards
    pts += rec_tds * 6      # 6 pts per TD
    
    # Fumbles lost
    fumbles_lost = row.get("fumbles_lost", 0)
    pts -= fumbles_lost * 1 # -1 pt per fumble lost
    
    # Bonuses
    if pass_yds >= 300:
        pts += 3  # +3 pts for 300+ passing yards
    if rush_yds >= 100:
        pts += 3  # +3 pts for 100+ rushing yards
    if rec_yds >= 100:
        pts += 3  # +3 pts for 100+ receiving yards
    
    return round(pts, 2)


def get_week_start_date(season, week):
    """Get the start date (Sunday) for a given season and week."""
    # NFL season typically starts first Thursday in September
    # Week 1 usually starts around September 5-10
    if week == 1:
        # First week of season - approximate start date
        start_date = datetime(season, 9, 5)  # September 5th
        # Find the first Sunday
        while start_date.weekday() != 6:  # 6 = Sunday
            start_date += timedelta(days=1)
    else:
        # Subsequent weeks - add 7 days per week
        week1_start = get_week_start_date(season, 1)
        start_date = week1_start + timedelta(weeks=week-1)
    
    return start_date


def build_game_date_mapping(seasons):
    """Build mapping from (season, week) to game_date (YYYYMMDD)."""
    print("Building game date mapping from schedules...")
    
    game_date_map = {}
    
    for season in seasons:
        try:
            # Load schedule for this season
            schedule = nfl.import_schedules([season])
            print(f"  Season {season}: {len(schedule)} games")
            
            for _, game in schedule.iterrows():
                if pd.notna(game.get('gameday')) and pd.notna(game.get('week')):
                    week = game['week']
                    game_date = pd.to_datetime(game['gameday'])
                    game_date_str = game_date.strftime('%Y%m%d')
                    
                    # Map this week to the game date
                    game_date_map[(season, week)] = game_date_str
                    
        except Exception as e:
            print(f"  Warning: Could not load schedule for season {season}: {e}")
            # Fallback: generate approximate dates
            for week in range(1, 19):  # Regular season weeks 1-18
                week_start = get_week_start_date(season, week)
                game_date_str = week_start.strftime('%Y%m%d')
                game_date_map[(season, week)] = game_date_str
    
    print(f"Total week-to-date mappings: {len(game_date_map)}")
    return game_date_map


def generate_labels(seasons, out_dir="data/raw/labels"):
    """Generate DraftKings fantasy points labels for specified seasons."""
    print(f"Generating real DraftKings fantasy points labels...")
    print(f"Seasons: {seasons}")
    print(f"Output directory: {out_dir}")
    
    # Ensure output directory exists
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    # Build game date mapping
    game_date_map = build_game_date_mapping(seasons)
    
    total_rows = 0
    
    for season in seasons:
        print(f"\nProcessing season {season}...")
        
        try:
            # Load weekly data for this season
            weekly_data = nfl.import_weekly_data([season])
            print(f"  Loaded {len(weekly_data)} weekly records")
            
            # Filter to only include rows with player_id
            valid_data = weekly_data[weekly_data['player_id'].notna()].copy()
            print(f"  Valid records with player_id: {len(valid_data)}")
            
            # Compute DraftKings fantasy points
            print("  Computing DraftKings fantasy points...")
            valid_data['dk_points'] = valid_data.apply(compute_dk_points, axis=1)
            
            # Add game_date using our mapping
            print("  Adding game dates...")
            valid_data['game_date'] = valid_data.apply(
                lambda row: game_date_map.get((row['season'], row['week']), None), 
                axis=1
            )
            
            # Filter to only include rows with valid game_date
            dated_data = valid_data[valid_data['game_date'].notna()].copy()
            print(f"  Records with valid game_date: {len(dated_data)}")
            
            # Select required columns
            labels = dated_data[[
                'game_date', 'player_id', 'dk_points', 
                'player_name', 'recent_team', 'position'
            ]].copy()
            
            # Rename recent_team to team for consistency
            labels = labels.rename(columns={'recent_team': 'team'})
            
            # Drop any rows with missing values
            final_labels = labels.dropna()
            print(f"  Final clean labels: {len(final_labels)} rows")
            
            # Save to CSV
            output_file = Path(out_dir) / f"labels_{season}.csv"
            final_labels.to_csv(output_file, index=False)
            print(f"  Saved to: {output_file}")
            
            total_rows += len(final_labels)
            
            # Show sample of generated labels
            print(f"  Sample labels:")
            sample = final_labels.head(3)
            for _, row in sample.iterrows():
                print(f"    {row['player_name']} ({row['position']}, {row['team']}): {row['dk_points']} pts")
            
        except Exception as e:
            print(f"  Error processing season {season}: {e}")
            continue
    
    print(f"\n✅ Label generation complete!")
    print(f"Total rows generated: {total_rows:,}")
    print(f"Output directory: {out_dir}")
    
    return total_rows


def main():
    """Main function to generate labels from command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate real DraftKings fantasy points from NFL API data"
    )
    parser.add_argument(
        "--seasons", 
        nargs="+", 
        type=int, 
        default=[2020, 2021, 2022, 2023, 2024],
        help="Seasons to process (default: 2020-2024)"
    )
    parser.add_argument(
        "--out-dir", 
        default="data/raw/labels",
        help="Output directory for label files (default: data/raw/labels)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("GENERATING REAL DRAFTKINGS FANTASY POINTS LABELS")
    print("=" * 60)
    
    try:
        total_rows = generate_labels(args.seasons, args.out_dir)
        print(f"\n🎯 Successfully generated {total_rows:,} real fantasy points labels!")
        
    except Exception as e:
        print(f"\n❌ Error during label generation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
