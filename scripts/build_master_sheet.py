#!/usr/bin/env python3
"""
First-pass pipeline for merging DraftKings salaries with roster data.
Applies cleaning rules and computes join keys for exact matching.
"""

import argparse
import pandas as pd
import re
import os
from pathlib import Path
from datetime import datetime


def generate_slate_id(df, source_path):
    """Generate unique ID for this specific slate"""
    slate_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    slate_size = len(df)
    return f"slate_{slate_date}_{slate_size}"


def clean_name(name):
    """Apply name cleaning rules: lowercase → remove punctuation → collapse spaces → drop suffix"""
    if pd.isna(name):
        return ""
    
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


def process_dk_file(dk_file, roster_df, roster_lookup, alias_lookup, args):
    """Process a single DK salary file and generate outputs"""
    import os
    import re
    
    # Read DK data
    print(f"Reading DraftKings salaries from {os.path.basename(dk_file)}...")
    dk_df = pd.read_csv(dk_file)
    print(f"DK data: {len(dk_df)} rows")
    print(f"DK columns: {list(dk_df.columns)}")
    
    # Generate slate ID and save snapshot
    slate_id = generate_slate_id(dk_df, dk_file)
    
    # Save slate snapshot
    slate_snapshot_path = f'data/processed/history/2025/{slate_id}_DKSalaries.csv'
    os.makedirs(os.path.dirname(slate_snapshot_path), exist_ok=True)
    dk_df.to_csv(slate_snapshot_path, index=False)
    print(f"Saved slate snapshot: {slate_snapshot_path}")
    
    # Also save as "current" slate for reference
    dk_df.to_csv('data/processed/current_slate.csv', index=False)
    print(f"Saved current slate: data/processed/current_slate.csv")
    
    # Parse slate date from filename if in batch mode
    slate_date = None
    if args.out_dir:
        # Extract date from filename: DKSalaries_YYYYMMDD.csv
        filename = os.path.basename(dk_file)
        date_match = re.search(r'DKSalaries_(\d{8})\.csv', filename)
        if date_match:
            slate_date = date_match.group(1)
            print(f"Parsed slate date: {slate_date}")
    
    # Add slate_date column to DK dataframe
    if slate_date:
        dk_df['slate_date'] = slate_date
    else:
        # Try to extract from Game Info if available
        if 'Game Info' in dk_df.columns:
            # Extract date from Game Info (assuming format like "NE@MIA 08/24 7:00PM ET")
            game_info_dates = dk_df['Game Info'].str.extract(r'(\d{2}/\d{2})')
            if not game_info_dates.empty and game_info_dates.iloc[0, 0]:
                # Convert MM/DD to YYYYMMDD (assuming current year)
                from datetime import datetime
                try:
                    date_str = game_info_dates.iloc[0, 0]
                    parsed_date = datetime.strptime(f"{args.season}/{date_str}", "%Y/%m/%d")
                    slate_date = parsed_date.strftime("%Y%m%d")
                    dk_df['slate_date'] = slate_date
                    print(f"Extracted slate date from Game Info: {slate_date}")
                except:
                    dk_df['slate_date'] = "unknown"
            else:
                dk_df['slate_date'] = "unknown"
        else:
            dk_df['slate_date'] = "unknown"
    
    # Apply cleaning rules to DK data
    print("Applying cleaning rules to DK data...")
    # Apply aliases first, then clean
    dk_df['name_for_cleaning'] = dk_df['Name'].map(alias_lookup).fillna(dk_df['Name'])
    dk_df['clean_name'] = dk_df['name_for_cleaning'].apply(clean_name)
    dk_df['norm_team'] = dk_df['TeamAbbrev'].apply(norm_team)
    dk_df['norm_pos'] = dk_df['Position'].apply(norm_pos)
    dk_df['join_key'] = dk_df.apply(lambda row: compute_join_key({**row, 'Name': row['name_for_cleaning']}, True), axis=1)
    
    # Merge DK data with roster data
    print("Merging data...")
    dk_df['player_id'] = dk_df['join_key'].map(roster_lookup)
    
    # Extract game_date from Game Info column
    print("Extracting game_date from Game Info...")
    dk_df['game_date'] = dk_df['Game Info'].str.extract(r'(\d{2}/\d{2}/\d{4})')
    if not dk_df['game_date'].empty and dk_df['game_date'].iloc[0]:
        # Convert MM/DD/YYYY to YYYYMMDD
        from datetime import datetime
        try:
            dk_df['game_date'] = pd.to_datetime(dk_df['game_date'], format='%m/%d/%Y').dt.strftime('%Y%m%d')
            print(f"Extracted game_date from Game Info")
        except:
            dk_df['game_date'] = dk_df['slate_date']  # Fallback to slate_date
            print(f"Failed to parse game_date, using slate_date as fallback")
    else:
        # Try alternative format MM/DD
        game_info_dates = dk_df['Game Info'].str.extract(r'(\d{2}/\d{2})')
        if not game_info_dates.empty and game_info_dates.iloc[0, 0]:
            try:
                date_str = game_info_dates.iloc[0, 0]
                parsed_date = datetime.strptime(f"{args.season}/{date_str}", "%Y/%m/%d")
                dk_df['game_date'] = parsed_date.strftime("%Y%m%d")
                print(f"Extracted game_date from MM/DD format: {dk_df['game_date'].iloc[0]}")
            except:
                dk_df['game_date'] = dk_df['slate_date']  # Fallback to slate_date
                print(f"Failed to parse game_date, using slate_date as fallback")
        else:
            dk_df['game_date'] = dk_df['slate_date']  # Fallback to slate_date
            print(f"Could not extract game_date, using slate_date as fallback")
    
    # Build name_team_key for fallback matching
    print("Building name_team_key for fallback matching...")
    dk_df['name_team_key'] = dk_df['clean_name'] + "|" + dk_df['norm_team']
    roster_df['name_team_key'] = roster_df['clean_name'] + "|" + roster_df['norm_team']
    
    # Create lookup for name_team_key fallback matching
    name_team_lookup = {}
    for _, row in roster_df.iterrows():
        if row['name_team_key'] and row['position'] in ['LS', 'K', 'FB']:
            key = row['name_team_key']
            if key not in name_team_lookup:
                name_team_lookup[key] = []
            name_team_lookup[key].append(row['player_id'])
    
    # Apply fallback matching for unmatched rows
    print("Applying fallback name+team matching...")
    fallback_matches = 0
    for idx, row in dk_df.iterrows():
        if pd.isna(row['player_id']) and row['name_team_key'] in name_team_lookup:
            candidates = name_team_lookup[row['name_team_key']]
            if len(candidates) == 1:  # Only accept if exactly one candidate
                dk_df.at[idx, 'player_id'] = candidates[0]
                dk_df.at[idx, 'method'] = 'fallback_name_team'
                fallback_matches += 1
                print(f"  Fallback match: {row['Name']} -> {candidates[0]} (method: fallback_name_team)")
    
    print(f"Fallback matches applied: {fallback_matches}")
    
    # Print summary of matches
    total_dk = len(dk_df)
    matched_count = len(dk_df[dk_df['player_id'].notna()])
    match_pct = matched_count/total_dk*100
    
    print(f"\n=== MATCHING SUMMARY ===")
    print(f"Roster rows: {len(roster_df)}")
    print(f"DK rows: {total_dk}")
    print(f"Match %: {match_pct:.1f}%")
    print(f"Sample matches:")
    sample_matches = dk_df[dk_df['player_id'].notna()].head(3)[['Name', 'TeamAbbrev', 'Position', 'player_id']]
    for _, row in sample_matches.iterrows():
        print(f"  {row['Name']} ({row['TeamAbbrev']}, {row['Position']}) -> {row['player_id']}")
    
    # Determine match method
    def get_match_method(row):
        if pd.isna(row['player_id']):
            return 'unmatched'
        elif row['Position'] == 'DST':
            return 'dst'
        elif 'method' in row and row['method'] == 'fallback_name_team':
            return 'fallback_name_team'
        else:
            return 'exact'
    
    dk_df['method'] = dk_df.apply(get_match_method, axis=1)
    
    # Determine output paths
    if args.out_dir and slate_date:
        # Batch mode with slate date
        master_out = f"{args.out_dir}/master_sheet_{args.season}_{slate_date}.csv"
        crosswalk_out = f"{args.out_dir}/crosswalk_{args.season}_{slate_date}.csv"
        unmatched_out = f"{args.out_dir}/unmatched_{args.season}_{slate_date}.csv"
    else:
        # Single file mode or no slate date
        master_out = args.master_out
        crosswalk_out = args.crosswalk_out
        unmatched_out = args.unmatched_out
    
    # Ensure output directory exists
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
    
    # Write master sheet (all DK rows + matched player_id)
    print(f"Writing master sheet to {master_out}...")
    master_columns = list(dk_df.columns)
    dk_df[master_columns].to_csv(master_out, index=False)
    
    # Write crosswalk (DK key columns + method + slate_date + game_date)
    print(f"Writing crosswalk to {crosswalk_out}...")
    crosswalk_columns = ['Name', 'Position', 'TeamAbbrev', 'Salary', 'Game Info', 'slate_date', 'game_date', 'join_key', 'method']
    crosswalk_df = dk_df[crosswalk_columns].copy()
    crosswalk_df.to_csv(crosswalk_out, index=False)
    
    # Write unmatched (DK rows with null player_id, sorted by Salary desc)
    print(f"Writing unmatched to {unmatched_out}...")
    unmatched_df = dk_df[dk_df['player_id'].isna()].copy()
    unmatched_df = unmatched_df.sort_values('Salary', ascending=False)
    unmatched_columns = ['Name', 'Position', 'TeamAbbrev', 'Salary', 'slate_date', 'game_date', 'join_key']
    unmatched_df[unmatched_columns].to_csv(unmatched_out, index=False)
    
    # Final validation and reporting
    total_dk = len(dk_df)
    matched_count = len(dk_df[dk_df['player_id'].notna()])
    unmatched_count = len(dk_df[dk_df['player_id'].isna()])
    match_pct = matched_count/total_dk*100
    
    print(f"\n=== FILE SUMMARY ===")
    print(f"File: {os.path.basename(dk_file)}")
    print(f"Total DK rows: {total_dk}")
    print(f"Matched: {matched_count} ({match_pct:.1f}%)")
    print(f"Unmatched: {unmatched_count}")
    
    # Show top 5 unmatched by salary
    if unmatched_count > 0:
        print(f"Top unmatched by salary:")
        top_unmatched = unmatched_df.head(5)[['Name', 'Position', 'TeamAbbrev', 'Salary']]
        for _, row in top_unmatched.iterrows():
            print(f"  {row['Name']} ({row['Position']}, {row['TeamAbbrev']}) - ${row['Salary']:,}")
    
    # Return report data for season tracking
    return {
        'slate_date': slate_date or 'unknown',
        'total_rows': total_dk,
        'matched': matched_count,
        'unmatched': unmatched_count,
        'matched_pct': match_pct
    }


def write_season_report(out_dir, season, report_rows):
    """Write season report with match statistics across all slates"""
    import os
    import pandas as pd
    
    # Count rows in xwalk files
    alias_rows = 0
    dst_rows = 0
    roster_additions_rows = 0
    
    try:
        aliases_df = pd.read_csv("data/xwalk/aliases.csv")
        alias_rows = len(aliases_df)
    except FileNotFoundError:
        pass
    
    try:
        dst_df = pd.read_csv("data/xwalk/synthetic_dst_roster_rows.csv")
        dst_rows = len(dst_df)
    except FileNotFoundError:
        pass
    
    try:
        additions_df = pd.read_csv("data/xwalk/roster_additions.csv")
        roster_additions_rows = len(additions_df)
    except FileNotFoundError:
        pass
    
    # Add xwalk file counts to each report row
    for row in report_rows:
        row['alias_rows'] = alias_rows
        row['dst_rows'] = dst_rows
        row['roster_additions_rows'] = roster_additions_rows
    
    # Create DataFrame and write to CSV
    report_df = pd.DataFrame(report_rows)
    report_path = f"{out_dir}/match_report_{season}.csv"
    
    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)
    
    # Write report
    report_df.to_csv(report_path, index=False)
    print(f"\nSeason report written to: {report_path}")
    print(f"Report contains {len(report_rows)} slate(s)")


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


def main():
    parser = argparse.ArgumentParser(description='Build master sheet from DK salaries and roster data')
    parser.add_argument('--season', default='2025', help='Season year (default: 2025)')
    parser.add_argument('--dk', help='Path to specific DraftKings salaries CSV (if not provided, processes all in --dk-dir)')
    parser.add_argument('--rosters', help='Path to roster data CSV (defaults to data/raw/rosters/rosters_<season>.csv)')
    parser.add_argument('--dk-dir', help='Directory containing DK salary files (defaults to data/raw/dk_slates/<season>/)')
    parser.add_argument('--out-dir', help='Output directory for processed files (defaults to data/processed/history/<season>/)')
    parser.add_argument('--master-out', help='Output path for master sheet (overrides --out-dir)')
    parser.add_argument('--crosswalk-out', help='Output path for crosswalk data (overrides --out-dir)')
    parser.add_argument('--unmatched-out', help='Output path for unmatched DK rows (overrides --out-dir)')
    parser.add_argument('--use-nfl-api-roster', action='store_true', help='Use NFL API roster data instead of external roster file')
    
    args = parser.parse_args()
    
    # Initialize season report tracking
    season_report_rows = []
    
    # Set default values for season-specific files
    if not args.rosters:
        args.rosters = f"data/raw/rosters/rosters_{args.season}.csv"
    if not args.dk_dir:
        args.dk_dir = f"data/raw/dk_slates/{args.season}/"
    if not args.out_dir:
        args.out_dir = f"data/processed/history/{args.season}/"
    
    # Set output paths (--out-dir takes precedence over individual file paths)
    if args.out_dir:
        if not args.master_out:
            args.master_out = f"{args.out_dir}/master_sheet_{args.season}.csv"
        if not args.crosswalk_out:
            args.crosswalk_out = f"{args.out_dir}/crosswalk_{args.season}.csv"
        if not args.unmatched_out:
            args.unmatched_out = f"{args.out_dir}/unmatched_{args.season}.csv"
    else:
        # Fallback to old defaults if no --out-dir
        if not args.master_out:
            args.master_out = f"data/processed/master_sheet_{args.season}.csv"
        if not args.crosswalk_out:
            args.crosswalk_out = f"data/processed/crosswalk_{args.season}.csv"
        if not args.unmatched_out:
            args.unmatched_out = f"data/processed/unmatched_{args.season}.csv"
    
    print(f"Processing season: {args.season}")
    
    # Determine processing mode
    if args.dk:
        # Single file mode
        dk_files = [args.dk]
        print(f"Single file mode: processing {args.dk}")
    else:
        # Batch mode - find all DK files in directory
        import glob
        
        # Ensure directory exists
        if not os.path.exists(args.dk_dir):
            print(f"DK directory not found: {args.dk_dir}")
            return
        
        # Find all DK salary files
        dk_files = glob.glob(os.path.join(args.dk_dir, "DKSalaries_*.csv"))
        dk_files.sort()  # Sort for consistent processing order
        
        if not dk_files:
            print(f"No DK salary files found in {args.dk_dir}")
            return
        
        print(f"Batch mode: found {len(dk_files)} DK files in {args.dk_dir}")
        for dk_file in dk_files:
            print(f"  {os.path.basename(dk_file)}")
    
    # Read and prepare roster data (once for all files)
    print("\nReading roster data...")
    if args.use_nfl_api_roster:
        print("Using NFL API roster data...")
        import nfl_data_py as nfl
        roster_df = nfl.import_seasonal_rosters([int(args.season)])
        print(f"NFL API roster data: {len(roster_df)} rows")
        print(f"NFL API roster columns: {list(roster_df.columns)}")
        
        # Normalize columns to match what the script expects
        roster_df = roster_df.rename(columns={
            'player_name': 'player_name',  # Keep as-is
            'recent_team': 'team',
            'position': 'position',
            'player_id': 'player_id'  # Keep as-is (this matches weekly)
        })
        
        # Filter to only keep the columns we need
        roster_df = roster_df[['player_name', 'team', 'position', 'player_id']].copy()
    else:
        roster_df = pd.read_csv(args.rosters)
        print(f"External roster data: {len(roster_df)} rows")
        print(f"External roster columns: {list(roster_df.columns)}")
    
    print(f"Final roster data: {len(roster_df)} rows")
    print(f"Final roster columns: {list(roster_df.columns)}")
    
    # Read and union synthetic DST rows
    print("Reading synthetic DST roster rows...")
    try:
        dst_df = pd.read_csv("data/xwalk/synthetic_dst_roster_rows.csv")
        print(f"Synthetic DST rows: {len(dst_df)} rows")
        # Union the DST rows into the roster dataframe
        roster_df = pd.concat([roster_df, dst_df], ignore_index=True)
        print(f"Combined roster data: {len(roster_df)} rows")
    except FileNotFoundError:
        print("No synthetic DST roster file found, proceeding with original roster data")
    
    # Read and union roster additions
    print("Reading roster additions...")
    try:
        additions_df = pd.read_csv("data/xwalk/roster_additions.csv")
        print(f"Roster additions: {len(additions_df)} rows")
        # Union the additions into the roster dataframe
        roster_df = pd.concat([roster_df, additions_df], ignore_index=True)
        print(f"Combined roster data: {len(roster_df)} rows")
    except FileNotFoundError:
        print("No roster additions file found, proceeding with existing roster data")
    
    # Read aliases and create lookup
    print("Reading aliases...")
    alias_lookup = {}
    try:
        aliases_df = pd.read_csv("data/xwalk/aliases.csv")
        print(f"Aliases: {len(aliases_df)} rows")
        for _, row in aliases_df.iterrows():
            if pd.notna(row['dk_name']) and pd.notna(row['roster_full_name']):
                alias_lookup[row['dk_name']] = row['roster_full_name']
                print(f"  Alias: {row['dk_name']} -> {row['roster_full_name']}")
    except FileNotFoundError:
        print("No aliases file found")
    
    # Apply cleaning rules to roster data (once for all files)
    print("Applying cleaning rules to roster data...")
    # Handle both original roster columns and synthetic DST columns
    if 'player_name' in roster_df.columns:
        roster_df['clean_name'] = roster_df['player_name'].apply(clean_name)
    else:
        roster_df['clean_name'] = roster_df['full_name'].apply(clean_name)
    
    roster_df['norm_team'] = roster_df['team'].apply(norm_team)
    roster_df['norm_pos'] = roster_df['position'].apply(norm_pos)
    roster_df['join_key'] = roster_df.apply(compute_join_key, axis=1, args=(False,))
    
    # Create lookup dictionary for roster data
    roster_lookup = {}
    for _, row in roster_df.iterrows():
        if row['join_key']:  # Skip empty join keys
            roster_lookup[row['join_key']] = row['player_id']
    
    # Apply manual overrides from xwalk_manual.csv (last priority)
    print("Reading manual crosswalk overrides...")
    try:
        manual_df = pd.read_csv("data/xwalk/xwalk_manual.csv")
        print(f"Manual overrides: {len(manual_df)} rows")
        for _, row in manual_df.iterrows():
            if row['join_key'] and row['player_id']:
                roster_lookup[row['join_key']] = row['player_id']
                print(f"  Override: {row['join_key']} -> {row['player_id']} ({row['reason']})")
    except FileNotFoundError:
        print("No manual crosswalk file found")
    
    # Process each DK file
    for dk_file in dk_files:
        print(f"\n{'='*60}")
        print(f"Processing: {os.path.basename(dk_file)}")
        print(f"{'='*60}")
        
        # Process single DK file and collect report data
        report_data = process_dk_file(dk_file, roster_df, roster_lookup, alias_lookup, args)
        if report_data:
            season_report_rows.append(report_data)
    
    # Write season report if we have data and are in batch mode
    if season_report_rows and args.out_dir:
        write_season_report(args.out_dir, args.season, season_report_rows)


if __name__ == "__main__":
    main()
