#!/usr/bin/env python3
"""
Build player ID mapping between NFL API and our roster data.
Uses name, team, and position matching to create a crosswalk.
"""

import argparse
import pandas as pd
import re
from pathlib import Path

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

def build_mapping(nfl_labels_path, roster_path, out_path):
    """Build player ID mapping between NFL API and roster data."""
    print("Building player ID mapping...")
    
    # Load data
    print("Loading NFL labels...")
    nfl_df = pd.read_csv(nfl_labels_path)
    print(f"NFL labels: {len(nfl_df)} rows")
    
    print("Loading roster data...")
    roster_df = pd.read_csv(roster_path)
    print(f"Roster: {len(roster_df)} rows")
    
    # Clean and normalize NFL data
    print("Cleaning NFL data...")
    nfl_df['clean_name'] = nfl_df['player_name'].apply(clean_name)
    nfl_df['norm_team'] = nfl_df['recent_team'].apply(norm_team)
    nfl_df['norm_pos'] = nfl_df['position'].apply(norm_pos)
    
    # Clean and normalize roster data
    print("Cleaning roster data...")
    if 'player_name' in roster_df.columns:
        roster_df['clean_name'] = roster_df['player_name'].apply(clean_name)
    else:
        roster_df['clean_name'] = roster_df['full_name'].apply(clean_name)
    
    roster_df['norm_team'] = roster_df['team'].apply(norm_team)
    roster_df['norm_pos'] = roster_df['position'].apply(norm_pos)
    
    # Create join keys
    nfl_df['join_key'] = nfl_df['clean_name'] + "|" + nfl_df['norm_team'] + "|" + nfl_df['norm_pos']
    roster_df['join_key'] = roster_df['clean_name'] + "|" + roster_df['norm_team'] + "|" + roster_df['norm_pos']
    
    # Create roster lookup
    roster_lookup = {}
    for _, row in roster_df.iterrows():
        if row['join_key']:
            roster_lookup[row['join_key']] = row['player_id']
    
    # Match NFL players to roster
    print("Matching players...")
    nfl_df['roster_player_id'] = nfl_df['join_key'].map(roster_lookup)
    
    # Count matches
    total_nfl = len(nfl_df)
    matched = nfl_df['roster_player_id'].notna().sum()
    unmatched = total_nfl - matched
    
    print(f"Total NFL players: {total_nfl}")
    print(f"Matched to roster: {matched} ({matched/total_nfl*100:.1f}%)")
    print(f"Unmatched: {unmatched} ({unmatched/total_nfl*100:.1f}%)")
    
    # Create mapping dataframe
    mapping_df = nfl_df[['player_id', 'roster_player_id', 'player_name', 'recent_team', 'position', 
                         'clean_name', 'norm_team', 'norm_pos', 'join_key']].copy()
    
    # Rename columns for clarity
    mapping_df = mapping_df.rename(columns={
        'player_id': 'nfl_player_id',
        'roster_player_id': 'roster_player_id',
        'recent_team': 'nfl_team',
        'position': 'nfl_position'
    })
    
    # Add match status
    mapping_df['matched'] = mapping_df['roster_player_id'].notna()
    
    # Sort by match status, then by name
    mapping_df = mapping_df.sort_values(['matched', 'player_name'])
    
    # Write mapping
    mapping_df.to_csv(out_path, index=False)
    print(f"Mapping written to: {out_path}")
    
    # Show sample of unmatched
    if unmatched > 0:
        print(f"\nSample unmatched players:")
        unmatched_sample = mapping_df[~mapping_df['matched']].head(10)
        for _, row in unmatched_sample.iterrows():
            print(f"  {row['player_name']} ({row['nfl_position']}, {row['nfl_team']})")
    
    return mapping_df

def main():
    parser = argparse.ArgumentParser(description='Build player ID mapping between NFL API and roster')
    parser.add_argument('--nfl-labels', required=True, help='Path to NFL labels CSV')
    parser.add_argument('--roster', required=True, help='Path to roster CSV')
    parser.add_argument('--out', default='data/xwalk/nfl_to_roster_mapping.csv', 
                       help='Output path for mapping CSV')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("BUILDING NFL API TO ROSTER PLAYER ID MAPPING")
    print("=" * 60)
    
    # Ensure output directory exists
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Build mapping
    mapping_df = build_mapping(args.nfl_labels, args.roster, args.out)
    
    print(f"\nMapping complete! Output: {args.out}")

if __name__ == "__main__":
    main()

