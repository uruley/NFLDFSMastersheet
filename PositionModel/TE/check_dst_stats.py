#!/usr/bin/env python3
"""
Check NFL API for DST Stats
See what defensive and special teams stats are available
"""
import pandas as pd
import numpy as np
import nfl_data_py as nfl

def explore_dst_stats():
    """Explore available DST stats from NFL API."""
    print("🔍 Exploring NFL API DST Stats")
    print("=" * 50)
    
    # Load recent data for all positions
    print("Loading 2024 data...")
    weekly_data = nfl.import_weekly_data([2024])
    
    print(f"Total records: {len(weekly_data)}")
    print(f"Data shape: {weekly_data.shape}")
    
    # Check for DST-specific columns
    print(f"\n📊 ALL AVAILABLE COLUMNS ({len(weekly_data.columns)} total):")
    for i, col in enumerate(weekly_data.columns):
        print(f"{i+1:2d}. {col}")
    
    # Look for defensive stats
    defensive_cols = [col for col in weekly_data.columns if any(x in col.lower() for x in [
        'defense', 'defensive', 'sack', 'interception', 'fumble', 'safety', 'td', 'touchdown',
        'points_allowed', 'yards_allowed', 'passing', 'rushing', 'total_yards'
    ])]
    
    print(f"\n🛡️ DEFENSIVE STATS ({len(defensive_cols)} columns):")
    for col in defensive_cols:
        non_null = weekly_data[col].notna().sum()
        unique_vals = weekly_data[col].nunique()
        print(f"  • {col}: {non_null} non-null, {unique_vals} unique values")
    
    # Look for special teams stats
    special_teams_cols = [col for col in weekly_data.columns if any(x in col.lower() for x in [
        'kick', 'punt', 'return', 'special', 'block', 'field_goal'
    ])]
    
    print(f"\n🏈 SPECIAL TEAMS STATS ({len(special_teams_cols)} columns):")
    for col in special_teams_cols:
        non_null = weekly_data[col].notna().sum()
        unique_vals = weekly_data[col].nunique()
        print(f"  • {col}: {non_null} non-null, {unique_vals} unique values")
    
    # Check for team-level defensive stats
    print(f"\n🏟️ TEAM-LEVEL ANALYSIS:")
    
    # Only use columns that actually exist
    existing_cols = []
    for col in ['sacks', 'interceptions', 'sack_fumbles', 'sack_fumbles_lost', 'special_teams_tds']:
        if col in weekly_data.columns:
            existing_cols.append(col)
    
    if existing_cols:
        # Group by team and week to see if we can construct DST stats
        team_stats = weekly_data.groupby(['recent_team', 'week', 'season'])[existing_cols].sum().reset_index()
        print(f"Team stats shape: {team_stats.shape}")
        print(f"Sample team stats:")
        print(team_stats.head(10).to_string(index=False))
    
    # Check for individual defensive player stats
    print(f"\n👤 INDIVIDUAL DEFENSIVE PLAYER STATS:")
    
    # Look at defensive positions
    defensive_positions = ['DE', 'DT', 'LB', 'CB', 'S', 'DB', 'DL', 'OLB', 'ILB', 'NT']
    defensive_players = weekly_data[weekly_data['position'].isin(defensive_positions)]
    
    if len(defensive_players) > 0:
        print(f"Defensive players found: {len(defensive_players)}")
        print(f"Positions: {defensive_players['position'].unique()}")
        
        # Check what stats are available for defensive players
        defensive_stats = defensive_players.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['player_id', 'season', 'week', 'fantasy_points', 'fantasy_points_ppr']
        defensive_stats = [col for col in defensive_stats if col not in exclude_cols]
        
        print(f"Available defensive stats: {len(defensive_stats)}")
        for col in defensive_stats[:20]:  # Show first 20
            non_null = defensive_players[col].notna().sum()
            if non_null > 0:
                print(f"  • {col}: {non_null} non-null")
    else:
        print("No defensive players found in standard positions")
    
    # Check for team defense records
    print(f"\n🏈 TEAM DEFENSE RECORDS:")
    
    # Look for records that might be team-level
    team_records = weekly_data[weekly_data['position'].isna() | 
                              (weekly_data['position'] == '') |
                              (weekly_data['position'] == 'DEF') |
                              (weekly_data['position'] == 'DST')]
    
    if len(team_records) > 0:
        print(f"Potential team defense records: {len(team_records)}")
        print(f"Sample team records:")
        print(team_records[['player_name', 'recent_team', 'week', 'position']].head(10).to_string(index=False))
    else:
        print("No obvious team defense records found")
    
    # Check if we can construct DST fantasy points
    print(f"\n🎯 DST FANTASY POINTS CONSTRUCTION:")
    
    # Try to find or construct DST fantasy points
    if 'fantasy_points' in weekly_data.columns:
        # Look for team-level fantasy points
        team_fantasy = weekly_data.groupby(['recent_team', 'week', 'season'])['fantasy_points'].sum().reset_index()
        print(f"Team fantasy points shape: {team_fantasy.shape}")
        
        # Check if this looks like DST scoring
        print(f"Sample team fantasy points:")
        print(team_fantasy.head(10).to_string(index=False))
        
        # Check ranges
        if len(team_fantasy) > 0:
            min_fp = team_fantasy['fantasy_points'].min()
            max_fp = team_fantasy['fantasy_points'].max()
            mean_fp = team_fantasy['fantasy_points'].mean()
            print(f"Team fantasy points range: [{min_fp:.2f}, {max_fp:.2f}], mean: {mean_fp:.2f}")
    
    return weekly_data

def check_dst_availability():
    """Check if we have enough data to model DST."""
    print(f"\n🔍 DST MODELING FEASIBILITY ASSESSMENT:")
    
    # Load data
    weekly_data = nfl.import_weekly_data([2024])
    
    # Check for key DST stats that actually exist
    potential_stats = [
        'sacks', 'interceptions', 'sack_fumbles', 'sack_fumbles_lost', 
        'special_teams_tds', 'passing_yards', 'rushing_yards', 'receiving_yards'
    ]
    
    available_stats = []
    missing_stats = []
    
    for stat in potential_stats:
        if stat in weekly_data.columns:
            non_null = weekly_data[stat].notna().sum()
            if non_null > 0:
                available_stats.append(stat)
                print(f"✅ {stat}: {non_null} records available")
            else:
                missing_stats.append(stat)
                print(f"❌ {stat}: No data available")
        else:
            missing_stats.append(stat)
            print(f"❌ {stat}: Column not found")
    
    # Check for team-level defensive data
    print(f"\n🏟️ TEAM-LEVEL DEFENSIVE DATA:")
    
    # Check if we can aggregate individual defensive stats to team level
    if 'sacks' in weekly_data.columns:
        team_sacks = weekly_data.groupby(['recent_team', 'week', 'season'])['sacks'].sum()
        print(f"✅ Team sacks aggregation: {len(team_sacks)} team-week records")
    
    if 'interceptions' in weekly_data.columns:
        team_ints = weekly_data.groupby(['recent_team', 'week', 'season'])['interceptions'].sum()
        print(f"✅ Team interceptions aggregation: {len(team_ints)} team-week records")
    
    if 'special_teams_tds' in weekly_data.columns:
        team_st_tds = weekly_data.groupby(['recent_team', 'week', 'season'])['special_teams_tds'].sum()
        print(f"✅ Team special teams TDs aggregation: {len(team_st_tds)} team-week records")
    
    print(f"\n📊 SUMMARY:")
    print(f"Available individual defensive stats: {len(available_stats)}/{len(potential_stats)}")
    print(f"Missing individual defensive stats: {len(missing_stats)}")
    
    if len(available_stats) >= 4:
        print(f"🎯 DST MODELING: PARTIALLY FEASIBLE with NFL API data")
        print(f"Available features: {', '.join(available_stats)}")
        print(f"Can aggregate individual stats to team level")
    elif len(available_stats) >= 2:
        print(f"⚠️ DST MODELING: LIMITED with NFL API data")
        print(f"Available features: {', '.join(available_stats)}")
        print(f"May need additional data sources for complete DST modeling")
    else:
        print(f"❌ DST MODELING: NOT FEASIBLE with NFL API data")
        print(f"Need alternative data sources or different approach")
    
    return available_stats, missing_stats

def main():
    """Main exploration function."""
    print("🚀 NFL API DST Stats Exploration")
    print("=" * 50)
    
    # Explore available stats
    weekly_data = explore_dst_stats()
    
    # Check DST modeling feasibility
    available_stats, missing_stats = check_dst_availability()
    
    print(f"\n🎯 EXPLORATION COMPLETE!")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if len(available_stats) >= 4:
        print(f"• NFL API has some defensive data that can be aggregated to team level")
        print(f"• Use team-level aggregation for sacks, interceptions, fumbles, special teams TDs")
        print(f"• Supplement with external sources for points allowed, yards allowed")
        print(f"• Consider hybrid approach: NFL API for individual stats + external for team stats")
    elif len(available_stats) >= 2:
        print(f"• NFL API has limited defensive data")
        print(f"• Supplement heavily with external sources (ESPN, Pro Football Reference)")
        print(f"• Consider different modeling approach or data source")
    else:
        print(f"• NFL API insufficient for DST modeling")
        print(f"• Need alternative data sources (ESPN, Pro Football Reference, etc.)")
        print(f"• Consider different modeling approach (team-based vs player-based)")

if __name__ == "__main__":
    main()
