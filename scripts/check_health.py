#!/usr/bin/env python3
"""
Health check script for master sheet match rates.
Reads match reports and identifies slates with poor match rates.
Optionally suggests fixes for unmatched players.
"""

import argparse
import pandas as pd
import os
import glob
from pathlib import Path


def load_match_report(season, custom_path=None):
    """Load match report for a given season"""
    if custom_path:
        report_path = custom_path
    else:
        report_path = f"data/processed/history/{season}/match_report_{season}.csv"
    
    if not os.path.exists(report_path):
        raise FileNotFoundError(f"Match report not found: {report_path}")
    
    print(f"Loading match report: {report_path}")
    df = pd.read_csv(report_path)
    print(f"Found {len(df)} slate(s) in report")
    
    return df


def check_health(df, season):
    """Check health of match rates and report issues"""
    print(f"\n=== HEALTH CHECK FOR SEASON {season} ===")
    
    # Check for poor match rates
    poor_matches = df[(df['matched_pct'] < 99.0) | (df['unmatched'] > 0)]
    
    if len(poor_matches) == 0:
        print("✅ All slates have perfect match rates (100%)")
        return poor_matches
    
    print(f"⚠️  Found {len(poor_matches)} slate(s) with match issues:")
    print()
    
    # Sort by worst match rate first
    poor_matches = poor_matches.sort_values('matched_pct')
    
    for _, row in poor_matches.iterrows():
        slate_date = row['slate_date']
        total_rows = row['total_rows']
        matched = row['matched']
        unmatched = row['unmatched']
        matched_pct = row['matched_pct']
        
        print(f"Slate {slate_date}:")
        print(f"  Total: {total_rows:,} | Matched: {matched:,} | Unmatched: {unmatched:,} | Rate: {matched_pct:.1f}%")
    
    print()
    return poor_matches


def suggest_fixes(poor_matches, season):
    """Suggest fixes for unmatched players"""
    if len(poor_matches) == 0:
        print("No fix suggestions needed - all slates have perfect match rates!")
        return
    
    print("=== FIX SUGGESTIONS ===")
    print()
    
    for _, row in poor_matches.iterrows():
        slate_date = row['slate_date']
        unmatched_count = row['unmatched']
        
        if unmatched_count == 0:
            continue
            
        print(f"Slate {slate_date} ({unmatched_count} unmatched):")
        
        # Load unmatched file for this slate
        slate_date_str = str(int(slate_date)) if isinstance(slate_date, float) else str(slate_date)
        unmatched_path = f"data/processed/history/{season}/unmatched_{season}_{slate_date_str}.csv"
        
        if not os.path.exists(unmatched_path):
            print(f"  ⚠️  Unmatched file not found: {unmatched_path}")
            continue
        
        try:
            unmatched_df = pd.read_csv(unmatched_path)
            
            if len(unmatched_df) == 0:
                print("  ✅ No unmatched players in file")
                continue
            
            # Sort by salary descending and take top 10
            top_unmatched = unmatched_df.sort_values('Salary', ascending=False).head(10)
            
            print(f"  Top unmatched players (sorted by salary):")
            
            # Group by position for better organization
            for position in top_unmatched['Position'].unique():
                pos_players = top_unmatched[top_unmatched['Position'] == position]
                print(f"    {position}:")
                
                for _, player in pos_players.iterrows():
                    name = player['Name']
                    team = player['TeamAbbrev']
                    salary = player['Salary']
                    pos = player['Position']
                    
                    # Generate stub suggestions
                    print(f"      {name} ({team}, ${salary:,})")
                    
                    # Suggest alias entry (user will need to find correct roster name)
                    print(f"        Alias stub: {name},<FIND_ROSTER_NAME>")
                    
                    # Suggest roster addition (user will need to create player_id)
                    clean_name = name.lower().replace(' ', '').replace('.', '').replace("'", '')
                    norm_team = team
                    norm_pos = 'DST' if pos in ['D/ST', 'DEF'] else pos.upper()
                    join_key = f"{clean_name}|{norm_team}|{norm_pos}"
                    
                    print(f"        Addition stub: EXT-{clean_name[:8]}-{team}-{pos},{name},{team},{pos},{clean_name},{norm_team},{norm_pos},{join_key}")
            
            print()
            
        except Exception as e:
            print(f"  ❌ Error reading unmatched file: {e}")
            continue


def main():
    parser = argparse.ArgumentParser(description='Health check for master sheet match rates')
    parser.add_argument('--season', required=True, help='Season to check (e.g., 2025)')
    parser.add_argument('--fix-suggestions', action='store_true', 
                       help='Generate fix suggestions for unmatched players')
    
    args = parser.parse_args()
    
    try:
        # Load match report
        match_report = load_match_report(args.season)
        
        # Check health
        poor_matches = check_health(match_report, args.season)
        
        # Suggest fixes if requested
        if args.fix_suggestions:
            suggest_fixes(poor_matches, args.season)
        
        # Summary
        total_slates = len(match_report)
        problem_slates = len(poor_matches)
        
        print("=== SUMMARY ===")
        print(f"Season: {args.season}")
        print(f"Total slates: {total_slates}")
        print(f"Problem slates: {problem_slates}")
        
        if problem_slates == 0:
            print("✅ All slates healthy!")
        else:
            print(f"⚠️  {problem_slates}/{total_slates} slates need attention")
            
            if args.fix_suggestions:
                print("\n💡 Review the fix suggestions above and:")
                print("   1. Add aliases to data/xwalk/aliases.csv")
                print("   2. Add missing players to data/xwalk/roster_additions.csv")
                print("   3. Re-run build_master_sheet.py for the season")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
