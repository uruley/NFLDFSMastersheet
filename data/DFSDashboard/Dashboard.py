#!/usr/bin/env python3
"""
PlayerMaster v2 Enrichment Script - Enhanced with QB Projection Debugging
Enriches existing PlayerMaster (with projections/SHAP) with current-week Vegas + TeamModels outputs.

Usage:
    python Dashboard.py --season 2025 --week 1 \
        --players data/processed/PlayerMaster_unified.csv \
        --teams data/outputs/predictions_2025w01.csv \
        --outdir data/processed

Outputs:
    - PlayerMaster_v2_<season>w<week>.csv (enriched player sheet)
    - *_unmatched_players.csv (players without team predictions)
    - Coverage summary printed to console
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def debug_projection_columns(df: pd.DataFrame, stage: str) -> None:
    """Debug helper to track projection-related columns at each stage."""
    print(f"\n=== DEBUG: {stage.upper()} ===")
    projection_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in 
                      ['proj', 'point', 'predict', 'forecast'])]
    
    print(f"Projection-related columns found: {projection_cols}")
    
    if projection_cols:
        for col in projection_cols:
            non_null = df[col].notna().sum()
            non_zero = (df[col] > 0).sum() if df[col].dtype in ['float64', 'int64'] else 0
            unique_vals = df[col].nunique()
            print(f"  {col}: {non_null} non-null, {non_zero} non-zero, {unique_vals} unique values")
            
            # Show sample values for debugging
            sample_vals = df[col].dropna().head(5).tolist()
            print(f"    Sample values: {sample_vals}")
    
    # Specifically check QB projections if position column exists
    if 'position' in df.columns:
        qb_mask = df['position'].str.upper() == 'QB'
        qb_count = qb_mask.sum()
        print(f"QB players found: {qb_count}")
        
        if qb_count > 0 and projection_cols:
            for col in projection_cols:
                qb_projections = df.loc[qb_mask, col].notna().sum()
                qb_nonzero = (df.loc[qb_mask, col] > 0).sum() if df[col].dtype in ['float64', 'int64'] else 0
                print(f"  QB {col}: {qb_projections} non-null, {qb_nonzero} non-zero")
                
                if qb_nonzero > 0:
                    qb_sample = df.loc[qb_mask & (df[col] > 0), ['Name', col]].head(3)
                    print(f"    Sample QB projections:\n{qb_sample.to_string(index=False)}")

def harmonize_player_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names and map common aliases to the canonical schema so we don't
    accidentally drop/overwrite projections during schema enforcement.
    """
    print(f"\n[HARMONIZE] Starting with {len(df.columns)} columns")
    debug_projection_columns(df, "Before Harmonization")
    
    # Strip whitespace and standardize exact strings
    df = df.copy()
    original_columns = list(df.columns)
    df.columns = [c.strip() for c in df.columns]
    
    # Check if any column names changed due to whitespace
    stripped_changes = [(orig, new) for orig, new in zip(original_columns, df.columns) if orig != new]
    if stripped_changes:
        print(f"[HARMONIZE] Whitespace stripped from columns: {stripped_changes}")

    # Log exact column names post-stripping for debugging
    print(f"[HARMONIZE] Columns after stripping whitespace: {list(df.columns)}")

    # Enhanced alias map → canonical
    alias_map = {
        # names / teams / pos / salary
        "player_name": "Name",
        "team": "TeamAbbrev",
        "recent_team": "TeamAbbrev",
        "pos": "position",
        "dk_salary": "salary",
        "salary_dk": "salary",
        "draftkings_salary": "salary",
        
        # projections - be more comprehensive
        "projection": "proj_points",
        "proj": "proj_points",
        "projected_points": "proj_points",
        "predicted_points": "proj_points",
        "predicted_points_ensemble": "proj_points",
        "fantasy_points": "proj_points",
        "fantasy_points_proj": "proj_points",
        "points_projection": "proj_points",
        "dk_projection": "proj_points",
        "draftkings_projection": "proj_points",
        
        # Handle common variations
        "Proj_Points": "proj_points",  # Case variations
        "PROJ_POINTS": "proj_points",
        "proj_pts": "proj_points",
    }
    
    # Apply aliases with detailed logging
    for src, dst in alias_map.items():
        if src in df.columns and dst not in df.columns:
            print(f"[HARMONIZE] Renaming '{src}' → '{dst}'")
            df.rename(columns={src: dst}, inplace=True)
        elif src in df.columns and dst in df.columns:
            print(f"[HARMONIZE] WARNING: Both '{src}' and '{dst}' exist, keeping '{dst}'")

    # Additional check for any projection column we might have missed
    potential_proj_cols = [col for col in df.columns if 
                          any(keyword in col.lower() for keyword in ['proj', 'point', 'predict']) 
                          and col != 'proj_points']
    
    if potential_proj_cols and 'proj_points' not in df.columns:
        print(f"[HARMONIZE] WARNING: Found potential projection columns but no 'proj_points': {potential_proj_cols}")
        # Auto-select the most likely candidate
        best_candidate = None
        for col in potential_proj_cols:
            if any(keyword in col.lower() for keyword in ['projection', 'proj_point', 'fantasy_point']):
                best_candidate = col
                break
        
        if best_candidate:
            print(f"[HARMONIZE] Auto-selecting '{best_candidate}' as proj_points")
            df.rename(columns={best_candidate: 'proj_points'}, inplace=True)

    debug_projection_columns(df, "After Harmonization")
    return df

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Enrich PlayerMaster with TeamModels predictions")
    parser.add_argument('--season', type=int, required=True, help='Season (e.g., 2025)')
    parser.add_argument('--week', type=int, required=True, help='Week number (e.g., 1)')
    parser.add_argument('--players', type=Path, required=True, 
                       help='Path to PlayerMaster_unified.csv')
    parser.add_argument('--teams', type=Path, required=True,
                       help='Path to team predictions CSV (e.g., predictions_2025w01.csv)')
    parser.add_argument('--outdir', type=Path, default=Path('data/DFSDashboard'),
                       help='Output directory (default: data/DFSDashboard/)')
    return parser.parse_args()

def normalize_team_codes(df, team_col):
    """Normalize team abbreviations using the standard mapping."""
    team_map = {
        'JAC': 'JAX', 'LA': 'LAR', 'STL': 'LAR', 'WSH': 'WAS', 
        'OAK': 'LV', 'SD': 'LAC'
    }
    
    if team_col in df.columns:
        original_values = df[team_col].value_counts()
        df[team_col] = df[team_col].map(lambda x: team_map.get(x, x))
        updated_values = df[team_col].value_counts()
        
        # Log any changes
        for old_team, new_team in team_map.items():
            if old_team in original_values.index:
                print(f"[NORMALIZE_TEAMS] Mapped {old_team} → {new_team} ({original_values[old_team]} players)")
    
    return df

def print_input_columns(players_df, teams_df):
    """Print exact columns for both input files."""
    print("=== INPUT COLUMN ANALYSIS ===")
    print(f"1) PlayerMaster_unified.csv columns ({len(players_df.columns)}):")
    for i, col in enumerate(players_df.columns, 1):
        print(f"   {i:2d}. '{col}'")  # Add quotes to see exact spacing
    
    print(f"\n2) Team predictions columns ({len(teams_df.columns)}):")
    for i, col in enumerate(teams_df.columns, 1):
        print(f"   {i:2d}. '{col}'")
    print()

def create_output_schema():
    """Define the exact output column schema."""
    return [
        'season', 'week', 'slate_id', 'player_id', 'Name', 'TeamAbbrev', 'OppAbbrev', 
        'is_home', 'position', 'salary', 'proj_points',
        'spread_line', 'total_line', 'team_implied_pts', 'opp_implied_pts', 'is_favorite',
        'team_pred_points', 'opp_pred_points', 'home_win_prob_team', 'edge_total', 'edge_spread',
        'shap_notes', 'shap_top_features',
        'status', 'game_status', 'practice_status', 'status_updated_at',
        'targets_l3', 'targets_l5', 'routes_l3', 'routes_l5', 'rush_att_l3', 'rush_att_l5', 
        'snaps_l3', 'snaps_l5',
        'target_share_l3', 'route_share_l3', 'rz_tgts_2024', 'rz_rush_2024',
        'build_timestamp', 'source_files'
    ]

def ensure_output_columns(df, schema):
    """Ensure all output columns exist with appropriate defaults."""
    print(f"\n[ENSURE_COLUMNS] Starting with {len(df.columns)} columns")
    debug_projection_columns(df, "Before Schema Enforcement")
    
    # CRITICAL: Preserve existing proj_points before creating defaults
    has_existing_projections = 'proj_points' in df.columns
    existing_proj_data = None
    if has_existing_projections:
        existing_proj_data = df['proj_points'].copy()
        existing_non_null = existing_proj_data.notna().sum()
        existing_non_zero = (existing_proj_data > 0).sum()
        print(f"[ENSURE_COLUMNS] Preserving existing proj_points: {existing_non_null} non-null, {existing_non_zero} non-zero")
    
    # CRITICAL: Preserve existing historical features before creating defaults
    historical_features = [
        'targets_l3', 'targets_l5', 'routes_l3', 'routes_l5', 'rush_att_l3', 
        'rush_att_l5', 'snaps_l3', 'snaps_l5', 'target_share_l3', 'route_share_l3',
        'rz_tgts_2024', 'rz_rush_2024'
    ]
    existing_hist_data = {}
    for feature in historical_features:
        if feature in df.columns:
            existing_hist_data[feature] = df[feature].copy()
            existing_non_zero = (existing_hist_data[feature] != 0).sum()
            print(f"[ENSURE_COLUMNS] Preserving existing {feature}: {existing_non_zero} non-zero values")
    
    for col in schema:
        if col not in df.columns:
            print(f"[ENSURE_COLUMNS] Creating missing column: {col}")
            
            if col in ['spread_line', 'total_line']:
                # Keep Vegas lines as NaN if truly unknown
                df[col] = np.nan
            elif col == 'proj_points':
                # This should NOT happen if harmonization worked correctly
                print(f"[ENSURE_COLUMNS] WARNING: Creating proj_points from scratch - this suggests harmonization failed!")
                df[col] = 0.0
            elif col in ['season', 'week', 'slate_id', 'player_id', 'salary',
                        'team_implied_pts', 'opp_implied_pts', 'is_favorite', 'team_pred_points', 
                        'opp_pred_points', 'home_win_prob_team', 'edge_total', 'edge_spread']:
                df[col] = 0.0
            elif col in ['targets_l3', 'targets_l5', 'routes_l3', 'routes_l5', 'rush_att_l3', 
                        'rush_att_l5', 'snaps_l3', 'snaps_l5', 'target_share_l3', 'route_share_l3',
                        'rz_tgts_2024', 'rz_rush_2024']:
                # Only set historical features to 0.0 if they don't already exist
                if col not in df.columns:
                    df[col] = 0.0
            else:
                df[col] = ""
    
    # CRITICAL: Restore preserved projection data
    if has_existing_projections and existing_proj_data is not None:
        print(f"[ENSURE_COLUMNS] Restoring preserved proj_points data...")
        df['proj_points'] = existing_proj_data
        restored_non_null = df['proj_points'].notna().sum()
        restored_non_zero = (df['proj_points'] > 0).sum()
        print(f"[ENSURE_COLUMNS] Restored proj_points: {restored_non_null} non-null, {restored_non_zero} non-zero")
    
    # CRITICAL: Restore preserved historical features data
    for feature, feature_data in existing_hist_data.items():
        if feature in df.columns:
            print(f"[ENSURE_COLUMNS] Restoring preserved {feature} data...")
            df[feature] = feature_data
            restored_non_zero = (df[feature] != 0).sum()
            print(f"[ENSURE_COLUMNS] Restored {feature}: {restored_non_zero} non-zero values")

    # Ensure proj_points is float64 to prevent type issues
    if 'proj_points' in df.columns:
        df['proj_points'] = df['proj_points'].astype('float64')
        final_non_null = df['proj_points'].notna().sum()
        final_non_zero = (df['proj_points'] > 0).sum()
        print(f"[ENSURE_COLUMNS] Final proj_points: dtype={df['proj_points'].dtype}, non-null={final_non_null}, non-zero={final_non_zero}")

    # Log columns before reordering
    print(f"[ENSURE_COLUMNS] Reordering to match schema...")
    missing_in_schema = [col for col in df.columns if col not in schema]
    if missing_in_schema:
        print(f"[ENSURE_COLUMNS] WARNING: Columns not in schema will be dropped: {missing_in_schema}")
    
    # Reorder columns to match schema exactly
    df = df[schema]
    
    debug_projection_columns(df, "After Schema Enforcement")
    return df

def compute_implied_points(spread_line, total_line, is_favorite):
    """Compute implied points based on spread and total."""
    if pd.isna(spread_line) or pd.isna(total_line):
        return np.nan, np.nan
    
    # If favorite, team gets more than half the total
    if is_favorite:
        team_implied = total_line/2 + abs(spread_line)/2
        opp_implied = total_line - team_implied
    else:
        opp_implied = total_line/2 + abs(spread_line)/2
        team_implied = total_line - opp_implied
    
    return team_implied, opp_implied

def enrich_players_with_teams(players_df, teams_df, season, week):
    """Enrich player data with team predictions."""
    print(f"\n[ENRICH] Starting enrichment for {season} Week {week}...")
    debug_projection_columns(players_df, "Players Before Enrichment")

    # Ensure team predictions are filtered to the target season/week (prevents cross-week mismatches)
    for col in ["season", "week"]:
        if col not in teams_df.columns:
            raise ValueError(f"Team predictions missing required column: {col}")
    teams_df = teams_df[(teams_df["season"] == season) & (teams_df["week"] == week)].copy()
    print(f"[ENRICH] Filtered to {len(teams_df)} team predictions for {season} Week {week}")
    
    # Filter players to target season/week if those columns exist and have values
    if 'season' in players_df.columns and 'week' in players_df.columns:
        # Check if season/week columns actually have values
        season_has_values = players_df['season'].notna().any()
        week_has_values = players_df['week'].notna().any()
        
        if season_has_values and week_has_values:
            week_mask = (players_df['season'] == season) & (players_df['week'] == week)
            players_filtered = players_df[week_mask].copy()
            print(f"[ENRICH] Filtered to {len(players_filtered)} players for {season} Week {week}")
        else:
            players_filtered = players_df.copy()
            print(f"[ENRICH] WARNING: Season/week columns exist but are empty, using all {len(players_filtered)} players")
    else:
        players_filtered = players_df.copy()
        print(f"[ENRICH] WARNING: No season/week columns found, using all {len(players_filtered)} players")
    
    debug_projection_columns(players_filtered, "Players After Filtering")
    
    # Normalize team codes in both dataframes
    players_filtered = normalize_team_codes(players_filtered, 'TeamAbbrev')
    teams_df = normalize_team_codes(teams_df, 'home_team')
    teams_df = normalize_team_codes(teams_df, 'away_team')
    
    # Check QB coverage specifically
    if 'position' in players_filtered.columns:
        qb_mask = players_filtered['position'].str.upper() == 'QB'
        qb_teams = players_filtered.loc[qb_mask, 'TeamAbbrev'].unique()
        team_pred_teams = set(teams_df['home_team'].tolist() + teams_df['away_team'].tolist())
        print(f"[ENRICH] QB teams in players: {sorted(qb_teams)}")
        print(f"[ENRICH] Teams in predictions: {sorted(team_pred_teams)}")
        missing_qb_teams = set(qb_teams) - team_pred_teams
        if missing_qb_teams:
            print(f"[ENRICH] WARNING: QB teams missing from predictions: {missing_qb_teams}")
    
    # Initialize new columns without overwriting existing data
    existing_cols = players_filtered.columns.tolist()
    new_cols = {
        'OppAbbrev': "",
        'team_pred_points': 0.0,
        'opp_pred_points': 0.0,
        'home_win_prob_team': 0.5,
        'spread_line': np.nan,
        'total_line': np.nan,
        'is_favorite': False
    }
    
    for col, default_val in new_cols.items():
        if col not in existing_cols:
            players_filtered[col] = default_val
        else:
            print(f"[ENRICH] Column '{col}' already exists, preserving existing values")
    
    # Check if is_home column exists
    has_is_home = 'is_home' in players_filtered.columns
    if not has_is_home:
        print("[ENRICH] WARNING: 'is_home' column not found, setting all players as away (is_home=False)")
        players_filtered['is_home'] = False
    
    # Track matched and unmatched players
    matched_mask = np.zeros(len(players_filtered), dtype=bool)
    
    # Match players to team predictions
    for idx, player in players_filtered.iterrows():
        team = player['TeamAbbrev']
        
        # Find matching game (check both home and away)
        home_game = teams_df[teams_df['home_team'] == team]
        away_game = teams_df[teams_df['away_team'] == team]
        
        if not home_game.empty:
            # Player's team is home
            game = home_game.iloc[0]
            players_filtered.loc[idx, 'is_home'] = True
            players_filtered.loc[idx, 'OppAbbrev'] = game['away_team']
            players_filtered.loc[idx, 'team_pred_points'] = game['pred_home_score']
            players_filtered.loc[idx, 'opp_pred_points'] = game['pred_away_score']
            players_filtered.loc[idx, 'home_win_prob_team'] = game['home_win_prob']
            players_filtered.loc[idx, 'spread_line'] = game['spread_line']
            players_filtered.loc[idx, 'total_line'] = game['total_line']
            players_filtered.loc[idx, 'is_favorite'] = game['spread_line'] < 0
            matched_mask[idx] = True
        elif not away_game.empty:
            # Player's team is away
            game = away_game.iloc[0]
            players_filtered.loc[idx, 'is_home'] = False
            players_filtered.loc[idx, 'OppAbbrev'] = game['home_team']
            players_filtered.loc[idx, 'team_pred_points'] = game['pred_away_score']
            players_filtered.loc[idx, 'opp_pred_points'] = game['pred_home_score']
            players_filtered.loc[idx, 'home_win_prob_team'] = 1 - game['home_win_prob']
            players_filtered.loc[idx, 'spread_line'] = -game['spread_line']  # Flip for away team
            players_filtered.loc[idx, 'total_line'] = game['total_line']
            players_filtered.loc[idx, 'is_favorite'] = game['spread_line'] > 0
            matched_mask[idx] = True
    
    # Compute implied points
    for idx, player in players_filtered.iterrows():
        if matched_mask[idx]:
            team_implied, opp_implied = compute_implied_points(
                player['spread_line'], player['total_line'], player['is_favorite']
            )
            players_filtered.loc[idx, 'team_implied_pts'] = team_implied
            players_filtered.loc[idx, 'opp_implied_pts'] = opp_implied
    
    # Add edge calculations
    players_filtered['edge_total'] = players_filtered['team_pred_points'] + players_filtered['opp_pred_points'] - players_filtered['total_line']
    players_filtered['edge_spread'] = players_filtered['team_pred_points'] - players_filtered['opp_pred_points'] - players_filtered['spread_line']
    
    # Add metadata
    players_filtered['build_timestamp'] = datetime.now().isoformat()
    players_filtered['source_files'] = f"PlayerMaster_unified.csv + {teams_df.name if hasattr(teams_df, 'name') else 'team_predictions.csv'}"
    
    debug_projection_columns(players_filtered, "Players After Enrichment")
    return players_filtered, matched_mask

def print_coverage_summary(players_df, matched_mask, season, week):
    """Print coverage statistics with enhanced QB focus."""
    print(f"\n=== COVERAGE SUMMARY: {season} Week {week} ===")
    
    total_players = len(players_df)
    matched_players = matched_mask.sum()
    unmatched_players = total_players - matched_players
    
    print(f"Total players: {total_players}")
    print(f"Matched to team predictions: {matched_players} ({matched_players/total_players*100:.1f}%)")
    print(f"Unmatched players: {unmatched_players} ({unmatched_players/total_players*100:.1f}%)")
    
    # Position breakdown with special focus on QBs
    if 'position' in players_df.columns:
        print(f"\nPosition breakdown:")
        for pos in sorted(players_df['position'].unique()):
            pos_mask = players_df['position'] == pos
            pos_total = pos_mask.sum()
            pos_matched = (pos_mask & matched_mask).sum()
            pos_with_proj = (pos_mask & (players_df['proj_points'] > 0)).sum() if 'proj_points' in players_df.columns else 0
            
            print(f"  {pos}: {pos_total} total, {pos_matched} matched ({pos_matched/pos_total*100:.1f}%), "
                  f"{pos_with_proj} with projections ({pos_with_proj/pos_total*100:.1f}%)")
            
            # Special QB analysis
            if pos.upper() == 'QB':
                qb_df = players_df[pos_mask]
                if 'proj_points' in qb_df.columns:
                    qb_proj_stats = qb_df['proj_points'].describe()
                    print(f"    QB projection stats: min={qb_proj_stats['min']:.1f}, "
                          f"max={qb_proj_stats['max']:.1f}, mean={qb_proj_stats['mean']:.1f}")
                    
                    # Show top QB projections
                    top_qbs = qb_df.nlargest(5, 'proj_points')[['Name', 'TeamAbbrev', 'proj_points']]
                    if len(top_qbs) > 0:
                        print(f"    Top QB projections:")
                        for _, qb in top_qbs.iterrows():
                            print(f"      {qb['Name']} ({qb['TeamAbbrev']}): {qb['proj_points']:.1f}")
    
    # Vegas lines coverage
    if 'spread_line' in players_df.columns and 'total_line' in players_df.columns:
        with_spread = players_df['spread_line'].notna().sum()
        with_total = players_df['total_line'].notna().sum()
        print(f"\nVegas lines coverage:")
        print(f"  spread_line: {with_spread}/{total_players} ({with_spread/total_players*100:.1f}%)")
        print(f"  total_line: {with_total}/{total_players} ({with_total/total_players*100:.1f}%)")
    
    # Enhanced sample focusing on QBs first
    print(f"\nSample rows (QBs first):")
    sample_cols = ['Name', 'position', 'TeamAbbrev', 'proj_points', 'team_pred_points', 'total_line']
    available_cols = [col for col in sample_cols if col in players_df.columns]
    
    if available_cols:
        # Show QBs first
        if 'position' in players_df.columns:
            qb_sample = players_df[players_df['position'].str.upper() == 'QB'][available_cols].head(5)
            if len(qb_sample) > 0:
                print("QB Sample:")
                print(qb_sample.to_string(index=False))
                
            # Then show other positions
            non_qb_sample = players_df[players_df['position'].str.upper() != 'QB'][available_cols].head(5)
            if len(non_qb_sample) > 0:
                print("\nOther Positions Sample:")
                print(non_qb_sample.to_string(index=False))
        else:
            print(players_df[available_cols].head(10).to_string(index=False))
    else:
        print("No sample columns available")

def main():
    """Main execution function."""
    args = parse_args()
    
    # Validate inputs
    if not args.players.exists():
        print(f"ERROR: PlayerMaster file not found: {args.players}")
        return 1
    
    if not args.teams.exists():
        print(f"ERROR: Team predictions file not found: {args.teams}")
        return 1
    
    # Create output directory
    args.outdir.mkdir(parents=True, exist_ok=True)
    
    print(f"=== PlayerMaster v2 Enrichment ===")
    print(f"Season: {args.season}")
    print(f"Week: {args.week}")
    print(f"Players: {args.players}")
    print(f"Teams: {args.teams}")
    print(f"Output: {args.outdir}")
    print()
    
    # Load data
    try:
        print(f"[LOAD] Loading PlayerMaster from: {args.players}")
        players_df = pd.read_csv(args.players)
        print(f"[LOAD] Loaded {len(players_df)} players with {len(players_df.columns)} columns")
        
        print(f"[LOAD] Loading team predictions from: {args.teams}")
        teams_df = pd.read_csv(args.teams)
        print(f"[LOAD] Loaded {len(teams_df)} team predictions with {len(teams_df.columns)} columns")
        
        # Store original filename for source tracking
        teams_df.name = args.teams.name
        
    except Exception as e:
        print(f"ERROR: Failed to load CSV files: {e}")
        return 1
    
    # Print input columns
    print_input_columns(players_df, teams_df)
    
    # Harmonize column names BEFORE enrichment
    players_df = harmonize_player_columns(players_df)
    
    # Enrich players with team data
    enriched_df, matched_mask = enrich_players_with_teams(players_df, teams_df, args.season, args.week)

    # Ensure all output columns exist (and reorder)
    output_schema = create_output_schema()
    enriched_df = ensure_output_columns(enriched_df, output_schema)

    # Write main output
    week_str = f"{args.week:02d}"
    output_path = args.outdir / f"PlayerMaster_v2_{args.season}w{week_str}.csv"
    enriched_df.to_csv(output_path, index=False)
    print(f"\n✅ Wrote enriched PlayerMaster: {output_path}")
    
    # Write unmatched players
    unmatched_df = enriched_df[~matched_mask]
    if len(unmatched_df) > 0:
        unmatched_path = args.outdir / f"PlayerMaster_v2_{args.season}w{week_str}_unmatched_players.csv"
        unmatched_df.to_csv(unmatched_path, index=False)
        print(f"✅ Wrote unmatched players: {unmatched_path}")
    
    # Print coverage summary
    print_coverage_summary(enriched_df, matched_mask, args.season, args.week)
    
    # Final QB projection verification
    if 'position' in enriched_df.columns and 'proj_points' in enriched_df.columns:
        qb_with_proj = enriched_df[(enriched_df['position'].str.upper() == 'QB') & (enriched_df['proj_points'] > 0)]
        print(f"\n🏈 FINAL QB VERIFICATION: {len(qb_with_proj)} QBs with projections > 0")
        
        if len(qb_with_proj) == 0:
            print("⚠️  WARNING: No QBs found with projections > 0!")
            # Show all QBs for debugging
            all_qbs = enriched_df[enriched_df['position'].str.upper() == 'QB']
            if len(all_qbs) > 0:
                print("All QBs in final output:")
                qb_debug_cols = ['Name', 'TeamAbbrev', 'proj_points'] + \
                              [col for col in all_qbs.columns if 'proj' in col.lower() or 'point' in col.lower()]
                qb_debug_cols = [col for col in qb_debug_cols if col in all_qbs.columns]
                print(all_qbs[qb_debug_cols].to_string(index=False))
        else:
            print("✅ QB projections successfully preserved!")
    
    return 0

if __name__ == "__main__":
    exit(main())