#!/usr/bin/env python3
"""
PlayerMaster from Projections - NFL DFS Script

Scans repository for position projection CSVs (QB, RB, WR, TE, DST),
normalizes them, and merges onto the master sheet to create a unified
PlayerMaster file for the optimizer.

Usage:
    python playermaster_from_projections.py [OPTIONS]

Default behavior:
    - Master: data/processed/master_sheet_2025.csv
    - Output: data/DFSDashboard/PlayerMaster_unified.csv
    - Scan: outputs/projections/, models/, data/processed/
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import re


def setup_logging(log_file: Optional[Path] = None) -> logging.Logger:
    """Setup logging configuration."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def norm_name(name: str) -> str:
    """Normalize player name: lowercase, strip punctuation and suffixes."""
    if pd.isna(name):
        return ""
    
    # Convert to string and lowercase
    name_str = str(name).lower().strip()
    
    # Remove common suffixes
    suffixes = ['jr', 'sr', 'ii', 'iii', 'iv', 'v']
    for suffix in suffixes:
        if name_str.endswith(f' {suffix}'):
            name_str = name_str[:-len(suffix)].strip()
    
    # Remove punctuation and extra spaces
    name_str = re.sub(r'[^\w\s]', '', name_str)
    name_str = re.sub(r'\s+', ' ', name_str).strip()
    
    return name_str


def norm_team(team: str) -> str:
    """Normalize team abbreviation with mapping."""
    if pd.isna(team):
        return ""
    
    team_str = str(team).upper().strip()
    
    # Team mapping
    team_map = {
        'JAC': 'JAX',
        'LA': 'LAR', 
        'STL': 'LAR',
        'WSH': 'WAS',
        'OAK': 'LV',
        'SD': 'LAC'
    }
    
    return team_map.get(team_str, team_str)


def pick_col(df: pd.DataFrame, candidates: List[str], fallback_candidates: Optional[List[str]] = None) -> Optional[str]:
    """
    Pick the best column from candidates, with case-insensitive matching.
    
    Args:
        df: DataFrame to search
        candidates: List of column names to try (case-insensitive)
        fallback_candidates: Optional fallback list if no candidates found
    
    Returns:
        Column name if found, None otherwise
    """
    df_cols = [col.lower() for col in df.columns]
    
    # Try exact matches first
    for candidate in candidates:
        if candidate.lower() in df_cols:
            # Find the actual column name with original case
            for col in df.columns:
                if col.lower() == candidate.lower():
                    return col
    
    # Try fallback candidates if provided
    if fallback_candidates:
        for candidate in fallback_candidates:
            if candidate.lower() in df_cols:
                for col in df.columns:
                    if col.lower() == candidate.lower():
                        return col
    
    return None


def detect_projection_column(df: pd.DataFrame, forced_col: Optional[str] = None) -> Optional[str]:
    """
    Detect the projection column in a DataFrame.
    
    Args:
        df: DataFrame to analyze
        forced_col: If provided, use this column name
    
    Returns:
        Column name if found, None otherwise
    """
    if forced_col:
        if forced_col in df.columns:
            return forced_col
        else:
            return None
    
    # Primary candidates for projection columns
    primary_candidates = [
        'predicted_points', 'proj_points', 'projection', 
        'predicted_points_ensemble', 'points', 'proj', 'proj_ppr', 'fp', 'fpts'
    ]
    
    # Try primary candidates
    proj_col = pick_col(df, primary_candidates)
    if proj_col:
        return proj_col
    
    # Fallback: find first numeric column that's not clearly salary/player_id/week/season
    exclude_patterns = ['salary', 'player_id', 'id', 'week', 'season', 'dk_salary']
    
    for col in df.columns:
        if col.lower() in exclude_patterns:
            continue
        
        # Check if column is numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            # Check if it looks like projections (reasonable range)
            values = df[col].dropna()
            if len(values) > 0:
                if 0 <= values.min() <= values.max() <= 50:  # Reasonable fantasy point range
                    return col
    
    return None


def is_projection_file(df: pd.DataFrame, min_rows: int = 20) -> bool:
    """
    Determine if a CSV file contains per-player projections.
    
    Args:
        df: DataFrame to analyze
        min_rows: Minimum rows to consider valid
    
    Returns:
        True if this appears to be a projection file
    """
    if len(df) < min_rows:
        return False
    
    # Check if it has essential columns for projections
    required_cols = ['position', 'Name', 'player_name']
    has_position = any(col.lower() in ['position', 'pos'] for col in df.columns)
    has_name = any(col.lower() in ['name', 'player_name'] for col in df.columns)
    
    return has_position and has_name


def normalize_master_sheet(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Normalize the master sheet DataFrame.
    
    Args:
        df: Raw master sheet DataFrame
        logger: Logger instance
    
    Returns:
        Normalized DataFrame with expected columns
    """
    logger.info(f"[MASTER] {len(df)} rows, columns: {list(df.columns)}")
    
    # Detect and map columns
    name_col = pick_col(df, ['Name', 'player_name'])
    team_col = pick_col(df, ['TeamAbbrev', 'team', 'recent_team'])
    pos_col = pick_col(df, ['position', 'pos'])
    salary_col = pick_col(df, ['salary', 'dk_salary'])
    player_id_col = pick_col(df, ['player_id', 'gsis_id', 'id'])
    
    # Validate required columns
    missing_cols = []
    if not name_col:
        missing_cols.append('Name/player_name')
    if not team_col:
        missing_cols.append('TeamAbbrev/team')
    if not pos_col:
        missing_cols.append('position/pos')
    if not salary_col:
        missing_cols.append('salary/dk_salary')
    if not player_id_col:
        missing_cols.append('player_id/gsis_id/id')
    
    if missing_cols:
        raise ValueError(f"Master sheet missing required columns: {missing_cols}")
    
    # Create normalized DataFrame
    result = df.copy()
    result['Name'] = df[name_col]
    result['TeamAbbrev'] = df[team_col]
    result['position'] = df[pos_col].str.upper()
    result['salary'] = pd.to_numeric(df[salary_col], errors='coerce')
    result['player_id'] = df[player_id_col].astype(str)
    
    # Create helper keys
    result['Name_norm'] = result['Name'].apply(norm_name)
    result['Team_norm'] = result['TeamAbbrev'].apply(norm_team)
    result['join_key'] = result['Name_norm'] + '|' + result['Team_norm'] + '|' + result['position']
    
    # Optional passthrough columns
    optional_cols = ['season', 'week', 'slate_id', 'spread_line', 'total_line']
    for col in optional_cols:
        if col in df.columns:
            result[col] = df[col]
    
    return result


def normalize_projection_file(df: pd.DataFrame, file_path: Path, forced_proj_col: Optional[str], logger: logging.Logger) -> Optional[pd.DataFrame]:
    """
    Normalize a projection file DataFrame.
    
    Args:
        df: Raw projection DataFrame
        file_path: Path to the file for logging
        forced_proj_col: If provided, use this projection column
        logger: Logger instance
    
    Returns:
        Normalized DataFrame with [player_id, join_key, proj_points] or None if invalid
    """
    logger.info(f"Processing: {file_path.name}")
    logger.info(f"  Columns: {list(df.columns)}")
    
    # Detect columns
    name_col = pick_col(df, ['Name', 'player_name'])
    team_col = pick_col(df, ['TeamAbbrev', 'team', 'recent_team'])
    pos_col = pick_col(df, ['position', 'pos'])
    player_id_col = pick_col(df, ['player_id', 'gsis_id', 'id'])
    
    # Detect projection column
    proj_col = detect_projection_column(df, forced_proj_col)
    if not proj_col:
        logger.warning(f"  No projection column found, skipping")
        return None
    
    logger.info(f"  Chosen projection column: {proj_col}")
    
    # Validate required columns
    missing_cols = []
    if not name_col:
        missing_cols.append('Name/player_name')
    if not team_col:
        missing_cols.append('TeamAbbrev/team')
    if not pos_col:
        missing_cols.append('position/pos')
    
    if missing_cols:
        logger.warning(f"  Missing required columns: {missing_cols}, skipping")
        return None
    
    # Detect SHAP notes column
    shap_col = pick_col(df, ['shap_rationale', 'shap_notes', 'rationale'])
    
    # Detect season and week columns for filtering
    season_col = pick_col(df, ['season'])
    week_col = pick_col(df, ['week'])
    
    # Filter for current week (2025 week 1) if season/week columns exist
    if season_col and week_col:
        logger.info(f"  Found season/week columns: {season_col}/{week_col}")
        current_week_mask = (df[season_col] == 2025) & (df[week_col] == 1)
        if current_week_mask.any():
            df = df[current_week_mask].copy()
            logger.info(f"  Filtered to {len(df)} rows for 2025 week 1")
        else:
            logger.warning(f"  No 2025 week 1 data found, using all data")
    else:
        logger.info(f"  No season/week columns found, using all data")
    
    # Create normalized DataFrame
    result = pd.DataFrame()
    result['player_id'] = df[player_id_col].astype(str) if player_id_col else None
    result['Name'] = df[name_col]
    result['TeamAbbrev'] = df[team_col]
    result['position'] = df[pos_col].str.upper()
    result['proj_points'] = pd.to_numeric(df[proj_col], errors='coerce')
    
    # Add SHAP notes if available
    if shap_col:
        result['shap_notes'] = df[shap_col]
        logger.info(f"  Found SHAP notes column: {shap_col}")
    else:
        result['shap_notes'] = ''
        logger.info(f"  No SHAP notes column found")
    
    # Add historical features if available
    historical_features = [
        'targets_l3', 'targets_l5', 'routes_l3', 'routes_l5', 
        'rush_att_l3', 'rush_att_l5', 'snaps_l3', 'snaps_l5',
        'target_share_l3', 'route_share_l3', 'rz_tgts_2024', 'rz_rush_2024'
    ]
    
    for feature in historical_features:
        if feature in df.columns:
            result[feature] = pd.to_numeric(df[feature], errors='coerce').fillna(0.0)
            logger.info(f"  Found historical feature: {feature}")
        else:
            result[feature] = 0.0
            logger.info(f"  Historical feature not found: {feature}, setting to 0.0")
    
    # FIX: Handle NaN teams by using player_id to look up team from master sheet
    nan_team_count = result['TeamAbbrev'].isna().sum()
    if nan_team_count > 0:
        logger.info(f"  Found {nan_team_count} rows with NaN teams, attempting to fill from master sheet")
        
        # Load master sheet to get team info
        try:
            master_path = Path('../../data/processed/master_sheet_2025.csv')
            if master_path.exists():
                master_df = pd.read_csv(master_path)
                logger.info(f"  Loaded master sheet with {len(master_df)} rows")
                
                # Create player_id to team mapping
                team_mapping = master_df.set_index('player_id')['TeamAbbrev'].to_dict()
                
                # Fill NaN teams using player_id lookup
                filled_count = 0
                for idx, row in result.iterrows():
                    if pd.isna(row['TeamAbbrev']) and row['player_id'] in team_mapping:
                        result.loc[idx, 'TeamAbbrev'] = team_mapping[row['player_id']]
                        filled_count += 1
                
                logger.info(f"  Filled {filled_count} NaN teams from master sheet")
                
                # Check remaining NaN teams
                remaining_nan = result['TeamAbbrev'].isna().sum()
                if remaining_nan > 0:
                    logger.warning(f"  {remaining_nan} rows still have NaN teams after lookup")
            else:
                logger.warning(f"  Master sheet not found at {master_path}, cannot fill NaN teams")
        except Exception as e:
            logger.warning(f"  Error loading master sheet: {e}")
    
    # Special handling for DST projections
    if 'DST' in result['position'].values:
        logger.info(f"  DST file detected - consolidating multiple opponent rows")
        
        # Group by team and take the highest projection for each team
        dst_consolidated = result[result['position'] == 'DST'].copy()
        if len(dst_consolidated) > 0:
            # Group by team and take max projection, keeping SHAP notes from the max projection row
            if 'shap_notes' in dst_consolidated.columns:
                # Find the row with max projection for each team to keep its SHAP notes
                max_idx = dst_consolidated.groupby('TeamAbbrev')['proj_points'].idxmax()
                dst_consolidated = dst_consolidated.loc[max_idx].copy()
            else:
                # Group by team and take max projection
                dst_consolidated = dst_consolidated.groupby('TeamAbbrev').agg({
                    'proj_points': 'max',
                    'Name': 'first',
                    'position': 'first'
                }).reset_index()
            
            # Transform DST player_id format: DEN_DST -> DEN-DEN to match master sheet
            dst_consolidated['player_id'] = dst_consolidated['TeamAbbrev'] + '-' + dst_consolidated['TeamAbbrev']
            
            # Update the result DataFrame
            result = result[result['position'] != 'DST']  # Remove original DST rows
            result = pd.concat([result, dst_consolidated], ignore_index=True)  # Add consolidated DST rows
            
            logger.info(f"  Consolidated {len(dst_consolidated)} DST teams")
    
    # Create helper keys
    result['Name_norm'] = result['Name'].apply(norm_name)
    result['Team_norm'] = result['TeamAbbrev'].apply(norm_team)
    result['join_key'] = result['Name_norm'] + '|' + result['Team_norm'] + '|' + result['position']
    
    # Remove rows with missing data
    result = result.dropna(subset=['Name', 'TeamAbbrev', 'position', 'proj_points'])
    
    logger.info(f"  {len(result)} valid projection rows")
    
    return result


def scan_for_projection_files(scan_dirs: List[Path], min_rows: int = 20) -> List[Path]:
    """
    Scan directories for projection CSV files.
    
    Args:
        scan_dirs: List of directories to scan
        min_rows: Minimum rows to consider valid
    
    Returns:
        List of paths to projection files
    """
    projection_files = []
    
    for scan_dir in scan_dirs:
        if not scan_dir.exists():
            continue
            
        for csv_file in scan_dir.rglob("*.csv"):
            try:
                # Quick check without loading full file
                df_sample = pd.read_csv(csv_file, nrows=min_rows + 10)
                if is_projection_file(df_sample, min_rows):
                    projection_files.append(csv_file)
            except Exception as e:
                # Skip files that can't be read
                continue
    
    return projection_files


def merge_projections(master_df: pd.DataFrame, projection_dfs: List[pd.DataFrame], logger: logging.Logger) -> pd.DataFrame:
    """
    Merge projections onto master sheet.
    
    Args:
        master_df: Normalized master sheet
        projection_dfs: List of normalized projection DataFrames
        logger: Logger instance
    
    Returns:
        Master sheet with merged projections
    """
    if not projection_dfs:
        logger.warning("No projection files to merge")
        master_df['proj_points'] = 0.0
        return master_df
    
    # Combine all projections
    all_projections = pd.concat(projection_dfs, ignore_index=True)
    
    # Remove duplicates by taking max projection for each player, keeping SHAP notes from max projection row
    if len(all_projections) > 0:
        duplicates = all_projections.duplicated(subset=['player_id', 'join_key'], keep=False)
        if duplicates.any():
            logger.info(f"[PROJ] Collapsing {duplicates.sum()} duplicate projections")
            if 'shap_notes' in all_projections.columns:
                # Keep SHAP notes from the row with max projection
                max_idx = all_projections.groupby(['player_id', 'join_key'])['proj_points'].idxmax()
                all_projections = all_projections.loc[max_idx].reset_index(drop=True)
            else:
                all_projections = all_projections.groupby(['player_id', 'join_key'])['proj_points'].max().reset_index()
    
    logger.info(f"[PROJ] Total loaded rows: {len(all_projections)} from {len(projection_dfs)} files")
    
    # Create lookups for projections, SHAP notes, and historical features
    proj_lookup = {}
    shap_lookup = {}
    
    # Historical features to preserve
    historical_features = [
        'targets_l3', 'targets_l5', 'routes_l3', 'routes_l5', 
        'rush_att_l3', 'rush_att_l5', 'snaps_l3', 'snaps_l5',
        'target_share_l3', 'route_share_l3', 'rz_tgts_2024', 'rz_rush_2024'
    ]
    
    # Create lookups for historical features
    hist_lookups = {feature: {} for feature in historical_features}
    
    # First pass: use player_id if available
    for _, row in all_projections.iterrows():
        player_id = row['player_id']
        if player_id and pd.notna(player_id):
            if player_id not in proj_lookup:
                proj_lookup[player_id] = row['proj_points']
                if 'shap_notes' in row and pd.notna(row['shap_notes']):
                    shap_lookup[player_id] = row['shap_notes']
                # Add historical features
                for feature in historical_features:
                    if feature in row and pd.notna(row[feature]) and row[feature] != 0:
                        hist_lookups[feature][player_id] = row[feature]
            else:
                # Take the higher projection if multiple exist
                if row['proj_points'] > proj_lookup[player_id]:
                    proj_lookup[player_id] = row['proj_points']
                    if 'shap_notes' in row and pd.notna(row['shap_notes']):
                        shap_lookup[player_id] = row['shap_notes']
                    # Update historical features for the higher projection
                    for feature in historical_features:
                        if feature in row and pd.notna(row[feature]) and row[feature] != 0:
                            hist_lookups[feature][player_id] = row[feature]
    
    # Second pass: fill in missing player_ids with join_key
    for _, row in all_projections.iterrows():
        player_id = row['player_id']
        join_key = row['join_key']
        proj_value = row['proj_points']
        
        # If no player_id or already processed, try join_key
        if not player_id or pd.isna(player_id) or player_id in proj_lookup:
            if join_key not in proj_lookup:
                proj_lookup[join_key] = proj_value
                if 'shap_notes' in row and pd.notna(row['shap_notes']):
                    shap_lookup[join_key] = row['shap_notes']
                # Add historical features
                for feature in historical_features:
                    if feature in row and pd.notna(row[feature]) and row[feature] != 0:
                        hist_lookups[feature][join_key] = row[feature]
            else:
                # Take the higher projection if multiple exist
                if proj_value > proj_lookup[join_key]:
                    proj_lookup[join_key] = proj_value
                    if 'shap_notes' in row and pd.notna(row['shap_notes']):
                        shap_lookup[join_key] = row['shap_notes']
                    # Update historical features for the higher projection
                    for feature in historical_features:
                        if feature in row and pd.notna(row[feature]) and row[feature] != 0:
                            hist_lookups[feature][join_key] = row[feature]
    
    # Apply projections to master sheet
    master_df['proj_points'] = 0.0  # Initialize with 0
    master_df['shap_notes'] = ''  # Initialize SHAP notes
    
    # Initialize historical features with 0.0
    for feature in historical_features:
        master_df[feature] = 0.0
    
    # First try player_id match
    player_id_mask = master_df['player_id'].isin(proj_lookup.keys())
    master_df.loc[player_id_mask, 'proj_points'] = master_df.loc[player_id_mask, 'player_id'].map(proj_lookup)
    # Add SHAP notes for player_id matches
    player_id_shap_mask = master_df['player_id'].isin(shap_lookup.keys())
    master_df.loc[player_id_shap_mask, 'shap_notes'] = master_df.loc[player_id_shap_mask, 'player_id'].map(shap_lookup)
    # Add historical features for player_id matches
    for feature in historical_features:
        feature_mask = master_df['player_id'].isin(hist_lookups[feature].keys())
        master_df.loc[feature_mask, feature] = master_df.loc[feature_mask, 'player_id'].map(hist_lookups[feature])
    
    # Then try join_key match for remaining players
    remaining_mask = master_df['proj_points'] == 0
    if remaining_mask.any():
        join_key_lookup = {k: v for k, v in proj_lookup.items() if '|' in k}  # Only join_keys have '|'
        join_key_mask = master_df['join_key'].isin(join_key_lookup.keys())
        master_df.loc[join_key_mask & remaining_mask, 'proj_points'] = master_df.loc[join_key_mask & remaining_mask, 'join_key'].map(join_key_lookup)
        # Add SHAP notes for join_key matches
        join_key_shap_mask = master_df['join_key'].isin(shap_lookup.keys())
        master_df.loc[join_key_shap_mask & remaining_mask, 'shap_notes'] = master_df.loc[join_key_shap_mask & remaining_mask, 'join_key'].map(shap_lookup)
        # Add historical features for join_key matches
        for feature in historical_features:
            feature_mask = master_df['join_key'].isin(hist_lookups[feature].keys())
            master_df.loc[feature_mask & remaining_mask, feature] = master_df.loc[feature_mask & remaining_mask, 'join_key'].map(hist_lookups[feature])
    
    # Calculate match statistics
    matched_count = (master_df['proj_points'] > 0).sum()
    total_players = len(master_df)
    match_percentage = (matched_count / total_players) * 100 if total_players > 0 else 0
    
    logger.info(f"[RESULT] {matched_count}/{total_players} players matched ({match_percentage:.1f}%)")
    
    return master_df



def create_optimizer_view(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Create the final optimizer view with required columns.
    
    Args:
        df: Master sheet with merged projections
    
    Returns:
        DataFrame with optimizer schema
    """
    # Required columns in exact order
    required_cols = [
        'season', 'week', 'slate_id', 'player_id', 'Name', 'TeamAbbrev', 'position', 'salary', 'proj_points', 
        'spread_line', 'total_line', 'shap_notes',
        'targets_l3', 'targets_l5', 'routes_l3', 'routes_l5', 'rush_att_l3', 'rush_att_l5', 
        'snaps_l3', 'snaps_l5', 'target_share_l3', 'route_share_l3', 'rz_tgts_2024', 'rz_rush_2024'
    ]
    
    # Create result DataFrame with all required columns
    result = df.copy()
    
    # Add missing columns with safe defaults
    for col in required_cols:
        if col not in result.columns:
            if col in ['season', 'week', 'slate_id', 'spread_line', 'total_line', 'shap_notes']:
                result[col] = ''  # Empty string for string columns
            else:
                result[col] = 0.0  # 0.0 for numeric columns
    
    # Ensure exact column order
    result = result[required_cols]
    
    # Ensure data types
    result['player_id'] = result['player_id'].astype(str)
    result['position'] = result['position'].astype(str)
    result['salary'] = pd.to_numeric(result['salary'], errors='coerce')
    result['proj_points'] = pd.to_numeric(result['proj_points'], errors='coerce')
    
    return result


def print_top_players(df: pd.DataFrame, logger: logging.Logger, top_n: int = 10):
    """Print top N players by projection points."""
    if 'proj_points' not in df.columns:
        return
    
    top_players = df.nlargest(top_n, 'proj_points')
    
    logger.info(f"\nTop {top_n} players by projection:")
    logger.info("Name | Position | Team | Salary | Proj Points")
    logger.info("-" * 50)
    
    for _, player in top_players.iterrows():
        name = player.get('Name', 'Unknown')
        position = player.get('position', 'Unknown')
        team = player.get('TeamAbbrev', 'Unknown')
        salary = player.get('salary', 0)
        proj = player.get('proj_points', 0)
        
        logger.info(f"{name} | {position} | {team} | {salary:,.0f} | {proj:.2f}")


def print_position_coverage(df: pd.DataFrame, logger: logging.Logger):
    """Print per-position coverage statistics."""
    if 'proj_points' not in df.columns or 'position' not in df.columns:
        return
    
    logger.info(f"\nPosition Coverage Summary:")
    logger.info("Position | Total | With Proj | Coverage %")
    logger.info("-" * 40)
    
    for position in ['QB', 'RB', 'WR', 'TE', 'DST']:
        pos_mask = df['position'] == position
        total_rows = pos_mask.sum()
        
        if total_rows > 0:
            with_proj = (df.loc[pos_mask, 'proj_points'] > 0).sum()
            coverage_pct = (with_proj / total_rows) * 100
            logger.info(f"{position:8} | {total_rows:5} | {with_proj:9} | {coverage_pct:8.1f}%")
        else:
            logger.info(f"{position:8} | {total_rows:5} | {0:9} | {0.0:8.1f}%")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Create unified PlayerMaster from projection files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--master', 
        type=Path, 
        default=Path('data/processed/master_sheet_2025.csv'),
        help='Path to master sheet CSV (default: data/processed/master_sheet_2025.csv)'
    )
    
    parser.add_argument(
        '--proj-dir', 
        type=Path,
        help='Custom directory to scan for projection files (overrides defaults)'
    )
    
    parser.add_argument(
        '--out', 
        type=Path, 
        default=Path('data/DFSDashboard/PlayerMaster_unified.csv'),
        help='Output path (default: data/DFSDashboard/PlayerMaster_unified.csv)'
    )
    
    parser.add_argument(
        '--proj-col', 
        type=str,
        help='Force projection column name for all files'
    )
    
    parser.add_argument(
        '--min-rows', 
        type=int, 
        default=20,
        help='Minimum rows to consider valid projection file (default: 20)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = args.out.parent / f"{args.out.stem}.log"
    logger = setup_logging(log_file)
    
    try:
        # Validate master file
        if not args.master.exists():
            logger.error(f"Master file not found: {args.master}")
            sys.exit(1)
        
        # Load current slate to get eligible players
        current_slate_path = 'data/processed/current_slate.csv'
        if not os.path.exists(current_slate_path):
            raise FileNotFoundError(f"Current slate not found at {current_slate_path}. Run build_master_sheet.py first!")
        
        current_slate = pd.read_csv(current_slate_path)
        eligible_players = set(current_slate['Name'].unique())
        logger.info(f"Loaded current slate with {len(eligible_players)} eligible players")
        
        # Load master sheet
        logger.info(f"Loading master sheet: {args.master}")
        master_df = pd.read_csv(args.master)
        master_df = normalize_master_sheet(master_df, logger)
        
        # Before creating PlayerMaster, filter to only current slate players
        if 'Name' in master_df.columns:
            before_count = len(master_df)
            master_df = master_df[master_df['Name'].isin(eligible_players)]
            after_count = len(master_df)
            logger.info(f"Filtered master sheet from {before_count} to {after_count} players (current slate only)")
        
        # Determine scan directories
        if args.proj_dir:
            scan_dirs = [args.proj_dir]
        else:
            # Expanded defaults: include PositionModel tree so we pick up per-position predictions
            # Use relative paths from script location (data/DFSDashboard)
            scan_dirs = [
                Path('outputs/projections'),
                Path('../../PositionModel'),
                Path('../../PositionModel/QB'),
                Path('../../PositionModel/RB'),
                Path('../../PositionModel/WR'),
                Path('../../PositionModel/TE'),
                Path('../../PositionModel/DST'),
                Path('../../models'),
                Path('../../data/processed'),
            ]
        
        # Scan for projection files
        logger.info("Scanning for projection files...")
        projection_files = scan_for_projection_files(scan_dirs, args.min_rows)
        
        if not projection_files:
            logger.warning("No projection files found")
            sys.exit(2)
        
        logger.info(f"Found {len(projection_files)} projection files:")
        for p in projection_files:
            logger.info(f"  - {p}")
        
        # Load and normalize projection files
        projection_dfs = []
        for file_path in projection_files:
            try:
                df = pd.read_csv(file_path)
                normalized_df = normalize_projection_file(df, file_path, args.proj_col, logger)
                if normalized_df is not None:
                    projection_dfs.append(normalized_df)
            except Exception as e:
                logger.warning(f"Error processing {file_path}: {e}")
                continue
        
        # Merge projections
        master_df = merge_projections(master_df, projection_dfs, logger)
        
        # Debug: Check if historical features are in master_df after merge
        logger.info(f"[DEBUG] After merge, master_df columns: {list(master_df.columns)}")
        if 'targets_l3' in master_df.columns:
            rb_mask = master_df['position'] == 'RB'
            non_zero_targets = (master_df.loc[rb_mask, 'targets_l3'] != 0).sum()
            logger.info(f"[DEBUG] RB players with non-zero targets_l3: {non_zero_targets}")
        else:
            logger.info("[DEBUG] targets_l3 not found in master_df after merge")
        
        # Create optimizer view
        optimizer_df = create_optimizer_view(master_df, logger)
        
        # Final validation - no players outside current slate
        pm_players = set(optimizer_df['Name'].unique())
        invalid_players = pm_players - eligible_players
        if invalid_players:
            logger.error(f"ERROR: {len(invalid_players)} players in PlayerMaster not in current slate!")
            logger.error(f"Examples: {list(invalid_players)[:5]}")
            optimizer_df = optimizer_df[~optimizer_df['Name'].isin(invalid_players)]
            logger.info(f"Removed invalid players, final count: {len(optimizer_df)}")
        
        # Print top players
        print_top_players(optimizer_df, logger)
        
        # Print position coverage
        print_position_coverage(optimizer_df, logger)
        
        # Save with slate info in filename
        slate_info = f"_{len(eligible_players)}players"
        output_path = args.out.parent / f"{args.out.stem}{slate_info}.csv"
        logger.info(f"[WRITE] Writing main output to: {output_path}")
        optimizer_df.to_csv(output_path, index=False)
        
        # Also save as "current" for easy reference
        current_path = args.out.parent / "PlayerMaster_current.csv"
        logger.info(f"[WRITE] Writing current reference to: {current_path}")
        optimizer_df.to_csv(current_path, index=False)
        
        # Create unmatched CSV (players with proj_points == 0)
        unmatched_df = optimizer_df[optimizer_df['proj_points'] == 0].copy()
        unmatched_path = args.out.parent / f"{args.out.stem}{slate_info}_unmatched.csv"
        logger.info(f"[WRITE] Writing unmatched players to: {unmatched_path}")
        unmatched_df.to_csv(unmatched_path, index=False)
        
        logger.info(f"Success! Files created:")
        logger.info(f"  Main: {output_path}")
        logger.info(f"  Current: {current_path}")
        logger.info(f"  Unmatched: {unmatched_path}")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
