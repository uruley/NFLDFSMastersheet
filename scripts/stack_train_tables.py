#!/usr/bin/env python3
"""
Stack training tables from multiple seasons into one consolidated file.
Reads all train_table_*.csv files from season directories and combines them.
"""

import argparse
import pandas as pd
import os
import glob
from pathlib import Path


def stack_train_tables(tables_pattern, output_file):
    """Stack all training tables matching the pattern"""
    print(f"Searching for training tables matching pattern: {tables_pattern}")
    
    # Find all matching files
    table_files = glob.glob(tables_pattern)
    
    if not table_files:
        raise FileNotFoundError(f"No files found matching pattern: {tables_pattern}")
    
    print(f"Found {len(table_files)} training table files:")
    for file in table_files:
        print(f"  {file}")
    
    # Read and stack all tables
    all_tables = []
    season_counts = {}
    
    for file_path in table_files:
        # Parse season from parent folder name
        path_parts = Path(file_path).parts
        season = None
        for part in path_parts:
            if part.isdigit() and len(part) == 4:  # Look for 4-digit year
                season = part
                break
        
        if not season:
            print(f"Warning: Could not parse season from path: {file_path}")
            season = "unknown"
        
        # Read the CSV
        print(f"Reading {file_path} (season: {season})...")
        df = pd.read_csv(file_path)
        
        # Add season column
        df['season'] = season
        
        # Track row count per season
        if season not in season_counts:
            season_counts[season] = 0
        season_counts[season] += len(df)
        
        all_tables.append(df)
        print(f"  Loaded {len(df)} rows")
    
    # Combine all tables using outer join on columns
    print(f"\nCombining {len(all_tables)} tables...")
    
    if len(all_tables) == 1:
        combined_df = all_tables[0]
    else:
        # Start with first table
        combined_df = all_tables[0]
        
        # Union with remaining tables, filling missing columns with NaN
        for i, df in enumerate(all_tables[1:], 1):
            print(f"  Unioning table {i+1}/{len(all_tables)}...")
            
            # Find columns that exist in current table but not in combined
            new_cols = [col for col in df.columns if col not in combined_df.columns]
            if new_cols:
                print(f"    Adding new columns: {new_cols}")
                for col in new_cols:
                    combined_df[col] = pd.NA
            
            # Find columns that exist in combined but not in current table
            missing_cols = [col for col in combined_df.columns if col not in df.columns]
            if missing_cols:
                print(f"    Filling missing columns: {missing_cols}")
                for col in missing_cols:
                    df[col] = pd.NA
            
            # Ensure both dataframes have same column order
            all_cols = list(set(combined_df.columns) | set(df.columns))
            combined_df = combined_df.reindex(columns=all_cols)
            df = df.reindex(columns=all_cols)
            
            # Concatenate
            combined_df = pd.concat([combined_df, df], ignore_index=True)
    
    print(f"Combined table: {len(combined_df)} total rows")
    
    # Print summary by season
    print(f"\n=== ROWS PER SEASON ===")
    season_summary = combined_df.groupby('season').size().sort_index()
    for season, count in season_summary.items():
        print(f"  {season}: {count:,} rows")
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Write combined table
    print(f"\nWriting combined table to: {output_file}")
    combined_df.to_csv(output_file, index=False)
    print(f"Successfully wrote {len(combined_df)} rows to {output_file}")
    
    return combined_df


def main():
    parser = argparse.ArgumentParser(description='Stack training tables from multiple seasons')
    parser.add_argument('--tables', required=True, help='Glob pattern for training table files')
    parser.add_argument('--out', required=True, help='Output path for combined training table')
    
    args = parser.parse_args()
    
    try:
        combined_df = stack_train_tables(args.tables, args.out)
        print(f"\n✅ Successfully stacked {len(combined_df)} total rows from all seasons")
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())

