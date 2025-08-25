#!/usr/bin/env python3
"""
Build training table by combining master sheet data with labels.
Loads all master sheet files from a directory and joins with labels data.
"""

import argparse
import pandas as pd
import os
import glob
from pathlib import Path


def load_master_sheets(masters_dir):
    """Load and stack all master sheet files from directory"""
    print(f"Loading master sheets from: {masters_dir}")
    
    # Find all master sheet files
    pattern = os.path.join(masters_dir, "master_sheet_*_*.csv")
    master_files = glob.glob(pattern)
    
    if not master_files:
        raise FileNotFoundError(f"No master sheet files found matching pattern: {pattern}")
    
    print(f"Found {len(master_files)} master sheet files:")
    for file in master_files:
        print(f"  {os.path.basename(file)}")
    
    # Load and stack all files
    master_dfs = []
    for file in master_files:
        df = pd.read_csv(file)
        master_dfs.append(df)
        print(f"  Loaded {len(df)} rows from {os.path.basename(file)}")
    
    # Vertical stack all dataframes
    combined_df = pd.concat(master_dfs, ignore_index=True)
    print(f"Combined master sheets: {len(combined_df)} total rows")
    
    return combined_df


def load_labels(labels_file=None, labels_dir=None):
    """Load labels data from file, directory, or both"""
    labels_dfs = []
    
    if labels_file:
        print(f"Loading labels from file: {labels_file}")
        file_df = pd.read_csv(labels_file)
        labels_dfs.append(file_df)
        print(f"  File labels: {len(file_df)} rows")
    
    if labels_dir:
        print(f"Loading labels from directory: {labels_dir}")
        # Find all CSV files in directory
        pattern = os.path.join(labels_dir, "*.csv")
        label_files = glob.glob(pattern)
        
        if not label_files:
            print(f"  Warning: No CSV files found in {labels_dir}")
        else:
            print(f"  Found {len(label_files)} label files:")
            for file in label_files:
                df = pd.read_csv(file)
                labels_dfs.append(df)
                print(f"    {os.path.basename(file)}: {len(df)} rows")
    
    if not labels_dfs:
        raise ValueError("No labels provided. Use --labels, --labels-dir, or both.")
    
    # Combine all labels
    if len(labels_dfs) == 1:
        combined_df = labels_dfs[0]
    else:
        combined_df = pd.concat(labels_dfs, ignore_index=True)
        print(f"  Combined labels: {len(combined_df)} total rows")
    
    # Deduplicate by (player_id, game_date), keeping last occurrence
    print("Deduplicating labels by (player_id, game_date)...")
    initial_count = len(combined_df)
    combined_df = combined_df.drop_duplicates(subset=['player_id', 'game_date'], keep='last')
    final_count = len(combined_df)
    duplicates_removed = initial_count - final_count
    
    if duplicates_removed > 0:
        print(f"  Removed {duplicates_removed} duplicate rows")
    
    print(f"Final labels: {len(combined_df)} rows")
    print(f"Label columns: {list(combined_df.columns)}")
    
    return combined_df


def join_masters_and_labels(masters_df, labels_df):
    """Join master sheet data with labels on player_id and date"""
    print("Joining master sheets with labels...")
    
    # Ensure we have required columns
    required_master_cols = ['slate_date', 'player_id', 'Name', 'Position', 'TeamAbbrev', 'Salary', 'Game Info', 'method']
    required_label_cols = ['player_id', 'dk_points']
    
    # Check master sheet columns
    missing_master_cols = [col for col in required_master_cols if col not in masters_df.columns]
    if missing_master_cols:
        raise ValueError(f"Missing required columns in master sheets: {missing_master_cols}")
    
    # Check label columns
    missing_label_cols = [col for col in required_label_cols if col not in labels_df.columns]
    if missing_label_cols:
        raise ValueError(f"Missing required columns in labels: {missing_label_cols}")
    
    # Check if game_date is available in both datasets
    if 'game_date' in masters_df.columns and 'game_date' in labels_df.columns:
        print("Primary join on (player_id, game_date)...")
        
        # Convert dates to string for comparison
        masters_df['game_date_str'] = masters_df['game_date'].astype(str)
        labels_df['game_date_str'] = labels_df['game_date'].astype(str)
        
        # Primary join on (player_id, game_date)
        joined_df = masters_df.merge(
            labels_df[['game_date_str', 'player_id', 'dk_points']], 
            left_on=['player_id', 'game_date_str'],
            right_on=['player_id', 'game_date_str'],
            how='left'
        )
        
        # Count unmatched rows
        unmatched_count = joined_df['dk_points'].isna().sum()
        if unmatched_count > 0:
            print(f"Warning: {unmatched_count} rows unmatched on (player_id, game_date)")
            
            # Fallback join on (player_id, slate_date) for unmatched rows
            if 'slate_date' in masters_df.columns:
                print("Attempting fallback join on (player_id, slate_date)...")
                
                # Get unmatched rows
                unmatched_mask = joined_df['dk_points'].isna()
                unmatched_master = masters_df[unmatched_mask].copy()
                
                # Convert slate_date to string for comparison
                unmatched_master['slate_date_str'] = unmatched_master['slate_date'].astype(str)
                
                # For fallback join, we need to create a slate_date column in labels_df
                # by extracting it from game_date (assuming same format)
                labels_df['slate_date_str'] = labels_df['game_date'].astype(str)
                
                # Try fallback join
                fallback_joined = unmatched_master.merge(
                    labels_df[['slate_date_str', 'player_id', 'dk_points']], 
                    left_on=['player_id', 'slate_date_str'],
                    right_on=['player_id', 'slate_date_str'],
                    how='left'
                )
                
                fallback_matches = fallback_joined['dk_points'].notna().sum()
                
                if fallback_matches > 0:
                    print(f"Fallback join matched {fallback_matches} additional rows")
                    # Update the main joined_df with fallback matches
                    for idx in fallback_joined[fallback_joined['dk_points'].notna()].index:
                        master_idx = unmatched_master.index[fallback_joined.index.get_loc(idx)]
                        joined_df.loc[master_idx, 'dk_points'] = fallback_joined.loc[idx, 'dk_points']
                else:
                    print("Fallback join provided no additional matches")
                
                # Clean up temporary columns
                labels_df = labels_df.drop(['slate_date_str'], axis=1)
                unmatched_master = unmatched_master.drop(['slate_date_str'], axis=1)
            else:
                print("Warning: Cannot attempt fallback join - missing slate_date column")
        
        # Clean up temporary columns
        joined_df = joined_df.drop(['game_date_str'], axis=1)
        masters_df = masters_df.drop(['game_date_str'], axis=1)
        labels_df = labels_df.drop(['game_date_str'], axis=1)
        
    else:
        # Fallback to slate_date join if game_date not available
        if 'slate_date' in masters_df.columns and 'slate_date' in labels_df.columns:
            print("Warning: game_date not available, falling back to (player_id, slate_date) join")
            
            # Convert dates to string for comparison
            masters_df['slate_date_str'] = masters_df['slate_date'].astype(str)
            labels_df['slate_date_str'] = labels_df['slate_date'].astype(str)
            
            joined_df = masters_df.merge(
                labels_df[['slate_date_str', 'player_id', 'dk_points']], 
                left_on=['player_id', 'slate_date_str'],
                right_on=['player_id', 'slate_date_str'],
                how='left'
            )
            
            # Clean up temporary columns
            joined_df = joined_df.drop(['slate_date_str'], axis=1)
            masters_df = masters_df.drop(['slate_date_str'], axis=1)
            labels_df = labels_df.drop(['slate_date_str'], axis=1)
            
        else:
            raise ValueError("Neither game_date nor slate_date available for joining")
    
    print(f"Joined data: {len(joined_df)} rows")
    
    return joined_df


def join_features(joined_df, usage_file=None, vegas_file=None, opp_file=None):
    """Join optional feature files to the training table"""
    feature_counts = {'usage': 0, 'vegas': 0, 'opp': 0}
    
    if usage_file:
        print(f"Joining usage features from: {usage_file}")
        try:
            usage_df = pd.read_csv(usage_file)
            # Convert dates to string for comparison
            usage_df['game_date_str'] = usage_df['game_date'].astype(str)
            joined_df['slate_date_str'] = joined_df['slate_date'].astype(str)
            
            # Join on player_id and date
            feature_cols = [col for col in usage_df.columns if col not in ['player_id', 'game_date']]
            usage_features = usage_df[['game_date_str', 'player_id'] + feature_cols].copy()
            
            # Prefix columns with 'usage_'
            usage_features.columns = ['game_date_str', 'player_id'] + [f'usage_{col}' for col in feature_cols]
            
            joined_df = joined_df.merge(
                usage_features,
                left_on=['player_id', 'slate_date_str'],
                right_on=['player_id', 'game_date_str'],
                how='left'
            )
            
            feature_counts['usage'] = len(feature_cols)
            print(f"  Added {len(feature_cols)} usage features")
        except Exception as e:
            print(f"  Warning: Could not load usage features: {e}")
    
    if vegas_file:
        print(f"Joining vegas features from: {vegas_file}")
        try:
            vegas_df = pd.read_csv(vegas_file)
            # Convert dates to string for comparison
            vegas_df['game_date_str'] = vegas_df['game_date'].astype(str)
            
            # Join on TeamAbbrev and date
            feature_cols = [col for col in vegas_df.columns if col not in ['TeamAbbrev', 'game_date']]
            vegas_features = vegas_df[['game_date_str', 'TeamAbbrev'] + feature_cols].copy()
            
            # Prefix columns with 'vegas_'
            vegas_features.columns = ['game_date_str', 'TeamAbbrev'] + [f'vegas_{col}' for col in feature_cols]
            
            joined_df = joined_df.merge(
                vegas_features,
                left_on=['TeamAbbrev', 'slate_date_str'],
                right_on=['TeamAbbrev', 'game_date_str'],
                how='left'
            )
            
            feature_counts['vegas'] = len(feature_cols)
            print(f"  Added {len(feature_cols)} vegas features")
        except Exception as e:
            print(f"  Warning: Could not load vegas features: {e}")
    
    if opp_file:
        print(f"Joining opponent features from: {opp_file}")
        try:
            opp_df = pd.read_csv(opp_file)
            # Convert dates to string for comparison
            opp_df['game_date_str'] = opp_df['game_date'].astype(str)
            
            # Join on TeamAbbrev and date
            feature_cols = [col for col in opp_df.columns if col not in ['TeamAbbrev', 'game_date']]
            opp_features = opp_df[['game_date_str', 'TeamAbbrev'] + feature_cols].copy()
            
            # Prefix columns with 'opp_'
            opp_features.columns = ['game_date_str', 'TeamAbbrev'] + [f'opp_{col}' for col in feature_cols]
            
            joined_df = joined_df.merge(
                opp_features,
                left_on=['TeamAbbrev', 'slate_date_str'],
                right_on=['TeamAbbrev', 'game_date_str'],
                how='left'
            )
            
            feature_counts['opp'] = len(feature_cols)
            print(f"  Added {len(feature_cols)} opponent features")
        except Exception as e:
            print(f"  Warning: Could not load opponent features: {e}")
    
    # Clean up temporary date columns
    if any([usage_file, vegas_file, opp_file]):
        joined_df = joined_df.drop('slate_date_str', axis=1, errors='ignore')
    
    print(f"Features attached: usage={feature_counts['usage']} cols, vegas={feature_counts['vegas']} cols, opp={feature_counts['opp']} cols")
    
    return joined_df


def write_output(joined_df, output_file):
    """Write output file with specified column order"""
    print(f"Writing output to: {output_file}")
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Start with core columns in desired order
    core_columns = [
        'slate_date', 'player_id', 'dk_points', 'Salary', 'Position', 
        'TeamAbbrev', 'Name', 'method', 'Game Info'
    ]
    
    # Add all other columns (features) after core columns
    all_columns = list(joined_df.columns)
    feature_columns = [col for col in all_columns if col not in core_columns]
    
    # Final column order: core columns first, then features
    output_columns = core_columns + feature_columns
    
    # Check if all required columns exist
    missing_cols = [col for col in core_columns if col not in joined_df.columns]
    if missing_cols:
        print(f"Warning: Missing core columns in output: {missing_cols}")
        # Only include columns that exist
        output_columns = [col for col in output_columns if col in joined_df.columns]
    
    output_df = joined_df[output_columns]
    output_df.to_csv(output_file, index=False)
    
    print(f"Output written: {len(output_df)} rows, {len(output_columns)} columns")
    if feature_columns:
        print(f"Feature columns added: {feature_columns}")


def print_summary(joined_df):
    """Print summary statistics"""
    total_rows = len(joined_df)
    labeled_rows = joined_df['dk_points'].notna().sum()
    unlabeled_rows = total_rows - labeled_rows
    
    print(f"\n=== TRAINING TABLE SUMMARY ===")
    print(f"Total rows: {total_rows:,}")
    print(f"Labeled rows: {labeled_rows:,} ({labeled_rows/total_rows*100:.1f}%)")
    print(f"Unlabeled rows: {unlabeled_rows:,} ({unlabeled_rows/total_rows*100:.1f}%)")
    
    # Show sample of labeled data
    if labeled_rows > 0:
        print(f"\nSample labeled rows:")
        sample_labeled = joined_df[joined_df['dk_points'].notna()].head(3)
        for _, row in sample_labeled.iterrows():
            print(f"  {row['Name']} ({row['Position']}, {row['TeamAbbrev']}) - {row['Salary']:,}")


def run_data_checks(joined_df):
    """Run data quality checks on the training table"""
    print(f"\n=== DATA QUALITY CHECKS ===")
    
    # Duplicate key check on (slate_date, player_id)
    print("Duplicate key check on (slate_date, player_id):")
    duplicates = joined_df.groupby(['slate_date', 'player_id']).size().reset_index(name='count')
    duplicates = duplicates[duplicates['count'] > 1]
    
    if len(duplicates) > 0:
        print(f"  Found {len(duplicates)} duplicate keys:")
        for _, row in duplicates.head(5).iterrows():
            print(f"    {row['slate_date']}, {row['player_id']} - {row['count']} occurrences")
    else:
        print("  No duplicate keys found")
    
    # Salary sanity check
    print("\nSalary sanity check:")
    non_dst_df = joined_df[joined_df['Position'] != 'DST']
    min_salary = non_dst_df['Salary'].min()
    max_salary = non_dst_df['Salary'].max()
    low_salary_count = len(non_dst_df[non_dst_df['Salary'] < 2500])
    
    print(f"  Min salary (non-DST): ${min_salary:,}")
    print(f"  Max salary (non-DST): ${max_salary:,}")
    print(f"  Rows with Salary < $2,500: {low_salary_count}")
    
    # Method distribution
    print("\nMethod distribution:")
    method_counts = joined_df['method'].value_counts()
    total_rows = len(joined_df)
    for method, count in method_counts.items():
        percentage = count / total_rows * 100
        print(f"  {method}: {count:,} ({percentage:.1f}%)")
    
    # Position distribution
    print("\nRow counts by Position:")
    position_counts = joined_df['Position'].value_counts()
    for position, count in position_counts.items():
        print(f"  {position}: {count:,}")


def main():
    parser = argparse.ArgumentParser(description='Build training table from master sheets and labels')
    parser.add_argument('--masters-dir', required=True, help='Directory containing master sheet files')
    parser.add_argument('--labels', help='Path to specific labels CSV file')
    parser.add_argument('--labels-dir', help='Directory containing labels CSV files (will stack all)')
    parser.add_argument('--usage', help='Path to usage features CSV (player-level)')
    parser.add_argument('--vegas', help='Path to vegas features CSV (team-level)')
    parser.add_argument('--opp', help='Path to opponent features CSV (team-level)')
    parser.add_argument('--out', required=True, help='Output path for training table')
    parser.add_argument('--checks', action='store_true', help='Run data quality checks after writing output')
    
    args = parser.parse_args()
    
    # Validate that at least one labels source is provided
    if not args.labels and not args.labels_dir:
        parser.error("At least one of --labels or --labels-dir must be provided")
    
    try:
        # Load master sheets
        masters_df = load_master_sheets(args.masters_dir)
        
        # Load labels
        labels_df = load_labels(args.labels, args.labels_dir)
        
        # Join data
        joined_df = join_masters_and_labels(masters_df, labels_df)
        
        # Join optional features
        joined_df = join_features(joined_df, args.usage, args.vegas, args.opp)
        
        # Write output
        write_output(joined_df, args.out)
        
        # Print summary
        print_summary(joined_df)
        
        # Run data quality checks if requested
        if args.checks:
            run_data_checks(joined_df)
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
