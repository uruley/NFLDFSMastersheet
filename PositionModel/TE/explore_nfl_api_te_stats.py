#!/usr/bin/env python3
"""
Explore NFL API TE Stats
See what columns and data are available for Tight Ends
"""
import pandas as pd
import numpy as np
import nfl_data_py as nfl

def explore_te_stats():
    """Explore available TE stats from NFL API."""
    print("🔍 Exploring NFL API TE Stats")
    print("=" * 50)
    
    # Load recent TE data
    print("Loading TE data from 2024 season...")
    weekly_data = nfl.import_weekly_data([2024])
    te_data = weekly_data[weekly_data['position'] == 'TE'].copy()
    
    print(f"Loaded {len(te_data)} TE weekly records")
    print(f"Data shape: {te_data.shape}")
    
    # Show all available columns
    print(f"\n📊 ALL AVAILABLE COLUMNS ({len(te_data.columns)} total):")
    for i, col in enumerate(te_data.columns):
        print(f"{i+1:2d}. {col}")
    
    # Analyze key stat columns
    print(f"\n🎯 KEY STATISTICS ANALYSIS:")
    
    # Receiving stats
    receiving_cols = [col for col in te_data.columns if 'receiving' in col.lower()]
    print(f"\n📥 Receiving Stats ({len(receiving_cols)} columns):")
    for col in receiving_cols:
        non_null = te_data[col].notna().sum()
        unique_vals = te_data[col].nunique()
        print(f"  • {col}: {non_null} non-null, {unique_vals} unique values")
    
    # Rushing stats
    rushing_cols = [col for col in te_data.columns if 'rushing' in col.lower()]
    print(f"\n🏃 Rushing Stats ({len(rushing_cols)} columns):")
    for col in rushing_cols:
        non_null = te_data[col].notna().sum()
        unique_vals = te_data[col].nunique()
        print(f"  • {col}: {non_null} non-null, {unique_vals} unique values")
    
    # Fantasy points
    fantasy_cols = [col for col in te_data.columns if 'fantasy' in col.lower()]
    print(f"\n⭐ Fantasy Points ({len(fantasy_cols)} columns):")
    for col in fantasy_cols:
        non_null = te_data[col].notna().sum()
        if non_null > 0:
            min_val = te_data[col].min()
            max_val = te_data[col].max()
            mean_val = te_data[col].mean()
            print(f"  • {col}: {non_null} non-null, range [{min_val:.2f}, {max_val:.2f}], mean {mean_val:.2f}")
    
    # Target and efficiency stats
    target_cols = [col for col in te_data.columns if any(x in col.lower() for x in ['target', 'share', 'wopr', 'epa'])]
    print(f"\n🎯 Target & Efficiency Stats ({len(target_cols)} columns):")
    for col in target_cols:
        non_null = te_data[col].notna().sum()
        if non_null > 0:
            unique_vals = te_data[col].nunique()
            print(f"  • {col}: {non_null} non-null, {unique_vals} unique values")
    
    # Sample data for top performers
    print(f"\n📈 SAMPLE DATA - Top 5 TE Performances:")
    
    # Calculate PPR fantasy points
    te_data['fantasy_points_ppr'] = (
        te_data['receptions'] * 1.0 +
        te_data['receiving_yards'] * 0.1 +
        te_data['rushing_yards'] * 0.1 +
        te_data['receiving_tds'] * 6.0 +
        te_data['rushing_tds'] * 6.0 +
        te_data.get('return_tds', 0) * 6.0 +
        te_data.get('two_point_conversions', 0) * 2.0 -
        te_data.get('fumbles_lost', 0) * 2.0
    ).fillna(0)
    
    # Show top performances
    top_performances = te_data.nlargest(5, 'fantasy_points_ppr')
    
    display_cols = [
        'player_name', 'recent_team', 'week', 'fantasy_points_ppr',
        'receptions', 'targets', 'receiving_yards', 'receiving_tds',
        'rushing_yards', 'rushing_tds'
    ]
    
    # Only show columns that exist
    existing_cols = [col for col in display_cols if col in top_performances.columns]
    print(top_performances[existing_cols].to_string(index=False))
    
    # Data quality analysis
    print(f"\n🔍 DATA QUALITY ANALYSIS:")
    
    # Check for missing data in key columns
    key_cols = ['receptions', 'targets', 'receiving_yards', 'receiving_tds', 'fantasy_points_ppr']
    existing_key_cols = [col for col in key_cols if col in te_data.columns]
    
    for col in existing_key_cols:
        missing_pct = (te_data[col].isna().sum() / len(te_data)) * 100
        print(f"  • {col}: {missing_pct:.1f}% missing")
    
    # Check data ranges
    print(f"\n📊 DATA RANGES:")
    for col in existing_key_cols:
        if te_data[col].dtype in ['int64', 'float64']:
            min_val = te_data[col].min()
            max_val = te_data[col].max()
            mean_val = te_data[col].mean()
            print(f"  • {col}: [{min_val:.2f}, {max_val:.2f}], mean {mean_val:.2f}")
    
    # Save sample data for inspection
    sample_file = "te_nfl_api_sample.csv"
    te_data.head(100).to_csv(sample_file, index=False)
    print(f"\n✅ Sample data saved to {sample_file}")
    
    return te_data

def analyze_feature_correlations(te_data):
    """Analyze correlations between features and fantasy points."""
    print(f"\n🔗 FEATURE CORRELATIONS WITH FANTASY POINTS:")
    
    # Select numeric columns for correlation analysis
    numeric_cols = te_data.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove non-stat columns
    exclude_cols = ['player_id', 'season', 'week', 'fantasy_points', 'fantasy_points_ppr']
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    if 'fantasy_points_ppr' in te_data.columns:
        target_col = 'fantasy_points_ppr'
    else:
        target_col = 'fantasy_points'
    
    if target_col not in te_data.columns:
        print(f"Target column {target_col} not found")
        return
    
    # Calculate correlations
    correlations = []
    for col in feature_cols:
        if te_data[col].notna().sum() > 0:  # Only if column has data
            corr = te_data[col].corr(te_data[target_col])
            if not pd.isna(corr):
                correlations.append((col, corr))
    
    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    
    print(f"Top 20 features by correlation with {target_col}:")
    for i, (col, corr) in enumerate(correlations[:20]):
        print(f"  {i+1:2d}. {col}: {corr:+.3f}")
    
    return correlations

def main():
    """Main exploration function."""
    print("🚀 NFL API TE Stats Exploration")
    print("=" * 50)
    
    # Explore available stats
    te_data = explore_te_stats()
    
    # Analyze correlations
    correlations = analyze_feature_correlations(te_data)
    
    print(f"\n🎯 EXPLORATION COMPLETE!")
    print(f"Key findings:")
    print(f"• {len(te_data.columns)} total columns available")
    print(f"• {len(te_data)} TE weekly records")
    print(f"• Fantasy points range: {te_data['fantasy_points_ppr'].min():.1f} to {te_data['fantasy_points_ppr'].max():.1f}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    print(f"• Use raw stats (receptions, targets, yards) rather than derived features")
    print(f"• Ensure consistent fantasy point calculation between training and inference")
    print(f"• Check for data leakage in feature engineering")

if __name__ == "__main__":
    main()
