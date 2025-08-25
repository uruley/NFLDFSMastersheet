#!/usr/bin/env python3
"""
Explore RB Features from NFL API
This script examines what features are available for running backs in the NFL data.
"""

import pandas as pd
import nfl_data_py as nfl
import numpy as np

def explore_rb_features():
    """Explore available features for running backs in NFL API data."""
    print("🔍 EXPLORING RB FEATURES FROM NFL API")
    print("=" * 60)
    
    # Load recent RB data to see what's available
    print("Loading recent RB data from NFL API...")
    try:
        # Load last 2 seasons of data for RBs
        weekly_data = nfl.import_weekly_data([2023, 2024])
        print(f"✅ Loaded {len(weekly_data):,} total weekly records")
        
        # Filter for RBs only
        rb_data = weekly_data[weekly_data['position'] == 'RB'].copy()
        print(f"✅ Found {len(rb_data):,} RB weekly records")
        
        # Show all available columns
        print(f"\n📊 ALL AVAILABLE COLUMNS ({len(rb_data.columns)} total):")
        for i, col in enumerate(rb_data.columns, 1):
            print(f"  {i:2d}. {col}")
        
        # Show RB-specific columns (filter by common RB terms)
        rb_terms = ['rush', 'carry', 'reception', 'target', 'snap', 'touch', 'block']
        rb_columns = [col for col in rb_data.columns if any(term in col.lower() for term in rb_terms)]
        
        print(f"\n🏃‍♂️ RB-SPECIFIC COLUMNS ({len(rb_columns)} found):")
        for i, col in enumerate(rb_columns, 1):
            print(f"  {i:2d}. {col}")
        
        # Show sample RB data
        print(f"\n📋 SAMPLE RB DATA (first 3 players, first 2 weeks):")
        sample_rb = rb_data.groupby('player_id').head(2).head(6)
        print(sample_rb[['player_name', 'recent_team', 'season', 'week', 'carries', 'rushing_yards', 'rushing_tds', 'receptions', 'receiving_yards', 'receiving_tds']].to_string())
        
        # Analyze feature distributions
        print(f"\n📈 FEATURE ANALYSIS:")
        
        # Rushing features
        rushing_features = ['carries', 'rushing_yards', 'rushing_tds', 'rushing_epa', 'rushing_first_downs']
        print(f"\n🏃‍♂️ RUSHING FEATURES:")
        for feature in rushing_features:
            if feature in rb_data.columns:
                non_zero = (rb_data[feature] > 0).sum()
                avg_val = rb_data[rb_data[feature] > 0][feature].mean() if non_zero > 0 else 0
                print(f"  • {feature:<20}: {non_zero:>5} non-zero values, avg: {avg_val:>6.2f}")
            else:
                print(f"  • {feature:<20}: ❌ NOT AVAILABLE")
        
        # Receiving features
        receiving_features = ['receptions', 'receiving_yards', 'receiving_tds', 'receiving_epa', 'targets', 'receiving_air_yards']
        print(f"\n🎯 RECEIVING FEATURES:")
        for feature in receiving_features:
            if feature in rb_data.columns:
                non_zero = (rb_data[feature] > 0).sum()
                avg_val = rb_data[rb_data[feature] > 0][feature].mean() if non_zero > 0 else 0
                print(f"  • {feature:<20}: {non_zero:>5} non-zero values, avg: {avg_val:>6.2f}")
            else:
                print(f"  • {feature:<20}: ❌ NOT AVAILABLE")
        
        # Efficiency features
        efficiency_features = ['yards_per_carry', 'yards_per_reception', 'catch_rate', 'touchdown_rate']
        print(f"\n⚡ EFFICIENCY FEATURES:")
        for feature in efficiency_features:
            if feature in rb_data.columns:
                non_zero = (rb_data[feature] > 0).sum()
                avg_val = rb_data[rb_data[feature] > 0][feature].mean() if non_zero > 0 else 0
                print(f"  • {feature:<20}: {non_zero:>5} non-zero values, avg: {avg_val:>6.2f}")
            else:
                print(f"  • {feature:<20}: ❌ NOT AVAILABLE")
        
        # Context features
        context_features = ['fantasy_points', 'opponent_team', 'game_total', 'spread_line', 'weather', 'wind_speed']
        print(f"\n🌍 CONTEXT FEATURES:")
        for feature in context_features:
            if feature in rb_data.columns:
                non_zero = (rb_data[feature] != 0).sum()
                avg_val = rb_data[feature].mean()
                print(f"  • {feature:<20}: {non_zero:>5} non-zero values, avg: {avg_val:>6.2f}")
            else:
                print(f"  • {feature:<20}: ❌ NOT AVAILABLE")
        
        # Show top RBs by fantasy points
        print(f"\n🏆 TOP 10 RB PERFORMANCES (by fantasy points):")
        top_rbs = rb_data.nlargest(10, 'fantasy_points')[['player_name', 'recent_team', 'season', 'week', 'fantasy_points', 'carries', 'rushing_yards', 'rushing_tds', 'receptions', 'receiving_yards', 'receiving_tds']]
        print(top_rbs.to_string(index=False))
        
        # Show feature correlations with fantasy points
        print(f"\n🔗 FEATURE CORRELATIONS WITH FANTASY POINTS:")
        numeric_cols = rb_data.select_dtypes(include=[np.number]).columns
        correlations = rb_data[numeric_cols].corr()['fantasy_points'].sort_values(ascending=False)
        
        print("Top 15 positive correlations:")
        for feature, corr in correlations.head(16).items():  # Skip fantasy_points itself
            if feature != 'fantasy_points':
                print(f"  • {feature:<25}: {corr:>6.3f}")
        
        print("\nTop 15 negative correlations:")
        for feature, corr in correlations.tail(15).items():
            print(f"  • {feature:<25}: {corr:>6.3f}")
        
        return rb_data
        
    except Exception as e:
        print(f"❌ Error loading NFL data: {e}")
        return None

def suggest_rb_features(rb_data):
    """Suggest the best features for RB modeling based on the data."""
    print(f"\n💡 RB FEATURE RECOMMENDATIONS:")
    print("=" * 60)
    
    if rb_data is None:
        print("❌ No data available for recommendations")
        return
    
    # Core rushing features
    print(f"🏃‍♂️ CORE RUSHING FEATURES (Must-have):")
    core_rushing = ['carries', 'rushing_yards', 'rushing_tds', 'rushing_epa']
    for feature in core_rushing:
        if feature in rb_data.columns:
            print(f"  ✅ {feature}")
        else:
            print(f"  ❌ {feature} - MISSING!")
    
    # Core receiving features
    print(f"\n🎯 CORE RECEIVING FEATURES (Must-have):")
    core_receiving = ['receptions', 'receiving_yards', 'receiving_tds', 'targets']
    for feature in core_receiving:
        if feature in rb_data.columns:
            print(f"  ✅ {feature}")
        else:
            print(f"  ❌ {feature} - MISSING!")
    
    # Efficiency features
    print(f"\n⚡ EFFICIENCY FEATURES (Important):")
    efficiency = ['yards_per_carry', 'yards_per_reception', 'rushing_first_downs', 'receiving_first_downs']
    for feature in efficiency:
        if feature in rb_data.columns:
            print(f"  ✅ {feature}")
        else:
            print(f"  ❌ {feature} - MISSING!")
    
    # Context features
    print(f"\n🌍 CONTEXT FEATURES (Important):")
    context = ['opponent_team', 'game_total', 'spread_line', 'team', 'week', 'season']
    for feature in context:
        if feature in rb_data.columns:
            print(f"  ✅ {feature}")
        else:
            print(f"  ❌ {feature} - MISSING!")
    
    # Derived features to create
    print(f"\n🔧 DERIVED FEATURES TO CREATE:")
    derived = [
        'total_touches (carries + receptions)',
        'total_yards (rushing + receiving)',
        'total_tds (rushing + receiving)',
        'touchdown_rate (tds / touches)',
        'yards_per_touch (total_yards / total_touches)',
        'red_zone_touches (if available)',
        'game_script (positive/negative game flow)',
        'workload_share (touches / team_total_touches)'
    ]
    for feature in derived:
        print(f"  🔧 {feature}")

def main():
    """Main function to explore RB features."""
    print("🚀 RB FEATURE EXPLORATION STARTING...")
    
    # Explore available features
    rb_data = explore_rb_features()
    
    # Suggest best features for modeling
    suggest_rb_features(rb_data)
    
    print(f"\n🎯 NEXT STEPS:")
    print(f"  1. Review the available features above")
    print(f"  2. Identify which features to use in RB model")
    print(f"  3. Create RB inference script based on QB structure")
    print(f"  4. Train RB model with selected features")

if __name__ == "__main__":
    main()
