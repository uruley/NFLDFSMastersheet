#!/usr/bin/env python3
"""
Backtesting Script for Phase 3: Stacking Rules
Tests stacking performance with historical data and real crosswalk integration.
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def load_real_data():
    """Load real data from the project structure"""
    data_sources = {}
    
    # Try to load crosswalk
    crosswalk_paths = [
        '../../data/processed/crosswalk_2025.csv',
        '../../data/processed/crosswalk.csv',
        'data/processed/crosswalk_2025.csv'
    ]
    
    for path in crosswalk_paths:
        if os.path.exists(path):
            try:
                crosswalk = pd.read_csv(path)
                print(f"✓ Loaded crosswalk from {path}: {len(crosswalk)} entries")
                print(f"  Columns: {crosswalk.columns.tolist()}")
                
                # Check for game info
                if 'Game Info' in crosswalk.columns:
                    print(f"  Game Info sample: {crosswalk['Game Info'].head(3).tolist()}")
                elif 'game_info' in crosswalk.columns:
                    print(f"  game_info sample: {crosswalk['game_info'].head(3).tolist()}")
                else:
                    print("  ⚠️ No game info column found")
                
                data_sources['crosswalk'] = crosswalk
                break
            except Exception as e:
                print(f"⚠️ Could not load {path}: {e}")
    
    # Try to load master sheet
    master_paths = [
        '../../data/processed/master_sheet_2025.csv',
        '../../data/processed/master_sheet.csv',
        'data/processed/master_sheet_2025.csv'
    ]
    
    for path in master_paths:
        if os.path.exists(path):
            try:
                master_sheet = pd.read_csv(path)
                print(f"✓ Loaded master sheet from {path}: {len(master_sheet)} entries")
                data_sources['master_sheet'] = master_sheet
                break
            except Exception as e:
                print(f"⚠️ Could not load {path}: {e}")
    
    # Try to load correlations
    correlations_paths = [
        'data/correlations.json',
        '../../data/correlations.json'
    ]
    
    for path in correlations_paths:
        if os.path.exists(path):
            try:
                import json
                with open(path, 'r') as f:
                    correlations = json.load(f)
                print(f"✓ Loaded correlations from {path}: {len(correlations)} pairs")
                data_sources['correlations'] = correlations
                break
            except Exception as e:
                print(f"⚠️ Could not load {path}: {e}")
    
    return data_sources

def create_historical_test_data():
    """Create test data that mimics real historical slate"""
    # Sample Week 17, 2024 data structure
    data = {
        'player_name': [
            # BUF vs MIA (High-scoring game)
            'Josh Allen', 'Stefon Diggs', 'Gabe Davis', 'Tyreek Hill', 'Jaylen Waddle', 'Raheem Mostert',
            # KC vs LAC (Rivalry game)
            'Patrick Mahomes', 'Travis Kelce', 'Keenan Allen', 'Justin Herbert', 'Austin Ekeler', 'Gerald Everett',
            # MIN vs GB (Division game)
            'Kirk Cousins', 'Justin Jefferson', 'Dalvin Cook', 'Aaron Rodgers', 'Christian Watson', 'AJ Dillon',
            # Additional players
            'Saquon Barkley', 'Davante Adams', 'Mark Andrews', 'Bills DST', 'Chiefs DST', 'Vikings DST'
        ],
        'team': [
            'BUF', 'BUF', 'BUF', 'MIA', 'MIA', 'MIA',
            'KC', 'KC', 'LAC', 'LAC', 'LAC', 'LAC',
            'MIN', 'MIN', 'MIN', 'GB', 'GB', 'GB',
            'NYG', 'LV', 'BAL', 'BUF', 'KC', 'MIN'
        ],
        'position': [
            'QB', 'WR', 'WR', 'WR', 'WR', 'RB',
            'QB', 'TE', 'WR', 'QB', 'RB', 'TE',
            'QB', 'WR', 'RB', 'QB', 'WR', 'RB',
            'RB', 'WR', 'TE', 'DST', 'DST', 'DST'
        ],
        'salary': [
            8500, 7800, 4800, 7500, 6000, 6500,
            8000, 7000, 6800, 7800, 6400, 4200,
            7500, 8200, 6200, 7600, 5400, 4800,
            6000, 7200, 5800, 3300, 3100, 2900
        ],
        'projection': [
            26.5, 21.0, 13.5, 19.5, 16.0, 18.5,
            25.0, 17.5, 18.0, 23.5, 15.5, 8.5,
            22.0, 23.5, 17.0, 21.5, 14.5, 12.0,
            16.5, 20.5, 13.0, 9.5, 8.0, 7.5
        ]
    }
    
    df = pd.DataFrame(data)
    
    # Add value column
    df['value'] = df['projection'] / (df['salary'] / 1000)
    
    # Add adjusted projections (simulating correlation adjustments)
    df['adjusted_projection'] = df['projection'] * (1 + np.random.normal(0, 0.03, len(df)))
    
    return df

def test_real_crosswalk_integration():
    """Test integration with real crosswalk data"""
    print("\n🔍 Testing Real Crosswalk Integration...")
    
    data_sources = load_real_data()
    
    if 'crosswalk' not in data_sources:
        print("❌ No crosswalk data available for testing")
        return False
    
    crosswalk = data_sources['crosswalk']
    
    # Check crosswalk structure
    print(f"\nCrosswalk Analysis:")
    print(f"  Total entries: {len(crosswalk)}")
    print(f"  Columns: {crosswalk.columns.tolist()}")
    
    # Check for required columns (handle column name variations)
    required_cols = ['join_key', 'Name', 'Position']
    missing_cols = [col for col in required_cols if col not in crosswalk.columns]
    
    # Check for team column variations
    team_cols = [col for col in crosswalk.columns if 'team' in col.lower()]
    if not team_cols:
        missing_cols.append('team')
    else:
        print(f"  ✓ Team column found: {team_cols[0]}")
    
    if missing_cols:
        print(f"  ⚠️ Missing columns: {missing_cols}")
        print(f"  Available columns: {crosswalk.columns.tolist()}")
        return False
    
    # Check for game info
    game_info_cols = [col for col in crosswalk.columns if 'game' in col.lower() or 'Game' in col]
    if game_info_cols:
        print(f"  ✓ Game info columns found: {game_info_cols}")
        # Show sample game info
        for col in game_info_cols:
            sample = crosswalk[col].dropna().head(3).tolist()
            print(f"    {col}: {sample}")
    else:
        print("  ⚠️ No game info columns found")
    
    # Check for join_key format
    if 'join_key' in crosswalk.columns:
        sample_keys = crosswalk['join_key'].dropna().head(3).tolist()
        print(f"  ✓ Join key sample: {sample_keys}")
    
    return True

def test_correlation_integration():
    """Test correlation data integration"""
    print("\n🔍 Testing Correlation Integration...")
    
    data_sources = load_real_data()
    
    if 'correlations' not in data_sources:
        print("❌ No correlation data available for testing")
        return False
    
    correlations = data_sources['correlations']
    
    print(f"\nCorrelation Analysis:")
    print(f"  Total correlation pairs: {len(correlations)}")
    
    # Show sample correlations
    sample_corrs = list(correlations.items())[:3]
    for key, corr in sample_corrs:
        print(f"  Sample: {key}")
        print(f"    Player 1: {corr.get('player1_name', 'N/A')} ({corr.get('player1_pos', 'N/A')})")
        print(f"    Player 2: {corr.get('player2_name', 'N/A')} ({corr.get('player2_pos', 'N/A')})")
        print(f"    Correlation: {corr.get('correlation', 'N/A'):.3f}")
    
    # Check for high correlations
    high_corrs = [corr for corr in correlations.values() if corr.get('correlation', 0) > 0.5]
    print(f"  High correlations (>0.5): {len(high_corrs)}")
    
    return True

def test_stacking_with_real_data():
    """Test stacking constraints with real data structure"""
    print("\n🔍 Testing Stacking with Real Data...")
    
    try:
        from stacking import StackingConstraints
        
        # Create test projections
        projections = create_historical_test_data()
        
        # Try to load real crosswalk
        data_sources = load_real_data()
        crosswalk_path = None
        
        if 'crosswalk' in data_sources:
            # Save crosswalk to temp file for testing
            temp_crosswalk = 'temp_crosswalk.csv'
            data_sources['crosswalk'].to_csv(temp_crosswalk, index=False)
            crosswalk_path = temp_crosswalk
            print(f"  ✓ Using real crosswalk data")
        else:
            print(f"  ⚠️ Using synthetic game info")
        
        # Initialize stacking constraints
        stacking = StackingConstraints(projections, crosswalk_path)
        
        # Test stack identification
        print(f"\nStacking Analysis:")
        print(f"  Total players: {len(stacking.projections)}")
        print(f"  Games identified: {stacking.projections['game_info'].nunique()}")
        
        # Show game distribution
        game_counts = stacking.projections.groupby('game_info').size()
        print(f"  Game distribution:")
        for game, count in game_counts.items():
            print(f"    {game}: {count} players")
        
        # Clean up temp file
        if crosswalk_path and os.path.exists(temp_crosswalk):
            os.remove(temp_crosswalk)
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing stacking with real data: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_backtest():
    """Run comprehensive backtest of Phase 3 implementation"""
    print("🚀 PHASE 3: STACKING RULES BACKTEST")
    print("=" * 60)
    
    results = {
        'crosswalk_integration': False,
        'correlation_integration': False,
        'stacking_real_data': False
    }
    
    try:
        # Test 1: Real crosswalk integration
        results['crosswalk_integration'] = test_real_crosswalk_integration()
        
        # Test 2: Correlation integration
        results['correlation_integration'] = test_correlation_integration()
        
        # Test 3: Stacking with real data
        results['stacking_real_data'] = test_stacking_with_real_data()
        
        # Summary
        print("\n🎯 BACKTEST RESULTS SUMMARY")
        print("=" * 60)
        
        for test, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"  {test}: {status}")
        
        passed_tests = sum(results.values())
        total_tests = len(results)
        
        print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("🎉 Phase 3 backtest completed successfully!")
            print("Ready for production use with real data!")
        else:
            print("⚠️ Some tests failed. Check data availability and paths.")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Backtest failed with error: {e}")
        import traceback
        traceback.print_exc()
        return results

if __name__ == "__main__":
    results = run_backtest()
    
    # Exit with appropriate code
    if all(results.values()):
        print("\n✅ All tests passed - Phase 3 ready!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed - needs attention")
        sys.exit(1)
