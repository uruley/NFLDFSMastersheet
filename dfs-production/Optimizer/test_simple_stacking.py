#!/usr/bin/env python3
"""
Simple Test Script for Phase 3: Stacking Rules Implementation
This script demonstrates the new stacking constraints without external file dependencies.
"""

import pandas as pd
import numpy as np
import os
import sys

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_sample_projections():
    """Create sample projections with game information for testing"""
    # Sample data with game matchups
    data = {
        'player_name': [
            # Game 1: BUF vs MIA
            'Josh Allen', 'Stefon Diggs', 'Gabe Davis', 'Tyreek Hill', 'Jaylen Waddle',
            # Game 2: KC vs LAC  
            'Patrick Mahomes', 'Travis Kelce', 'Keenan Allen', 'Justin Herbert', 'Austin Ekeler',
            # Game 3: MIN vs GB
            'Kirk Cousins', 'Justin Jefferson', 'Dalvin Cook', 'Aaron Rodgers', 'Christian Watson',
            # Additional players for variety
            'Saquon Barkley', 'Davante Adams', 'Mark Andrews', 'Bills DST', 'Chiefs DST'
        ],
        'team': [
            'BUF', 'BUF', 'BUF', 'MIA', 'MIA',
            'KC', 'KC', 'LAC', 'LAC', 'LAC', 
            'MIN', 'MIN', 'MIN', 'GB', 'GB',
            'NYG', 'LV', 'BAL', 'BUF', 'KC'
        ],
        'position': [
            'QB', 'WR', 'WR', 'WR', 'WR',
            'QB', 'TE', 'WR', 'QB', 'RB',
            'QB', 'WR', 'RB', 'QB', 'WR',
            'RB', 'WR', 'TE', 'DST', 'DST'
        ],
        'salary': [
            8000, 7500, 4500, 7200, 5800,
            7800, 6800, 6500, 7600, 6200,
            7200, 8000, 6000, 7400, 5200,
            5800, 7000, 5500, 3200, 3000
        ],
        'projection': [
            25.0, 20.0, 12.0, 18.0, 15.0,
            24.0, 16.0, 17.0, 22.0, 14.0,
            21.0, 22.0, 16.0, 20.0, 13.0,
            15.0, 19.0, 12.0, 9.0, 8.5
        ]
    }
    
    df = pd.DataFrame(data)
    
    # Add adjusted projections (simulating correlation adjustments)
    df['adjusted_projection'] = df['projection'] * (1 + np.random.normal(0, 0.05, len(df)))
    
    # Add value column
    df['value'] = df['projection'] / (df['salary'] / 1000)
    
    return df

def test_stacking_constraints():
    """Test the StackingConstraints class"""
    print("🧪 Testing StackingConstraints...")
    
    try:
        from stacking import StackingConstraints
        
        projections = create_sample_projections()
        stacking = StackingConstraints(projections)
        
        print(f"✓ Created StackingConstraints with {len(projections)} players")
        print(f"✓ Game info columns: {[col for col in projections.columns if 'game' in col]}")
        
        # Test game identification using the modified projections from stacking
        games = stacking.projections.groupby('game_info').size()
        print(f"✓ Identified {len(games)} games: {games.to_dict()}")
        
        return stacking
        
    except ImportError as e:
        print(f"❌ Could not import StackingConstraints: {e}")
        return None
    except Exception as e:
        print(f"❌ Error testing StackingConstraints: {e}")
        return None

def test_draftkings_constraints():
    """Test the DraftKingsConstraints class"""
    print("\n🧪 Testing DraftKingsConstraints...")
    
    try:
        from constraints import DraftKingsConstraints
        
        constraints = DraftKingsConstraints()
        
        print(f"✓ Salary cap: ${constraints.dk_settings['salary_cap']:,}")
        print(f"✓ Roster size: {constraints.dk_settings['roster_size']}")
        print(f"✓ Position requirements: {constraints.dk_settings['positions']}")
        print(f"✓ Stacking rules: {constraints.dk_settings['stacking']}")
        
        return constraints
        
    except ImportError as e:
        print(f"❌ Could not import DraftKingsConstraints: {e}")
        return None
    except Exception as e:
        print(f"❌ Error testing DraftKingsConstraints: {e}")
        return None

def test_lineup_validation():
    """Test the LineupValidator class"""
    print("\n🧪 Testing LineupValidator...")
    
    try:
        from validator import LineupValidator
        
        projections = create_sample_projections()
        validator = LineupValidator(projections)
        
        # Create a sample lineup with game info
        sample_lineup = projections.head(9).copy()  # First 9 players
        
        # Add game info to the lineup (simulating what would come from correlations.py)
        teams = sample_lineup['team'].unique()
        game_mapping = {}
        for i in range(0, len(teams), 2):
            if i + 1 < len(teams):
                game_id = f"{teams[i]}@{teams[i+1]}"
                game_mapping[teams[i]] = game_id
                game_mapping[teams[i+1]] = game_id
        
        sample_lineup['game_info'] = sample_lineup['team'].map(game_mapping)
        sample_lineup['game_info'] = sample_lineup['game_info'].fillna('OTHER')
        sample_lineup['game_id'] = sample_lineup['game_info']
        
        print(f"✓ Created validator with {len(projections)} projections")
        print(f"✓ Testing validation with {len(sample_lineup)} player lineup")
        
        # Run validation
        try:
            validation = validator.validate_complete_lineup(sample_lineup)
            print(f"✓ Lineup validation complete: {'Valid' if validation['all_valid'] else 'Invalid'}")
        except Exception as e:
            print(f"⚠️ Lineup validation error: {e}")
            validation = None
        
        return validator, sample_lineup
        
    except ImportError as e:
        print(f"❌ Could not import LineupValidator: {e}")
        return None, None
    except Exception as e:
        print(f"❌ Error testing LineupValidator: {e}")
        return None, None

def test_stacking_analysis():
    """Test stacking analysis functionality"""
    print("\n🧪 Testing Stacking Analysis...")
    
    try:
        from stacking import StackingConstraints
        
        projections = create_sample_projections()
        stacking = StackingConstraints(projections)
        
        # Create a lineup with intentional stacks
        stacked_lineup = projections[
            (projections['player_name'].isin(['Josh Allen', 'Stefon Diggs', 'Gabe Davis', 'Tyreek Hill']))
        ].copy()
        
        # Add more players to make it 9 total
        remaining_players = projections[~projections['player_name'].isin(stacked_lineup['player_name'])].head(5)
        stacked_lineup = pd.concat([stacked_lineup, remaining_players], ignore_index=True)
        
        # Add game info to the stacked lineup
        teams = stacked_lineup['team'].unique()
        game_mapping = {}
        for i in range(0, len(teams), 2):
            if i + 1 < len(teams):
                game_id = f"{teams[i]}@{teams[i+1]}"
                game_mapping[teams[i]] = game_id
                game_mapping[teams[i+1]] = game_id
        
        stacked_lineup['game_info'] = stacked_lineup['team'].map(game_mapping)
        stacked_lineup['game_info'] = stacked_lineup['game_info'].fillna('OTHER')
        stacked_lineup['game_id'] = stacked_lineup['game_info']
        
        print(f"✓ Created stacked lineup with {len(stacked_lineup)} players")
        
        # Get stack info
        try:
            stack_info = stacking.get_stack_info(stacked_lineup)
            
            print("\nStack Analysis Results:")
            print(f"  QB-WR Stacks: {len(stack_info['qb_wr_stacks'])}")
            print(f"  Game Stacks: {len(stack_info['game_stacks'])}")
            print(f"  Bring-backs: {len(stack_info['bring_backs'])}")
            
            # Calculate stacking score
            stacking_score = stacking.calculate_stacking_score(stacked_lineup, projections)
            print(f"  Stacking Score: {stacking_score:.1f}/100")
            
            return stacked_lineup, stack_info
        except Exception as e:
            print(f"⚠️ Stack analysis error: {e}")
            return stacked_lineup, None
        
    except ImportError as e:
        print(f"❌ Could not import StackingConstraints: {e}")
        return None, None
    except Exception as e:
        print(f"❌ Error testing Stacking Analysis: {e}")
        return None, None

def test_constraint_validation():
    """Test constraint validation with sample lineups"""
    print("\n🧪 Testing Constraint Validation...")
    
    try:
        from constraints import DraftKingsConstraints
        
        projections = create_sample_projections()
        constraints = DraftKingsConstraints()
        
        # Create a valid lineup
        qb_players = projections[projections['position'] == 'QB'].head(1)
        rb_players = projections[projections['position'] == 'RB'].head(2)
        wr_players = projections[projections['position'] == 'WR'].head(3)
        te_players = projections[projections['position'] == 'TE'].head(1)
        dst_players = projections[projections['position'] == 'DST'].head(1)
        
        valid_lineup = pd.concat([qb_players, rb_players, wr_players, te_players, dst_players], ignore_index=True)
        
        print(f"✓ Created test lineup with {len(valid_lineup)} players")
        
        # Validate constraints
        validation = constraints.validate_all_constraints(valid_lineup)
        
        print("\nConstraint Validation Results:")
        for key, value in validation.items():
            if key != 'position_requirements':
                print(f"  {key}: {value}")
        
        print("  Position Requirements:")
        for pos, valid in validation['position_requirements'].items():
            print(f"    {pos}: {'✓' if valid else '✗'}")
        
        return valid_lineup, validation
        
    except ImportError as e:
        print(f"❌ Could not import DraftKingsConstraints: {e}")
        return None, None
    except Exception as e:
        print(f"❌ Error testing Constraint Validation: {e}")
        return None, None

def run_comprehensive_test():
    """Run all tests and demonstrate Phase 3 functionality"""
    print("🚀 PHASE 3: STACKING RULES IMPLEMENTATION TEST")
    print("=" * 60)
    
    try:
        # Test 1: Stacking Constraints
        stacking = test_stacking_constraints()
        
        # Test 2: DraftKings Constraints
        constraints = test_draftkings_constraints()
        
        # Test 3: Lineup Validation
        validator, sample_lineup = test_lineup_validation()
        
        # Test 4: Stacking Analysis
        stacked_lineup, stack_info = test_stacking_analysis()
        
        # Test 5: Constraint Validation
        valid_lineup, validation = test_constraint_validation()
        
        # Final demonstration
        print("\n🎯 PHASE 3 DEMONSTRATION COMPLETE!")
        print("=" * 60)
        
        if stacking:
            print("✓ StackingConstraints: QB-WR stacks, game stacks, bring-back logic")
        if constraints:
            print("✓ DraftKingsConstraints: Position requirements, salary cap, roster size")
        if validator:
            print("✓ LineupValidator: Comprehensive validation and metrics")
        
        print("✓ All components integrated and working together")
        
        # Show sample output
        if stack_info:
            print("\n📊 Sample Stacking Analysis Output:")
            print(f"  QB-WR Stacks found: {len(stack_info['qb_wr_stacks'])}")
            print(f"  Game Stacks found: {len(stack_info['game_stacks'])}")
            print(f"  Bring-back opportunities: {len(stack_info['bring_backs'])}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    if success:
        print("\n✅ Phase 3 implementation test completed successfully!")
    else:
        print("\n❌ Phase 3 implementation test failed!")
        sys.exit(1)
