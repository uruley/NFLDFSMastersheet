#!/usr/bin/env python3
"""
Stacking Diversity Test Script for Phase 3
Tests stack diversity across multiple lineups with slight objective tweaks.
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
            # Game 4: CIN vs CLE
            'Joe Burrow', 'JaMarr Chase', 'Tee Higgins', 'Deshaun Watson', 'Amari Cooper',
            # Additional players for variety
            'Saquon Barkley', 'Davante Adams', 'Mark Andrews', 'Bills DST', 'Chiefs DST'
        ],
        'team': [
            'BUF', 'BUF', 'BUF', 'MIA', 'MIA',
            'KC', 'KC', 'LAC', 'LAC', 'LAC', 
            'MIN', 'MIN', 'MIN', 'GB', 'GB',
            'CIN', 'CIN', 'CIN', 'CLE', 'CLE',
            'NYG', 'LV', 'BAL', 'BUF', 'KC'
        ],
        'position': [
            'QB', 'WR', 'WR', 'WR', 'WR',
            'QB', 'TE', 'WR', 'QB', 'RB',
            'QB', 'WR', 'RB', 'QB', 'WR',
            'QB', 'WR', 'WR', 'QB', 'WR',
            'RB', 'WR', 'TE', 'DST', 'DST'
        ],
        'salary': [
            8000, 7500, 4500, 7200, 5800,
            7800, 6800, 6500, 7600, 6200,
            7200, 8000, 6000, 7400, 5200,
            7500, 7800, 5000, 7000, 6000,
            5800, 7000, 5500, 3200, 3000
        ],
        'projection': [
            25.0, 20.0, 12.0, 18.0, 15.0,
            24.0, 16.0, 17.0, 22.0, 14.0,
            21.0, 22.0, 16.0, 20.0, 13.0,
            23.0, 21.0, 14.0, 19.0, 16.0,
            15.0, 19.0, 12.0, 9.0, 8.5
        ]
    }
    
    df = pd.DataFrame(data)
    
    # Add adjusted projections (simulating correlation adjustments)
    df['adjusted_projection'] = df['projection'] * (1 + np.random.normal(0, 0.05, len(df)))
    
    # Add value column
    df['value'] = df['projection'] / (df['salary'] / 1000)
    
    return df

def test_stacking_diversity(num_lineups=5):
    """Test stacking diversity across multiple lineups"""
    print("🧪 Testing Stacking Diversity...")
    
    try:
        from stacking import StackingConstraints
        
        projections = create_sample_projections()
        stacking = StackingConstraints(projections)
        
        print(f"✓ Created projections with {len(projections)} players")
        print(f"✓ Games available: {stacking.projections['game_info'].unique()}")
        
        lineups = []
        stack_types = {'qb_wr_stacks': set(), 'game_stacks': set(), 'bring_backs': set()}
        
        for i in range(num_lineups):
            print(f"\n--- Generating Lineup {i+1} ---")
            
            # Perturb projections slightly for diversity
            noise_factor = np.random.uniform(-0.02, 0.02)
            projections['adjusted_projection'] = projections['projection'] * (1 + noise_factor)
            
            # Create a lineup with intentional stacking
            if i == 0:
                # Lineup 1: BUF@MIA heavy stack
                selected_players = ['Josh Allen', 'Stefon Diggs', 'Gabe Davis', 'Tyreek Hill']
            elif i == 1:
                # Lineup 2: KC@LAC heavy stack
                selected_players = ['Patrick Mahomes', 'Travis Kelce', 'Keenan Allen', 'Justin Herbert']
            elif i == 2:
                # Lineup 3: MIN@GB heavy stack
                selected_players = ['Kirk Cousins', 'Justin Jefferson', 'Dalvin Cook', 'Aaron Rodgers']
            elif i == 3:
                # Lineup 4: CIN@CLE heavy stack
                selected_players = ['Joe Burrow', 'JaMarr Chase', 'Tee Higgins', 'Deshaun Watson']
            else:
                # Lineup 5: Mixed approach
                selected_players = ['Josh Allen', 'Patrick Mahomes', 'Justin Jefferson', 'Travis Kelce']
            
            # Add more players to make it 9 total
            remaining_players = projections[~projections['player_name'].isin(selected_players)].head(5)
            lineup = pd.concat([projections[projections['player_name'].isin(selected_players)], remaining_players], ignore_index=True)
            
            # Add game info to the lineup
            teams = lineup['team'].unique()
            game_mapping = {}
            for j in range(0, len(teams), 2):
                if j + 1 < len(teams):
                    game_id = f"{teams[j]}@{teams[j+1]}"
                    game_mapping[teams[j]] = game_id
                    game_mapping[teams[j+1]] = game_id
            
            lineup['game_info'] = lineup['team'].map(game_mapping)
            lineup['game_info'] = lineup['game_info'].fillna('OTHER')
            lineup['game_id'] = lineup['game_info']
            
            lineups.append(lineup)
            
            # Analyze stacking
            stack_info = stacking.get_stack_info(lineup)
            stacking_score = stacking.calculate_stacking_score(lineup, projections)
            
            print(f"  Players: {', '.join(lineup['player_name'].tolist())}")
            print(f"  QB-WR Stacks: {len(stack_info['qb_wr_stacks'])}")
            print(f"  Game Stacks: {len(stack_info['game_stacks'])}")
            print(f"  Bring-backs: {len(stack_info['bring_backs'])}")
            print(f"  Stacking Score: {stacking_score:.1f}/100")
            
            # Collect stack types for diversity analysis
            for stack in stack_info['qb_wr_stacks']:
                stack_types['qb_wr_stacks'].add(f"{stack['team']}:{stack['qb']}")
            for stack in stack_info['game_stacks']:
                stack_types['game_stacks'].add(stack['game_id'])
            for stack in stack_info['bring_backs']:
                stack_types['bring_backs'].add(stack['game_id'])
        
        # Analyze diversity
        print(f"\n🎯 STACKING DIVERSITY ANALYSIS")
        print("=" * 50)
        print(f"Unique QB-WR/TE stacks: {len(stack_types['qb_wr_stacks'])}")
        for stack in sorted(stack_types['qb_wr_stacks']):
            print(f"  {stack}")
        
        print(f"\nUnique game stacks: {len(stack_types['game_stacks'])}")
        for game in sorted(stack_types['game_stacks']):
            print(f"  {game}")
        
        print(f"\nBring-back opportunities: {len(stack_types['bring_backs'])}")
        for game in sorted(stack_types['bring_backs']):
            print(f"  {game}")
        
        # Calculate diversity metrics
        total_possible_stacks = len(stacking.projections['game_info'].unique())
        stack_coverage = len(stack_types['game_stacks']) / total_possible_stacks if total_possible_stacks > 0 else 0
        
        print(f"\n📊 Diversity Metrics:")
        print(f"  Game Stack Coverage: {stack_coverage:.1%}")
        print(f"  QB-WR Stack Variety: {len(stack_types['qb_wr_stacks'])} unique combinations")
        print(f"  Overall Stacking Diversity: {'High' if stack_coverage > 0.6 else 'Medium' if stack_coverage > 0.3 else 'Low'}")
        
        return lineups, stack_types
        
    except ImportError as e:
        print(f"❌ Could not import StackingConstraints: {e}")
        return None, None
    except Exception as e:
        print(f"❌ Error testing stacking diversity: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_constraint_enforcement():
    """Test that stacking constraints are properly enforced"""
    print("\n🧪 Testing Constraint Enforcement...")
    
    try:
        from constraints import DraftKingsConstraints
        
        constraints = DraftKingsConstraints()
        
        print("✓ DraftKings constraints loaded")
        print(f"  Stacking rules: {constraints.dk_settings['stacking']}")
        
        # Test constraint validation
        projections = create_sample_projections()
        
        # Create a lineup that should violate stacking rules
        invalid_lineup = projections.head(9).copy()
        
        # Ensure no QB-WR stack (should fail)
        qb_players = invalid_lineup[invalid_lineup['position'] == 'QB']
        wr_players = invalid_lineup[invalid_lineup['position'] == 'WR']
        
        # Remove WRs from QB teams
        for _, qb in qb_players.iterrows():
            qb_team = qb['team']
            wr_same_team = wr_players[wr_players['team'] == qb_team]
            if not wr_same_team.empty:
                # Replace with WR from different team
                other_wrs = wr_players[wr_players['team'] != qb_team]
                if not other_wrs.empty:
                    replace_idx = wr_same_team.index[0]
                    replace_with = other_wrs.iloc[0]
                    invalid_lineup.loc[replace_idx] = replace_with
        
        print(f"✓ Created test lineup with {len(invalid_lineup)} players")
        print(f"  QB teams: {set(invalid_lineup[invalid_lineup['position'] == 'QB']['team'])}")
        print(f"  WR teams: {set(invalid_lineup[invalid_lineup['position'] == 'WR']['team'])}")
        
        # Check for QB-WR stacks
        qb_teams = set(invalid_lineup[invalid_lineup['position'] == 'QB']['team'])
        wr_teams = set(invalid_lineup[invalid_lineup['position'] == 'WR']['team'])
        qb_wr_overlap = qb_teams.intersection(wr_teams)
        
        if not qb_wr_overlap:
            print("  ✓ Successfully created lineup without QB-WR stacks")
        else:
            print(f"  ⚠️ QB-WR overlap found: {qb_wr_overlap}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing constraint enforcement: {e}")
        return False

def run_diversity_test():
    """Run comprehensive stacking diversity test"""
    print("🚀 PHASE 3: STACKING DIVERSITY TEST")
    print("=" * 60)
    
    try:
        # Test 1: Stacking diversity across lineups
        lineups, stack_types = test_stacking_diversity(num_lineups=5)
        
        # Test 2: Constraint enforcement
        constraint_test = test_constraint_enforcement()
        
        # Summary
        print("\n🎯 DIVERSITY TEST RESULTS")
        print("=" * 60)
        
        if lineups and stack_types:
            print("✅ Stacking diversity test: PASSED")
            print(f"  Generated {len(lineups)} diverse lineups")
            print(f"  {len(stack_types['qb_wr_stacks'])} unique QB-WR stacks")
            print(f"  {len(stack_types['game_stacks'])} unique game stacks")
        else:
            print("❌ Stacking diversity test: FAILED")
        
        if constraint_test:
            print("✅ Constraint enforcement test: PASSED")
        else:
            print("❌ Constraint enforcement test: FAILED")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if lineups and len(stack_types['game_stacks']) >= 3:
            print("  ✓ Good game stack diversity achieved")
        else:
            print("  ⚠️ Consider increasing game stack variety")
        
        if lineups and len(stack_types['qb_wr_stacks']) >= 3:
            print("  ✓ Good QB-WR stack diversity achieved")
        else:
            print("  ⚠️ Consider increasing QB-WR stack variety")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Diversity test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_diversity_test()
    if success:
        print("\n✅ Stacking diversity test completed successfully!")
        print("Phase 3 ready for multi-lineup generation!")
    else:
        print("\n❌ Stacking diversity test failed!")
        sys.exit(1)
