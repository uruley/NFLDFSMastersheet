# optimizer/lineup_optimizer.py
import pandas as pd
import numpy as np
from pulp import *
from correlations import CorrelationCalculator
from stacking import StackingConstraints  # New import

class LineupOptimizer:
    def __init__(self):
        """Initialize DraftKings lineup optimizer"""
        self.salary_cap = 50000
        self.roster_size = 9
        self.projections = pd.DataFrame()
        self.correlation_calculator = CorrelationCalculator()
        self.stacking_constraints = None  # Initialize later with projections

    def load_projections(self, qb_file, rb_file, wr_file, te_file, dst_file=None):
        """Load all position projections from your CSV files"""
        # Load each position file
        qb_df = self.standardize_projection_df(pd.read_csv(qb_file), 'QB')
        rb_df = self.standardize_projection_df(pd.read_csv(rb_file), 'RB')
        wr_df = self.standardize_projection_df(pd.read_csv(wr_file), 'WR')
        te_df = self.standardize_projection_df(pd.read_csv(te_file), 'TE')
        
        # Combine all projections
        all_dfs = [qb_df, rb_df, wr_df, te_df]
        
        # Add DST if available (or use placeholder)
        if dst_file:
            dst_df = self.standardize_projection_df(pd.read_csv(dst_file), 'DST')
            all_dfs.append(dst_df)
        else:
            dst_df = self.create_placeholder_dst()
            all_dfs.append(dst_df)
        
        self.projections = pd.concat(all_dfs, ignore_index=True)
        
        # Load or calculate correlations
        correlations = self.correlation_calculator.load_correlations()
        if not correlations:
            print("No correlations found, calculating new ones...")
            weekly_data, rosters = self.correlation_calculator.fetch_historical_data()
            correlations = self.correlation_calculator.calculate_correlations(weekly_data, rosters)
        
        # Apply correlation adjustments
        self.projections = self.correlation_calculator.apply_correlation_adjustments(self.projections)
        print(f"Applied correlation adjustments. Columns: {self.projections.columns.tolist()}")
        
        # Initialize stacking constraints
        self.stacking_constraints = StackingConstraints(self.projections)
        
        # Debug output
        print(f"Loaded {len(self.projections)} total players")
        print("\nSample adjusted projections:")
        print(self.projections[['player_name', 'team', 'position', 'salary', 'projection', 'adjusted_projection']].head(10))
        return self.projections

    def standardize_projection_df(self, df, position):
        """Standardize column names across different position files"""
        print(f"\nStandardizing {position} data...")
        print(f"Original columns: {df.columns.tolist()}")
        
        column_mapping = {
            'Name': 'player_name',
            'Team': 'team',
            'Position': 'position',
            'Salary': 'salary',
            'Player_ID': 'player_id',
            'Ensemble_Prediction': 'projection',
            'Value': 'value',
            'Predicted_Points': 'projection',
            'predicted_points_ensemble': 'projection',
            'dk_salary': 'salary',
            'recent_team': 'team',
            'player_name': 'player_name'
        }
        
        df = df.rename(columns=column_mapping)
        print(f"After renaming: {df.columns.tolist()}")
        
        df['position'] = position
        keep_cols = ['player_name', 'team', 'position', 'salary', 'player_id', 'projection', 'value']
        available_cols = [col for col in keep_cols if col in df.columns]
        print(f"Available columns: {available_cols}")
        
        missing_cols = [col for col in keep_cols if col not in df.columns]
        if missing_cols:
            print(f"⚠️ Missing columns: {missing_cols}")
            print(f"Sample data before filtering:")
            print(df.head(2))
            if 'value' not in df.columns:
                df['value'] = df['projection'] / (df.get('salary', 5000) / 1000)
                print(f"Added 'value' column with calculated values")
            if 'salary' not in df.columns:
                df['salary'] = 5000
                print(f"Added 'salary' column with default $5000")
        
        # Fix invalid salaries (0 or negative values)
        invalid_salaries = (df['salary'] <= 0) | df['salary'].isna()
        if invalid_salaries.any():
            invalid_count = invalid_salaries.sum()
            print(f"⚠️ Found {invalid_count} players with invalid salaries (≤0 or NaN)")
            
            # Set reasonable defaults based on position
            position_defaults = {
                'QB': 6500, 'RB': 6000, 'WR': 5000, 'TE': 4500, 'DST': 3000
            }
            
            for pos, default_salary in position_defaults.items():
                pos_mask = (df['position'] == pos) & invalid_salaries
                if pos_mask.any():
                    df.loc[pos_mask, 'salary'] = default_salary
                    print(f"  Set {pos} players to default ${default_salary:,}")
            
            # Recalculate value for fixed salaries
            df.loc[invalid_salaries, 'value'] = df.loc[invalid_salaries, 'projection'] / (df.loc[invalid_salaries, 'salary'] / 1000)
        
        available_cols = [col for col in keep_cols if col in df.columns]
        print(f"Final available columns: {available_cols}")
        return df[available_cols]

    def create_placeholder_dst(self):
        """Create placeholder DST projections until you build DST model"""
        dst_data = {
            'player_name': ['Bills DST', 'Cowboys DST', 'Ravens DST', 'Jets DST', '49ers DST'],
            'team': ['BUF', 'DAL', 'BAL', 'NYJ', 'SF'],
            'position': ['DST', 'DST', 'DST', 'DST', 'DST'],
            'salary': [3200, 3000, 2900, 2700, 2800],
            'player_id': ['DST_BUF', 'DST_DAL', 'DST_BAL', 'DST_NYJ', 'DST_SF'],
            'projection': [9.0, 8.5, 8.0, 7.5, 8.2],
            'value': [2.8, 2.83, 2.76, 2.78, 2.93]
        }
        return pd.DataFrame(dst_data)

    def optimize_single_lineup(self):
        """Generate one optimal lineup with stacking constraints"""
        if self.projections.empty:
            raise ValueError("No projections loaded. Call load_projections first.")
        
        prob = LpProblem("DFS_Lineup", LpMaximize)
        players = self.projections.index.tolist()
        player_vars = LpVariable.dicts("players", players, cat='Binary')
        
        # Objective: Maximize adjusted projections
        prob += lpSum([player_vars[i] * self.projections.loc[i, 'adjusted_projection'] for i in players])
        
        # Salary constraint
        prob += lpSum([player_vars[i] * self.projections.loc[i, 'salary'] for i in players]) <= self.salary_cap
        
        # Roster size constraint
        prob += lpSum([player_vars[i] for i in players]) == self.roster_size
        
        # Position constraints
        position_requirements = {
            'QB': {'min': 1, 'max': 1},
            'RB': {'min': 2, 'max': 3},
            'WR': {'min': 3, 'max': 4},
            'TE': {'min': 1, 'max': 2},
            'DST': {'min': 1, 'max': 1}
        }
        for pos, reqs in position_requirements.items():
            pos_players = self.projections[self.projections['position'] == pos].index.tolist()
            if pos_players:
                prob += lpSum([player_vars[i] for i in pos_players]) >= reqs['min']
                prob += lpSum([player_vars[i] for i in pos_players]) <= reqs['max']
        
        # FLEX constraint
        flex_players = self.projections[self.projections['position'].isin(['RB', 'WR', 'TE'])].index.tolist()
        prob += lpSum([player_vars[i] for i in flex_players]) == 7
        
        # Max players per team
        for team in self.projections['team'].unique():
            if pd.notna(team):
                team_players = self.projections[self.projections['team'] == team].index.tolist()
                prob += lpSum([player_vars[i] for i in team_players]) <= 7
        
        # Add stacking constraints
        if self.stacking_constraints:
            self.stacking_constraints.add_qb_wr_stack(prob, player_vars)
            self.stacking_constraints.add_game_stack(prob, player_vars, min_players=4)
            self.stacking_constraints.add_bring_back(prob, player_vars)
        
        # Solve
        prob.solve()
        if LpStatus[prob.status] != 'Optimal':
            print(f"Warning: Solution status is {LpStatus[prob.status]}")
            return None
        
        # Extract lineup
        lineup_indices = [i for i in players if player_vars[i].varValue == 1]
        lineup = self.projections.loc[lineup_indices].copy()
        
        # Sort by position
        position_order = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 4, 'DST': 5}
        lineup['pos_order'] = lineup['position'].map(position_order)
        lineup = lineup.sort_values('pos_order')
        
        return lineup

    def display_lineup(self, lineup):
        """Pretty print the lineup with stack details"""
        print("\n" + "="*60)
        print("OPTIMAL DRAFTKINGS LINEUP (WITH CORRELATIONS & STACKS)")
        print("="*60)
        
        for _, player in lineup.iterrows():
            print(f"{player['position']:4} {player['player_name']:20} {player['team']:4} "
                  f"${player['salary']:5,}  {player['adjusted_projection']:5.1f} pts")
        
        print("-"*60)
        print(f"{'Total Salary:':30} ${lineup['salary'].sum():,}")
        print(f"{'Projected Points:':30} {lineup['adjusted_projection'].sum():.2f}")
        print(f"{'Avg Value (pts/$1000):':30} {lineup['value'].mean():.2f}")
        print("="*60)
        
        # Display stack info
        if self.stacking_constraints:
            stack_info = self.stacking_constraints.get_stack_info(lineup)
            print("\nStacking Details:")
            for stack_type, stacks in stack_info.items():
                print(f"\n{stack_type.replace('_', ' ').title()}:")
                for stack in stacks:
                    print(stack)
        
        self.validate_lineup(lineup)

    def validate_lineup(self, lineup):
        """Ensure lineup meets DraftKings requirements"""
        pos_counts = lineup['position'].value_counts()
        checks = {
            '✓ 9 players': len(lineup) == 9,
            '✓ Under $50,000': lineup['salary'].sum() <= 50000,
            '✓ 1 QB': pos_counts.get('QB', 0) == 1,
            '✓ 2-3 RB': 2 <= pos_counts.get('RB', 0) <= 3,
            '✓ 3-4 WR': 3 <= pos_counts.get('WR', 0) <= 4,
            '✓ 1-2 TE': 1 <= pos_counts.get('TE', 0) <= 2,
            '✓ 1 DST': pos_counts.get('DST', 0) == 1,
            '✓ Valid FLEX': sum(pos_counts.get(p, 0) for p in ['RB', 'WR', 'TE']) == 7
        }
        
        print("\nLineup Validation:")
        all_valid = True
        for check, result in checks.items():
            status = "✓" if result else "✗"
            print(f"  {status} {check}")
            if not result:
                all_valid = False
        return all_valid

    def save_lineup_to_csv(self, lineup, filename=None):
        """Save the optimal lineup to a CSV file with stacking analysis"""
        if filename is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"optimal_lineup_{timestamp}.csv"
        
        # Clean up lineup data for export
        export_data = lineup.copy()
        
        # Keep only essential columns for clean output
        essential_cols = ['position', 'player_name', 'team', 'salary', 'projection', 'adjusted_projection', 'value']
        available_cols = [col for col in essential_cols if col in export_data.columns]
        export_data = export_data[available_cols]
        
        # Add stacking analysis if available
        if self.stacking_constraints:
            try:
                stack_info = self.stacking_constraints.get_stack_info(lineup)
                stacking_score = self.stacking_constraints.calculate_stacking_score(lineup, self.projections)
                
                # Add stacking summary
                stack_summary = pd.DataFrame({
                    'position': ['STACKING_ANALYSIS'],
                    'player_name': [f"QB-WR Stacks: {len(stack_info['qb_wr_stacks'])}, Game Stacks: {len(stack_info['game_stacks'])}, Bring-backs: {len(stack_info['bring_backs'])}"],
                    'team': [f"Stacking Score: {stacking_score:.1f}/100"],
                    'salary': [''],
                    'projection': [''],
                    'value': ['']
                })
                
                # Add game info if available
                if 'game_info' in lineup.columns:
                    export_data['game_info'] = lineup['game_info']
                
                export_data = pd.concat([stack_summary, export_data], ignore_index=True)
            except Exception as e:
                print(f"⚠️ Could not add stacking analysis: {e}")
        
        # Create summary row
        summary_row = pd.DataFrame({
            'position': ['SUMMARY'],
            'player_name': [f"Total: {len(lineup)} players"],
            'team': [''],
            'salary': [lineup['salary'].sum()],
            'projection': [lineup['adjusted_projection'].sum()],
            'value': [lineup['value'].mean()],
            'game_info': [f"Salary Cap: ${self.salary_cap:,}"] if 'game_info' in export_data.columns else [f"Salary Cap: ${self.salary_cap:,}"]
        })
        
        final_export = pd.concat([summary_row, export_data], ignore_index=True)
        
        # Ensure output directory exists
        import os
        os.makedirs('output', exist_ok=True)
        
        csv_path = f"output/{filename}"
        final_export.to_csv(csv_path, index=False)
        print(f"\n💾 Lineup saved to: {csv_path}")
        print(f"📊 Lineup includes stacking analysis and game information")
        return csv_path

def test_optimizer():
    """Test the optimizer with your projection files"""
    optimizer = LineupOptimizer()
    optimizer.load_projections(
        qb_file='../../PositionModel/QB/qb_predictions_fixed.csv',
        rb_file='../../PositionModel/RB/rb_predictions.csv',
        wr_file='../../PositionModel/WR/wr_predictions_with_salaries.csv',
        te_file='../../PositionModel/TE/te_predictions_with_salaries.csv',
        dst_file=None
    )
    lineup = optimizer.optimize_single_lineup()
    if lineup is not None:
        optimizer.display_lineup(lineup)
        try:
            csv_path = optimizer.save_lineup_to_csv(lineup)
            print(f"\n🎯 Lineup successfully exported to: {csv_path}")
        except Exception as e:
            print(f"⚠️ Warning: Could not save to CSV: {e}")
            print("   Lineup data is still available in the optimizer object")
        return lineup
    else:
        print("Failed to generate valid lineup")
        return None

if __name__ == "__main__":
    lineup = test_optimizer()