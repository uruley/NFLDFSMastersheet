# optimizer/constraints.py
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Set
import json
import os

class DraftKingsConstraints:
    def __init__(self, data_dir='data'):
        """Initialize DraftKings specific constraints"""
        self.data_dir = data_dir
        self.dk_settings_file = os.path.join(data_dir, 'dk_settings.json')
        self.load_dk_settings()
        
    def load_dk_settings(self):
        """Load DraftKings specific settings and requirements"""
        if os.path.exists(self.dk_settings_file):
            with open(self.dk_settings_file, 'r') as f:
                self.dk_settings = json.load(f)
        else:
            # Default DraftKings settings
            self.dk_settings = {
                "roster_size": 9,
                "salary_cap": 50000,
                "positions": {
                    "QB": {"min": 1, "max": 1},
                    "RB": {"min": 2, "max": 3},
                    "WR": {"min": 3, "max": 4},
                    "TE": {"min": 1, "max": 2},
                    "DST": {"min": 1, "max": 1}
                },
                "stacking": {
                    "min_qb_wr_stack": 1,
                    "min_game_stack": 4,
                    "max_players_per_team": 4,
                    "bring_back_required": True
                }
            }
    
    def validate_position_requirements(self, lineup: pd.DataFrame) -> Dict[str, bool]:
        """Validate that lineup meets DraftKings position requirements"""
        position_counts = lineup['position'].value_counts()
        requirements = self.dk_settings['positions']
        
        validation = {}
        for pos, req in requirements.items():
            count = position_counts.get(pos, 0)
            validation[pos] = req['min'] <= count <= req['max']
        
        return validation
    
    def validate_salary_cap(self, lineup: pd.DataFrame) -> bool:
        """Validate that lineup is under salary cap"""
        total_salary = lineup['salary'].sum()
        return total_salary <= self.dk_settings['salary_cap']
    
    def validate_roster_size(self, lineup: pd.DataFrame) -> bool:
        """Validate that lineup has correct number of players"""
        return len(lineup) == self.dk_settings['roster_size']
    
    def validate_all_constraints(self, lineup: pd.DataFrame) -> Dict[str, any]:
        """Validate all DraftKings constraints"""
        results = {
            'position_requirements': self.validate_position_requirements(lineup),
            'salary_cap': self.validate_salary_cap(lineup),
            'roster_size': self.validate_roster_size(lineup),
            'total_salary': lineup['salary'].sum(),
            'salary_remaining': self.dk_settings['salary_cap'] - lineup['salary'].sum()
        }
        
        # Overall validation
        all_valid = (
            all(results['position_requirements'].values()) and
            results['salary_cap'] and
            results['roster_size']
        )
        results['all_valid'] = all_valid
        
        return results
    
    def get_constraint_violations(self, lineup: pd.DataFrame) -> List[str]:
        """Get list of specific constraint violations"""
        violations = []
        validation = self.validate_all_constraints(lineup)
        
        # Check position requirements
        for pos, valid in validation['position_requirements'].items():
            if not valid:
                count = lineup[lineup['position'] == pos].shape[0]
                req = self.dk_settings['positions'][pos]
                violations.append(f"{pos}: {count} players (need {req['min']}-{req['max']})")
        
        # Check salary cap
        if not validation['salary_cap']:
            violations.append(f"Salary: ${lineup['salary'].sum():,} (cap: ${self.dk_settings['salary_cap']:,})")
        
        # Check roster size
        if not validation['roster_size']:
            violations.append(f"Roster size: {len(lineup)} (need {self.dk_settings['roster_size']})")
        
        return violations
    
    def suggest_constraint_fixes(self, lineup: pd.DataFrame, projections: pd.DataFrame) -> List[str]:
        """Suggest specific fixes for constraint violations"""
        suggestions = []
        validation = self.validate_all_constraints(lineup)
        
        # Position fixes
        for pos, valid in validation['position_requirements'].items():
            if not valid:
                count = lineup[lineup['position'] == pos].shape[0]
                req = self.dk_settings['positions'][pos]
                
                if count < req['min']:
                    # Need to add players
                    available = projections[
                        (projections['position'] == pos) & 
                        (~projections['player_name'].isin(lineup['player_name']))
                    ]
                    if not available.empty:
                        best_options = available.nlargest(3, 'projection')[['player_name', 'salary', 'projection']]
                        suggestions.append(f"Add {pos}: {best_options.to_dict('records')}")
                
                elif count > req['max']:
                    # Need to remove players
                    current_players = lineup[lineup['position'] == pos]
                    worst_players = current_players.nsmallest(count - req['max'], 'projection')['player_name'].tolist()
                    suggestions.append(f"Remove {pos}: {worst_players}")
        
        # Salary fixes
        if not validation['salary_cap']:
            overage = validation['total_salary'] - self.dk_settings['salary_cap']
            suggestions.append(f"Reduce salary by ${overage:,}")
            
            # Find highest salary players to potentially replace
            high_salary = lineup.nlargest(3, 'salary')[['player_name', 'position', 'salary']]
            suggestions.append(f"High salary players: {high_salary.to_dict('records')}")
        
        return suggestions
    
    def create_position_constraints(self, prob, player_vars, projections):
        """Create position constraints for PuLP optimization"""
        constraints = {}
        
        for pos, req in self.dk_settings['positions'].items():
            pos_players = projections[projections['position'] == pos]
            pos_vars = [player_vars[i] for i in pos_players.index]
            
            # Min constraint
            if req['min'] > 0:
                min_constraint = prob.addConstraint(
                    sum(pos_vars) >= req['min'],
                    name=f"{pos}_min"
                )
                constraints[f"{pos}_min"] = min_constraint
            
            # Max constraint
            if req['max'] < float('inf'):
                max_constraint = prob.addConstraint(
                    sum(pos_vars) <= req['max'],
                    name=f"{pos}_max"
                )
                constraints[f"{pos}_max"] = max_constraint
        
        return constraints
    
    def create_salary_constraint(self, prob, player_vars, projections):
        """Create salary cap constraint for PuLP optimization"""
        salary_constraint = prob.addConstraint(
            sum(player_vars[i] * projections.loc[i, 'salary'] for i in projections.index) <= self.dk_settings['salary_cap'],
            name="salary_cap"
        )
        return salary_constraint
    
    def create_roster_size_constraint(self, prob, player_vars):
        """Create roster size constraint for PuLP optimization"""
        roster_constraint = prob.addConstraint(
            sum(player_vars) == self.dk_settings['roster_size'],
            name="roster_size"
        )
        return roster_constraint
    
    def get_available_players_by_position(self, projections: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Get available players grouped by position"""
        available = {}
        for pos in self.dk_settings['positions'].keys():
            pos_players = projections[projections['position'] == pos]
            available[pos] = pos_players.sort_values('projection', ascending=False)
        
        return available
    
    def calculate_lineup_value(self, lineup: pd.DataFrame) -> Dict[str, float]:
        """Calculate various value metrics for a lineup"""
        total_projection = lineup['projection'].sum()
        total_salary = lineup['salary'].sum()
        avg_projection = total_projection / len(lineup)
        avg_salary = total_salary / len(lineup)
        
        # Value per dollar
        value_per_dollar = total_projection / total_salary if total_salary > 0 else 0
        
        # Position-specific metrics
        pos_metrics = {}
        for pos in self.dk_settings['positions'].keys():
            pos_players = lineup[lineup['position'] == pos]
            if not pos_players.empty:
                pos_metrics[pos] = {
                    'count': len(pos_players),
                    'avg_projection': pos_players['projection'].mean(),
                    'avg_salary': pos_players['salary'].mean(),
                    'total_projection': pos_players['projection'].sum()
                }
        
        return {
            'total_projection': total_projection,
            'total_salary': total_salary,
            'avg_projection': avg_projection,
            'avg_salary': avg_salary,
            'value_per_dollar': value_per_dollar,
            'position_metrics': pos_metrics,
            'salary_efficiency': (total_projection / total_salary) * 1000  # Points per $1000
        }
    
    def validate_lineup_for_export(self, lineup: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate lineup is ready for export to DraftKings"""
        validation = self.validate_all_constraints(lineup)
        violations = self.get_constraint_violations(lineup)
        
        is_valid = validation['all_valid']
        
        if is_valid:
            # Additional checks for export readiness
            required_cols = ['player_name', 'position', 'team', 'salary', 'projection']
            missing_cols = [col for col in required_cols if col not in lineup.columns]
            
            if missing_cols:
                violations.append(f"Missing columns for export: {missing_cols}")
                is_valid = False
            
            # Check for duplicate players
            if lineup['player_name'].duplicated().any():
                violations.append("Duplicate players in lineup")
                is_valid = False
        
        return is_valid, violations
