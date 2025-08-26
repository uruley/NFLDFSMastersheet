# optimizer/validator.py
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Set
from stacking import StackingConstraints
from constraints import DraftKingsConstraints

class LineupValidator:
    def __init__(self, projections: pd.DataFrame, data_dir='data'):
        """Initialize lineup validator with projections and constraints"""
        self.projections = projections
        self.stacking_constraints = StackingConstraints(projections)
        self.dk_constraints = DraftKingsConstraints(data_dir)
        
    def validate_complete_lineup(self, lineup: pd.DataFrame) -> Dict[str, any]:
        """Comprehensive validation of a DFS lineup"""
        results = {
            'basic_constraints': self.dk_constraints.validate_all_constraints(lineup),
            'stacking_validation': self.stacking_constraints.get_stack_info(lineup),
            'stacking_score': self.stacking_constraints.calculate_stacking_score(lineup, self.projections),
            'export_ready': self.dk_constraints.validate_lineup_for_export(lineup)
        }
        
        # Overall validation
        all_valid = (
            results['basic_constraints']['all_valid'] and
            results['export_ready'][0]
        )
        results['all_valid'] = all_valid
        
        return results
    
    def get_all_violations(self, lineup: pd.DataFrame) -> List[str]:
        """Get all constraint violations for a lineup"""
        violations = []
        
        # Basic constraint violations
        basic_violations = self.dk_constraints.get_constraint_violations(lineup)
        violations.extend(basic_violations)
        
        # Stacking violations
        stacking_validation = self.stacking_constraints.validate_stacking_rules(lineup, self.projections)
        for rule, passed in stacking_validation.items():
            if not passed:
                violations.append(f"Stacking: {rule.replace('_', ' ').title()} rule violated")
        
        # Export violations
        export_valid, export_violations = self.dk_constraints.validate_lineup_for_export(lineup)
        violations.extend(export_violations)
        
        return violations
    
    def suggest_all_fixes(self, lineup: pd.DataFrame) -> Dict[str, List[str]]:
        """Get suggestions for fixing all types of violations"""
        suggestions = {
            'constraint_fixes': self.dk_constraints.suggest_constraint_fixes(lineup, self.projections),
            'stacking_improvements': self.stacking_constraints.suggest_stacking_improvements(lineup, self.projections)
        }
        return suggestions
    
    def calculate_lineup_metrics(self, lineup: pd.DataFrame) -> Dict[str, any]:
        """Calculate comprehensive metrics for a lineup"""
        metrics = {
            'basic_metrics': self.dk_constraints.calculate_lineup_value(lineup),
            'stacking_metrics': self.stacking_constraints.get_stacking_metrics(lineup, self.projections),
            'stacking_score': self.stacking_constraints.calculate_stacking_score(lineup, self.projections)
        }
        
        # Combine metrics
        combined_metrics = {
            'total_projection': metrics['basic_metrics']['total_projection'],
            'total_salary': metrics['basic_metrics']['total_salary'],
            'value_per_dollar': metrics['basic_metrics']['value_per_dollar'],
            'salary_efficiency': metrics['basic_metrics']['salary_efficiency'],
            'stacking_score': metrics['stacking_score'],
            'stacking_quality': self._get_stacking_quality_label(metrics['stacking_score']),
            'position_distribution': metrics['basic_metrics']['position_metrics'],
            'team_distribution': metrics['stacking_metrics']['team_distribution'],
            'game_stack_info': metrics['stacking_metrics']['main_game_stack']
        }
        
        return combined_metrics
    
    def _get_stacking_quality_label(self, score: float) -> str:
        """Convert stacking score to quality label"""
        if score >= 80:
            return "Excellent"
        elif score >= 60:
            return "Good"
        elif score >= 40:
            return "Fair"
        elif score >= 20:
            return "Poor"
        else:
            return "Very Poor"
    
    def print_validation_report(self, lineup: pd.DataFrame):
        """Print a comprehensive validation report"""
        print("\n" + "="*80)
        print("COMPREHENSIVE LINEUP VALIDATION REPORT")
        print("="*80)
        
        # Basic validation
        validation = self.validate_complete_lineup(lineup)
        basic_constraints = validation['basic_constraints']
        
        print("\n📋 BASIC CONSTRAINTS:")
        print(f"  Position Requirements: {'✓' if all(basic_constraints['position_requirements'].values()) else '✗'}")
        print(f"  Salary Cap: {'✓' if basic_constraints['salary_cap'] else '✗'}")
        print(f"  Roster Size: {'✓' if basic_constraints['roster_size'] else '✗'}")
        print(f"  Total Salary: ${basic_constraints['total_salary']:,}")
        print(f"  Salary Remaining: ${basic_constraints['salary_remaining']:,}")
        
        # Stacking validation
        print("\n🎯 STACKING ANALYSIS:")
        stacking_score = validation['stacking_score']
        print(f"  Stacking Score: {stacking_score:.1f}/100 ({self._get_stacking_quality_label(stacking_score)})")
        
        stacking_info = validation['stacking_validation']
        if stacking_info['qb_wr_stacks']:
            print("  QB-WR/TE Stacks:")
            for stack in stacking_info['qb_wr_stacks']:
                print(f"    {stack['qb']} + {', '.join(stack['wr_te'])} ({stack['team']})")
        else:
            print("  ⚠️  No QB-WR/TE stacks found")
        
        if stacking_info['game_stacks']:
            print("  Game Stacks:")
            for stack in stacking_info['game_stacks']:
                print(f"    {stack['game_id']}: {', '.join(stack['players'])}")
        else:
            print("  ⚠️  No game stacks found")
        
        if stacking_info['bring_backs']:
            print("  Bring-Back Players:")
            for bring_back in stacking_info['bring_backs']:
                print(f"    {bring_back['game_id']}: {', '.join(bring_back['team1_players'])} vs {', '.join(bring_back['team2_players'])}")
        
        # Metrics
        print("\n📊 LINEUP METRICS:")
        metrics = self.calculate_lineup_metrics(lineup)
        print(f"  Total Projection: {metrics['total_projection']:.2f} pts")
        print(f"  Value per Dollar: {metrics['value_per_dollar']:.3f}")
        print(f"  Salary Efficiency: {metrics['salary_efficiency']:.1f} pts/$1000")
        
        # Position distribution
        print("\n👥 POSITION DISTRIBUTION:")
        for pos, pos_metrics in metrics['position_distribution'].items():
            print(f"  {pos}: {pos_metrics['count']} players, "
                  f"avg {pos_metrics['avg_projection']:.1f} pts, "
                  f"avg ${pos_metrics['avg_salary']:,.0f}")
        
        # Team distribution
        print("\n🏈 TEAM DISTRIBUTION:")
        for team, count in metrics['team_distribution'].items():
            print(f"  {team}: {count} players")
        
        # Game stack info
        if metrics['game_stack_info']:
            game_id, game_data = metrics['game_stack_info']
            print(f"\n🎮 MAIN GAME STACK: {game_id}")
            print(f"  Teams: {', '.join(game_data['teams'])}")
            print(f"  Players: {', '.join(game_data['players'])}")
            print(f"  Total Players: {game_data['player_count']}")
        
        # Overall validation
        print("\n" + "="*80)
        if validation['all_valid']:
            print("✅ LINEUP IS VALID AND READY FOR EXPORT!")
        else:
            print("❌ LINEUP HAS VALIDATION ISSUES")
            violations = self.get_all_violations(lineup)
            if violations:
                print("\nViolations found:")
                for violation in violations:
                    print(f"  • {violation}")
        
        print("="*80)
        
        return validation
    
    def export_validation_summary(self, lineup: pd.DataFrame, filename: str = None) -> str:
        """Export validation summary to CSV"""
        if filename is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"lineup_validation_{timestamp}.csv"
        
        # Get all validation data
        validation = self.validate_complete_lineup(lineup)
        metrics = self.calculate_lineup_metrics(lineup)
        
        # Create summary DataFrame
        summary_data = {
            'Metric': [
                'Total Projection', 'Total Salary', 'Salary Remaining',
                'Value per Dollar', 'Salary Efficiency', 'Stacking Score',
                'Stacking Quality', 'Position Count QB', 'Position Count RB',
                'Position Count WR', 'Position Count TE', 'Position Count DST',
                'Max Players per Team', 'Game Stack Count', 'QB-WR Stack Count'
            ],
            'Value': [
                metrics['total_projection'],
                metrics['total_salary'],
                validation['basic_constraints']['salary_remaining'],
                metrics['value_per_dollar'],
                metrics['salary_efficiency'],
                metrics['stacking_score'],
                metrics['stacking_quality'],
                metrics['position_distribution'].get('QB', {}).get('count', 0),
                metrics['position_distribution'].get('RB', {}).get('count', 0),
                metrics['position_distribution'].get('WR', {}).get('count', 0),
                metrics['position_distribution'].get('TE', {}).get('count', 0),
                metrics['position_distribution'].get('DST', {}).get('count', 0),
                max(metrics['team_distribution'].values()) if metrics['team_distribution'] else 0,
                len(validation['stacking_validation']['game_stacks']),
                len(validation['stacking_validation']['qb_wr_stacks'])
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save to CSV
        csv_path = f"output/{filename}"
        summary_df.to_csv(csv_path, index=False)
        print(f"\n💾 Validation summary exported to: {csv_path}")
        
        return csv_path

def test_validator():
    """Test the validator with sample data"""
    # Create sample projections
    sample_projections = pd.DataFrame({
        'player_name': ['Josh Allen', 'Stefon Diggs', 'Gabe Davis', 'Dalvin Cook', 'Justin Jefferson'],
        'team': ['BUF', 'BUF', 'BUF', 'MIN', 'MIN'],
        'position': ['QB', 'WR', 'WR', 'RB', 'WR'],
        'salary': [8000, 7500, 4500, 6000, 8000],
        'projection': [25.0, 20.0, 12.0, 15.0, 22.0],
        'adjusted_projection': [25.0, 20.0, 12.0, 15.0, 22.0]
    })
    
    # Create sample lineup
    sample_lineup = sample_projections.head(5).copy()
    
    # Test validator
    validator = LineupValidator(sample_projections)
    validator.print_validation_report(sample_lineup)
    
    return validator

if __name__ == "__main__":
    validator = test_validator()
