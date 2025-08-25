"""
Test Identity System
Test with a real DraftKings CSV
Verify you can match 95%+ of players
Document any failures for manual review
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from identity.build_registry import build_player_registry
from identity.dk_mapper import DraftKingsMapper

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('identity_test.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class IdentitySystemTester:
    """
    Comprehensive tester for the identity system.
    Tests player registry building and DK mapping with real data.
    """
    
    def __init__(self, test_data_dir: str = "../data/staging/current_week"):
        self.test_data_dir = Path(test_data_dir)
        self.test_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Test results
        self.test_results = {
            'registry_build': {},
            'dk_mapping': {},
            'overall_success': False
        }
        
    def test_registry_build(self):
        """Test 1: Build the player registry"""
        logger.info("=" * 60)
        logger.info("TEST 1: BUILDING PLAYER REGISTRY")
        logger.info("=" * 60)
        
        try:
            # Build registry
            start_time = datetime.now()
            registry = build_player_registry()
            end_time = datetime.now()
            
            # Validate registry
            validation_results = self._validate_registry(registry)
            
            self.test_results['registry_build'] = {
                'success': True,
                'total_players': len(registry),
                'build_time': (end_time - start_time).total_seconds(),
                'validation': validation_results
            }
            
            logger.info(f"✅ Registry build SUCCESS: {len(registry)} players")
            logger.info(f"Build time: {self.test_results['registry_build']['build_time']:.2f} seconds")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Registry build FAILED: {str(e)}")
            self.test_results['registry_build'] = {
                'success': False,
                'error': str(e)
            }
            return False
    
    def _validate_registry(self, registry: pd.DataFrame) -> dict:
        """Validate the built registry for data quality"""
        validation = {
            'total_players': len(registry),
            'unique_uuids': registry['player_uid'].nunique(),
            'missing_names': registry['display_name'].isna().sum(),
            'missing_positions': registry['position'].isna().sum(),
            'missing_teams': registry['team'].isna().sum(),
            'position_distribution': registry['position'].value_counts().to_dict(),
            'team_distribution': registry['team'].value_counts().to_dict()
        }
        
        # Check for critical issues
        validation['uuid_uniqueness'] = validation['total_players'] == validation['unique_uuids']
        validation['no_missing_names'] = validation['missing_names'] == 0
        validation['no_missing_positions'] = validation['missing_positions'] == 0
        validation['no_missing_teams'] = validation['missing_teams'] == 0
        
        return validation
    
    def test_dk_mapping(self, dk_csv_path: str = None):
        """Test 2: Test DK mapping with real data"""
        logger.info("=" * 60)
        logger.info("TEST 2: TESTING DK MAPPING")
        logger.info("=" * 60)
        
        try:
            # Initialize mapper
            mapper = DraftKingsMapper()
            
            # Load DK data (real or sample)
            if dk_csv_path and Path(dk_csv_path).exists():
                dk_data = self._load_real_dk_data(dk_csv_path)
                logger.info(f"Loaded real DK data: {len(dk_data)} players")
            else:
                dk_data = self._create_sample_dk_data()
                logger.info(f"Using sample DK data: {len(dk_data)} players")
            
            # Test mapping
            start_time = datetime.now()
            mapper.load_dk_players(dk_data)
            mappings = mapper.build_mappings()
            end_time = datetime.now()
            
            # Analyze results
            analysis = mapper.analyze_mappings()
            
            # Calculate success rate
            success_rate = (analysis['successful_matches'] / analysis['total_dk_players']) * 100
            
            self.test_results['dk_mapping'] = {
                'success': True,
                'total_dk_players': analysis['total_dk_players'],
                'successful_matches': analysis['successful_matches'],
                'success_rate': success_rate,
                'mapping_time': (end_time - start_time).total_seconds(),
                'analysis': analysis
            }
            
            logger.info(f"✅ DK mapping SUCCESS: {success_rate:.1f}% match rate")
            logger.info(f"Mapping time: {self.test_results['dk_mapping']['mapping_time']:.2f} seconds")
            
            # Check if we meet the 95% threshold
            if success_rate >= 95.0:
                logger.info("🎯 TARGET ACHIEVED: 95%+ match rate!")
            else:
                logger.warning(f"⚠️  TARGET MISSED: {success_rate:.1f}% (need 95%+)")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ DK mapping FAILED: {str(e)}")
            self.test_results['dk_mapping'] = {
                'success': False,
                'error': str(e)
            }
            return False
    
    def _load_real_dk_data(self, csv_path: str) -> pd.DataFrame:
        """Load real DK data from CSV"""
        try:
            dk_data = pd.read_csv(csv_path)
            
            # Standardize column names
            column_mapping = {
                'Name': 'dk_player_name',
                'Position': 'dk_position',
                'Team': 'dk_team',
                'ID': 'dk_player_id',
                'player_name': 'dk_player_name',
                'position': 'dk_position',
                'team': 'dk_team',
                'player_id': 'dk_player_id'
            }
            
            dk_data = dk_data.rename(columns=column_mapping)
            
            # Ensure required columns exist
            required_cols = ['dk_player_name', 'dk_position', 'dk_team']
            missing_cols = [col for col in required_cols if col not in dk_data.columns]
            
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Create DK player ID if not present
            if 'dk_player_id' not in dk_data.columns:
                dk_data['dk_player_id'] = dk_data.apply(
                    lambda row: f"DK_{row['dk_position']}_{row.name:03d}", axis=1
                )
            
            return dk_data
            
        except Exception as e:
            logger.error(f"Failed to load DK data: {e}")
            raise
    
    def _create_sample_dk_data(self) -> pd.DataFrame:
        """Create sample DK data for testing when real data isn't available"""
        sample_data = [
            {'dk_player_name': 'Patrick Mahomes', 'dk_position': 'QB', 'dk_team': 'KC'},
            {'dk_player_name': 'Christian McCaffrey', 'dk_position': 'RB', 'dk_team': 'SF'},
            {'dk_player_name': 'Tyreek Hill', 'dk_position': 'WR', 'dk_team': 'MIA'},
            {'dk_player_name': 'Travis Kelce', 'dk_position': 'TE', 'dk_team': 'KC'},
            {'dk_player_name': 'Justin Tucker', 'dk_position': 'K', 'dk_team': 'BAL'},
            {'dk_player_name': 'San Francisco 49ers', 'dk_position': 'DST', 'dk_team': 'SF'},
            {'dk_player_name': 'Josh Allen', 'dk_position': 'QB', 'dk_team': 'BUF'},
            {'dk_player_name': 'Austin Ekeler', 'dk_position': 'RB', 'dk_team': 'LAC'},
            {'dk_player_name': 'Stefon Diggs', 'dk_position': 'WR', 'dk_team': 'HOU'},
            {'dk_player_name': 'Mark Andrews', 'dk_position': 'TE', 'dk_team': 'BAL'}
        ]
        
        df = pd.DataFrame(sample_data)
        df['dk_player_id'] = df.apply(
            lambda row: f"DK_{row['dk_position']}_{row.name:03d}", axis=1
        )
        
        return df
    
    def document_failures(self):
        """Document any mapping failures for manual review"""
        logger.info("=" * 60)
        logger.info("DOCUMENTING FAILURES FOR MANUAL REVIEW")
        logger.info("=" * 60)
        
        try:
            mapper = DraftKingsMapper()
            
            # Get unmatched players
            unmatched = mapper.get_unmatched_players()
            
            if len(unmatched) > 0:
                # Save unmatched players to CSV for manual review
                unmatched_path = self.test_data_dir / "unmatched_dk_players.csv"
                unmatched.to_csv(unmatched_path, index=False)
                
                logger.warning(f"⚠️  {len(unmatched)} unmatched players saved to {unmatched_path}")
                logger.warning("Review these manually and update registry if needed")
                
                # Show sample of unmatched players
                print("\nSample unmatched players:")
                print(unmatched[['dk_player_name', 'dk_position', 'dk_team']].head(10))
                
                # Analyze failure patterns
                self._analyze_failure_patterns(unmatched)
            else:
                logger.info("✅ No unmatched players found!")
                
        except Exception as e:
            logger.error(f"Failed to document failures: {e}")
    
    def _analyze_failure_patterns(self, unmatched: pd.DataFrame):
        """Analyze patterns in failed matches"""
        logger.info("\nFailure Pattern Analysis:")
        
        # Position breakdown
        pos_failures = unmatched['dk_position'].value_counts()
        logger.info("Failures by position:")
        for pos, count in pos_failures.items():
            logger.info(f"  {pos}: {count}")
        
        # Team breakdown
        team_failures = unmatched['dk_team'].value_counts()
        logger.info("Failures by team:")
        for team, count in team_failures.head(10).items():
            logger.info(f"  {team}: {count}")
        
        # Name length analysis
        unmatched['name_length'] = unmatched['dk_player_name'].str.len()
        avg_length = unmatched['name_length'].mean()
        logger.info(f"Average name length of failures: {avg_length:.1f}")
    
    def run_full_test(self, dk_csv_path: str = None):
        """Run the complete identity system test"""
        logger.info("🚀 STARTING IDENTITY SYSTEM TEST")
        logger.info("=" * 80)
        
        # Test 1: Registry build
        registry_success = self.test_registry_build()
        
        if not registry_success:
            logger.error("❌ Registry build failed. Stopping tests.")
            return False
        
        # Test 2: DK mapping
        mapping_success = self.test_dk_mapping(dk_csv_path)
        
        if not mapping_success:
            logger.error("❌ DK mapping failed.")
            return False
        
        # Document failures
        self.document_failures()
        
        # Overall assessment
        success_rate = self.test_results['dk_mapping']['success_rate']
        if success_rate >= 95.0:
            self.test_results['overall_success'] = True
            logger.info("🎉 OVERALL TEST RESULT: SUCCESS!")
            logger.info(f"✅ Achieved {success_rate:.1f}% match rate (target: 95%+)")
        else:
            logger.warning("⚠️  OVERALL TEST RESULT: PARTIAL SUCCESS")
            logger.warning(f"❌ Only achieved {success_rate:.1f}% match rate (target: 95%+)")
        
        # Save test results
        self._save_test_results()
        
        return self.test_results['overall_success']
    
    def _save_test_results(self):
        """Save detailed test results to file"""
        results_path = self.test_data_dir / "identity_test_results.json"
        
        # Convert timestamps to strings for JSON serialization
        results_copy = self.test_results.copy()
        if 'dk_mapping' in results_copy and 'analysis' in results_copy['dk_mapping']:
            analysis = results_copy['dk_mapping']['analysis']
            if 'mapping_created_date' in analysis:
                analysis['mapping_created_date'] = str(analysis['mapping_created_date'])
        
        # Save as JSON
        import json
        with open(results_path, 'w') as f:
            json.dump(results_copy, f, indent=2, default=str)
        
        logger.info(f"Test results saved to {results_path}")


def main():
    """Main test execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test the Identity System')
    parser.add_argument('--dk-csv', type=str, help='Path to real DK CSV file for testing')
    parser.add_argument('--test-dir', type=str, default='../data/staging/current_week',
                       help='Directory to save test results')
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = IdentitySystemTester(args.test_dir)
    
    # Run tests
    success = tester.run_full_test(args.dk_csv)
    
    if success:
        print("\n🎉 IDENTITY SYSTEM TEST: PASSED!")
        print("Your identity system is working correctly and can match 95%+ of players!")
    else:
        print("\n❌ IDENTITY SYSTEM TEST: FAILED!")
        print("Check the logs for details on what went wrong.")
    
    return success


if __name__ == "__main__":
    main()

