"""
Maps DraftKings players to your registry.
This solves your matching problem forever.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from difflib import SequenceMatcher
import re
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DraftKingsMapper:
    """
    Maps DraftKings player IDs to the master player registry.
    This creates the bridge between DK's naming conventions and your standardized player identities.
    """
    
    def __init__(self, registry_path: str = "data/registry/dim_players.parquet"):
        self.registry_path = Path(registry_path)
        self.registry = None
        self.dk_players = None
        self.mappings = None
        
        # Load the registry
        self.load_registry()
        
    def load_registry(self):
        """Load the master player registry"""
        try:
            self.registry = pd.read_parquet(self.registry_path)
            logger.info(f"Loaded registry with {len(self.registry)} players")
        except Exception as e:
            logger.error(f"Failed to load registry: {e}")
            raise
    
    def load_dk_players(self, dk_data: pd.DataFrame):
        """
        Load DraftKings player data.
        
        Args:
            dk_data: DataFrame with DK player information
                    Expected columns: ['dk_player_id', 'dk_player_name', 'position', 'team', ...]
        """
        self.dk_players = dk_data.copy()
        
        # Normalize DK names for matching
        self.dk_players['dk_name_normalized'] = self.dk_players['dk_player_name'].apply(
            self.normalize_name
        )
        
        logger.info(f"Loaded {len(self.dk_players)} DK players")
        return self.dk_players
    
    def normalize_name(self, name: str) -> str:
        """
        Normalize player names for better matching.
        Removes special characters, converts to lowercase, handles common variations.
        """
        if pd.isna(name):
            return ""
        
        # Convert to string and lowercase
        name = str(name).lower().strip()
        
        # Remove special characters but keep spaces
        name = re.sub(r'[^a-z\s]', '', name)
        
        # Handle common name variations
        name = re.sub(r'\s+', ' ', name)  # Multiple spaces to single space
        
        # Common abbreviations
        name = name.replace('jr', '')
        name = name.replace('sr', '')
        name = name.replace('ii', '')
        name = name.replace('iii', '')
        name = name.replace('iv', '')
        
        return name.strip()
    
    def exact_match(self, dk_name: str, dk_position: str, dk_team: str) -> Optional[str]:
        """
        Try exact match on normalized name + position + team.
        Returns player_uid if found, None otherwise.
        """
        dk_normalized = self.normalize_name(dk_name)
        
        # Filter registry by position and team
        candidates = self.registry[
            (self.registry['position'] == dk_position) &
            (self.registry['team'] == dk_team)
        ]
        
        # Try exact name match
        exact_matches = candidates[
            candidates['name_normalized'] == dk_normalized
        ]
        
        if len(exact_matches) == 1:
            return exact_matches.iloc[0]['player_uid']
        elif len(exact_matches) > 1:
            logger.warning(f"Multiple exact matches for {dk_name} ({dk_position}, {dk_team})")
            return None
        
        return None
    
    def fuzzy_match(self, dk_name: str, dk_position: str, dk_team: str, 
                   threshold: float = 0.85) -> Tuple[Optional[str], float]:
        """
        Try fuzzy matching when exact match fails.
        Returns (player_uid, confidence_score) or (None, 0.0)
        """
        dk_normalized = self.normalize_name(dk_name)
        
        # Filter by position and team
        candidates = self.registry[
            (self.registry['position'] == dk_position) &
            (self.registry['team'] == dk_team)
        ]
        
        if len(candidates) == 0:
            return None, 0.0
        
        # Calculate similarity scores
        best_match = None
        best_score = 0.0
        
        for _, candidate in candidates.iterrows():
            registry_name = candidate['name_normalized']
            
            # Use SequenceMatcher for string similarity
            similarity = SequenceMatcher(None, dk_normalized, registry_name).ratio()
            
            if similarity > best_score and similarity >= threshold:
                best_score = similarity
                best_match = candidate['player_uid']
        
        return best_match, best_score
    
    def position_team_match(self, dk_name: str, dk_position: str, dk_team: str,
                           threshold: float = 0.75) -> Tuple[Optional[str], float]:
        """
        Try matching by name only when position/team match fails.
        Useful for players who changed teams or positions.
        """
        dk_normalized = self.normalize_name(dk_name)
        
        # Filter by position only
        candidates = self.registry[
            self.registry['position'] == dk_position
        ]
        
        if len(candidates) == 0:
            return None, 0.0
        
        # Calculate similarity scores
        best_match = None
        best_score = 0.0
        
        for _, candidate in candidates.iterrows():
            registry_name = candidate['name_normalized']
            similarity = SequenceMatcher(None, dk_normalized, registry_name).ratio()
            
            if similarity > best_score and similarity >= threshold:
                best_score = similarity
                best_match = candidate['player_uid']
        
        # Reduce confidence for position-only matches
        adjusted_score = best_score * 0.8
        
        return best_match, adjusted_score
    
    def create_mapping(self, dk_row: pd.Series) -> Dict:
        """
        Create a mapping for a single DK player.
        Tries multiple matching strategies with confidence scores.
        """
        dk_name = dk_row['dk_player_name']
        dk_position = dk_row['dk_position']
        dk_team = dk_row['dk_team']
        dk_id = dk_row['dk_player_id']
        
        # Strategy 1: Exact match
        player_uid = self.exact_match(dk_name, dk_position, dk_team)
        if player_uid:
            return {
                'dk_player_id': dk_id,
                'player_uid': player_uid,
                'dk_player_name': dk_name,
                'match_method': 'exact',
                'confidence_score': 1.0,
                'fallback_used': False
            }
        
        # Strategy 2: Fuzzy match with position + team
        player_uid, confidence = self.fuzzy_match(dk_name, dk_position, dk_team)
        if player_uid:
            return {
                'dk_player_id': dk_id,
                'player_uid': player_uid,
                'dk_player_name': dk_name,
                'match_method': 'fuzzy_position_team',
                'confidence_score': confidence,
                'fallback_used': False
            }
        
        # Strategy 3: Position-only match (lower confidence)
        player_uid, confidence = self.position_team_match(dk_name, dk_position, dk_team)
        if player_uid:
            return {
                'dk_player_id': dk_id,
                'player_uid': player_uid,
                'dk_player_name': dk_name,
                'match_method': 'fuzzy_position_only',
                'confidence_score': confidence,
                'fallback_used': True
            }
        
        # No match found
        return {
            'dk_player_id': dk_id,
            'player_uid': None,
            'dk_player_name': dk_name,
            'match_method': 'no_match',
            'confidence_score': 0.0,
            'fallback_used': False
        }
    
    def build_mappings(self) -> pd.DataFrame:
        """
        Build all DK player mappings.
        Returns DataFrame with mapping information and confidence scores.
        """
        if self.dk_players is None:
            raise ValueError("DK players must be loaded before building mappings")
        
        logger.info("Building DK player mappings...")
        
        mappings = []
        for _, dk_row in self.dk_players.iterrows():
            mapping = self.create_mapping(dk_row)
            mappings.append(mapping)
        
        self.mappings = pd.DataFrame(mappings)
        
        # Add metadata
        self.mappings['mapping_created_date'] = pd.Timestamp.now()
        self.mappings['mapping_version'] = '1.0'
        
        logger.info(f"Created {len(self.mappings)} mappings")
        return self.mappings
    
    def analyze_mappings(self) -> Dict:
        """
        Analyze the quality of mappings.
        Returns summary statistics and identifies potential issues.
        """
        if self.mappings is None:
            raise ValueError("Mappings must be built before analysis")
        
        analysis = {
            'total_dk_players': len(self.mappings),
            'successful_matches': len(self.mappings[self.mappings['player_uid'].notna()]),
            'failed_matches': len(self.mappings[self.mappings['player_uid'].isna()]),
            'exact_matches': len(self.mappings[self.mappings['match_method'] == 'exact']),
            'fuzzy_matches': len(self.mappings[self.mappings['match_method'].str.contains('fuzzy')]),
            'fallback_used': len(self.mappings[self.mappings['fallback_used'] == True]),
            'avg_confidence': self.mappings['confidence_score'].mean(),
            'high_confidence': len(self.mappings[self.mappings['confidence_score'] >= 0.9]),
            'medium_confidence': len(self.mappings[(self.mappings['confidence_score'] >= 0.7) & 
                                                (self.mappings['confidence_score'] < 0.9)]),
            'low_confidence': len(self.mappings[self.mappings['confidence_score'] < 0.7])
        }
        
        # Log analysis results
        logger.info("Mapping Analysis:")
        for metric, value in analysis.items():
            logger.info(f"  {metric}: {value}")
        
        return analysis
    
    def save_mappings(self, output_path: str = "../data/registry/map_dk.csv"):
        """
        Save the DK mappings to CSV file.
        """
        if self.mappings is None:
            raise ValueError("Mappings must be built before saving")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save mappings
        self.mappings.to_csv(output_path, index=False)
        logger.info(f"Saved {len(self.mappings)} mappings to {output_path}")
        
        # Also save high-confidence mappings separately
        high_conf_path = output_path.parent / "map_dk_high_confidence.csv"
        high_conf = self.mappings[self.mappings['confidence_score'] >= 0.8]
        high_conf.to_csv(high_conf_path, index=False)
        logger.info(f"Saved {len(high_conf)} high-confidence mappings to {high_conf_path}")
        
        return output_path
    
    def get_unmatched_players(self) -> pd.DataFrame:
        """
        Get list of DK players that couldn't be matched.
        Useful for manual review and registry updates.
        """
        if self.mappings is None:
            raise ValueError("Mappings must be built first")
        
        unmatched = self.mappings[self.mappings['player_uid'].isna()].copy()
        
        if len(unmatched) > 0:
            logger.warning(f"Found {len(unmatched)} unmatched DK players")
            logger.warning("Review these manually and update registry if needed")
        
        return unmatched
    
    def match_slate(self, dk_slate: pd.DataFrame) -> pd.DataFrame:
        """
        Match a DraftKings slate to the registry.
        This is the main method for testing with real DK data.
        
        Args:
            dk_slate: DataFrame with DK player data
                     Expected columns: ['Name', 'Position', 'Team'] or similar
        """
        # Standardize column names for DK data
        column_mapping = {
            'Name': 'dk_player_name',
            'Position': 'dk_position', 
            'Team': 'dk_team',
            'TeamAbbrev': 'dk_team',  # DK uses TeamAbbrev
            'player_name': 'dk_player_name',
            'position': 'dk_position',
            'team': 'dk_team'
        }
        
        # Rename columns if they exist
        for old_col, new_col in column_mapping.items():
            if old_col in dk_slate.columns:
                dk_slate = dk_slate.rename(columns={old_col: new_col})
        
        # Ensure required columns exist
        required_cols = ['dk_player_name', 'dk_position', 'dk_team']
        missing_cols = [col for col in required_cols if col not in dk_slate.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}. Available columns: {list(dk_slate.columns)}")
        
        # Create DK player ID if not present
        if 'dk_player_id' not in dk_slate.columns:
            dk_slate['dk_player_id'] = dk_slate.apply(
                lambda row: f"DK_{row['dk_position']}_{row.name:03d}", axis=1
            )
        
        # Load DK players and build mappings
        self.load_dk_players(dk_slate)
        mappings = self.build_mappings()
        
        return mappings


def main():
    """
    Example usage of the DK Mapper.
    """
    logger.info("=" * 60)
    logger.info("DRAFTKINGS PLAYER MAPPER")
    logger.info("=" * 60)
    
    # Initialize mapper
    mapper = DraftKingsMapper()
    
    # Example DK data structure (replace with your actual DK data)
    sample_dk_data = pd.DataFrame([
        {
            'dk_player_id': 'DK_QB_001',
            'dk_player_name': 'Patrick Mahomes',
            'dk_position': 'QB',
            'dk_team': 'KC'
        },
        {
            'dk_player_id': 'DK_RB_001',
            'dk_player_name': 'Christian McCaffrey',
            'dk_position': 'RB',
            'dk_team': 'SF'
        }
    ])
    
    # Load DK players
    mapper.load_dk_players(sample_dk_data)
    
    # Build mappings
    mappings = mapper.build_mappings()
    
    # Analyze results
    analysis = mapper.analyze_mappings()
    
    # Save mappings
    output_path = mapper.save_mappings()
    
    # Show unmatched players
    unmatched = mapper.get_unmatched_players()
    if len(unmatched) > 0:
        print("\nUnmatched players:")
        print(unmatched[['dk_player_name', 'dk_position', 'dk_team']])
    
    logger.info("DK mapping completed!")


if __name__ == "__main__":
    main()
