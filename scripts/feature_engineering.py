"""
Unified Feature Engineering Module for NFL DFS Pipeline

This module ensures consistent feature engineering between training and inference
by providing the same calculation logic for all derived features.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


def calculate_derived_features(player_features: Dict, position: str, team: str, 
                             current_week: int) -> Dict[str, float]:
    """
    Calculate all derived features consistently between training and inference.
    
    Args:
        player_features: Dictionary of raw player stats
        position: Player position (QB, RB, WR, TE)
        team: Team abbreviation
        current_week: Current NFL week
        
    Returns:
        Dictionary of calculated features
    """
    features = {}
    
    # Extract raw stats with safe defaults
    passing_yards = player_features.get('passing_yards', 0)
    passing_tds = player_features.get('passing_tds', 0)
    interceptions = player_features.get('interceptions', 0)
    rushing_yards = player_features.get('rushing_yards', 0)
    rushing_tds = player_features.get('rushing_tds', 0)
    receptions = player_features.get('receptions', 0)
    receiving_yards = player_features.get('receiving_yards', 0)
    receiving_tds = player_features.get('receiving_tds', 0)
    fumbles_lost = player_features.get('fumbles_lost', 0)
    carries = player_features.get('carries', 0)
    targets = player_features.get('targets', 0)
    
    # Base features (only the ones the corrected model expects)
    touches = carries + targets
    features['total_touches'] = float(touches)
    
    # Touchdown rate (avoid division by zero and cap unrealistic values)
    total_tds = passing_tds + rushing_tds + receiving_tds
    if touches > 0:
        # Cap TD rate at 50% to prevent model from going crazy
        # A 50% TD rate is already extremely high (1 TD per 2 touches)
        raw_td_rate = float(total_tds) / float(touches)
        features['touchdown_rate'] = min(raw_td_rate, 0.5)
    else:
        features['touchdown_rate'] = 0.0
    
    # Yards per touch (avoid division by zero and cap unrealistic values)
    total_yards = passing_yards + rushing_yards + receiving_yards
    if touches > 0:
        # Cap yards per touch at 50 to prevent model from going crazy
        # 50 yards per touch is already extremely high
        raw_ypt = float(total_yards) / float(touches)
        features['yards_per_touch'] = min(raw_ypt, 50.0)
    else:
        features['yards_per_touch'] = 0.0
    
    # Carry efficiency (avoid division by zero)
    if carries > 0:
        features['carry_efficiency'] = float(rushing_yards) / float(carries)
    else:
        features['carry_efficiency'] = 0.0
    
    # Target efficiency (avoid division by zero)
    if targets > 0:
        features['target_efficiency'] = float(receiving_yards) / float(targets)
    else:
        features['target_efficiency'] = 0.0
    
    # Position features
    features['position_QB'] = 1.0 if position == 'QB' else 0.0
    features['position_RB'] = 1.0 if position == 'RB' else 0.0
    features['position_WR'] = 1.0 if position == 'WR' else 0.0
    features['position_TE'] = 1.0 if position == 'TE' else 0.0
    
    features['is_qb'] = 1.0 if position == 'QB' else 0.0
    features['is_skill'] = 1.0 if position in ['RB', 'WR', 'TE'] else 0.0
    
    # Conference feature (placeholder - you might want to load actual conference data)
    features['conference_afc'] = 0.5  # Neutral value
    
    # Team features (will be filled by the calling function)
    # We don't create them here since we need to know all possible teams
    
    # Temporal features
    features['week_early'] = 1.0 if current_week <= 4 else 0.0
    features['week_mid'] = 1.0 if 5 <= current_week <= 12 else 0.0
    features['week_late'] = 1.0 if current_week >= 13 else 0.0
    features['week_progression'] = float(current_week) / 18.0  # Normalize to 0-1
    features['weeks_since_bye'] = 0.0  # Placeholder
    
    # Optional features (only if the model expects them)
    features['breakaway_rate'] = 0.0  # Placeholder - you'd need actual data
    
    # Catch rate (avoid division by zero)
    if targets > 0:
        features['catch_rate'] = float(receptions) / float(targets)
    else:
        features['catch_rate'] = 0.0
    
    # Completion rate (avoid division by zero)
    attempts = player_features.get('passing_attempts', 0)
    completions = player_features.get('passing_completions', 0)
    if attempts > 0:
        features['completion_rate'] = float(completions) / float(attempts)
    else:
        features['completion_rate'] = 0.0
    
    # Air yards per attempt (placeholder)
    features['air_yards_per_attempt'] = 0.0
    
    return features


def add_team_features(features: Dict[str, float], team: str, all_teams: List[str]) -> Dict[str, float]:
    """
    Add team one-hot encoding features.
    
    Args:
        features: Existing features dictionary
        team: Current team abbreviation
        all_teams: List of all possible teams
        
    Returns:
        Updated features dictionary with team features
    """
    for team_abbr in all_teams:
        feature_name = f'team_{team_abbr}'
        features[feature_name] = 1.0 if team_abbr == team else 0.0
    
    return features


def scale_features(features: Dict[str, float], feature_stats: Dict) -> Dict[str, float]:
    """
    Scale features using training statistics to match training distribution.
    
    Args:
        features: Raw features dictionary
        feature_stats: Training statistics from feature_stats.json
        
    Returns:
        Scaled features dictionary
    """
    if not feature_stats:
        return features
    
    scaled_features = features.copy()
    
    for feature_name, value in features.items():
        if feature_name in feature_stats.get('means', {}):
            mean = feature_stats['means'][feature_name]
            std = feature_stats['stds'][feature_name]
            if std > 0:  # Avoid division by zero
                scaled_features[feature_name] = (value - mean) / std
    
    return scaled_features


def create_feature_vector(features: Dict[str, float], feature_names: List[str]) -> np.ndarray:
    """
    Create feature vector in the order expected by the model.
    
    Args:
        features: Features dictionary
        feature_names: Ordered list of feature names expected by model
        
    Returns:
        numpy array of features in correct order
    """
    feature_vector = np.zeros(len(feature_names))
    
    for i, feature_name in enumerate(feature_names):
        if feature_name in features:
            feature_vector[i] = features[feature_name]
    
    return feature_vector


def get_all_teams_from_training_data(training_data_path: str) -> List[str]:
    """
    Extract all unique teams from training data for consistent team encoding.
    
    Args:
        training_data_path: Path to training data CSV
        
    Returns:
        List of team abbreviations
    """
    try:
        df = pd.read_csv(training_data_path)
        team_columns = [col for col in df.columns if col.startswith('team_')]
        teams = [col.replace('team_', '') for col in team_columns]
        return sorted(teams)
    except Exception as e:
        print(f"Warning: Could not load team list from training data: {e}")
        # Fallback to common NFL teams
        return ['ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE', 'DAL', 
                'DEN', 'DET', 'GB', 'HOU', 'IND', 'JAX', 'KC', 'LA', 'LAC', 'LV', 
                'MIA', 'MIN', 'NE', 'NO', 'NYG', 'NYJ', 'PHI', 'PIT', 'SEA', 'SF', 
                'TB', 'TEN', 'WAS']
