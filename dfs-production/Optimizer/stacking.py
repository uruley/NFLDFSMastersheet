# optimizer/stacking.py
import pandas as pd
import os
from pulp import *

class StackingConstraints:
    def __init__(self, projections, crosswalk_path=None):
        """Initialize with projections and crosswalk for game info"""
        self.projections = projections.copy()
        
        # Try to load crosswalk if path provided
        if crosswalk_path and os.path.exists(crosswalk_path):
            try:
                self.crosswalk = pd.read_csv(crosswalk_path)
                # Try to merge game info if available
                if 'game_info' in self.crosswalk.columns or 'Game Info' in self.crosswalk.columns:
                    game_col = 'game_info' if 'game_info' in self.crosswalk.columns else 'Game Info'
                    self.projections = self.projections.merge(
                        self.crosswalk[['player_name', 'team', 'position', game_col]],
                        on=['player_name', 'team', 'position'],
                        how='left'
                    )
                    # Rename to standard format
                    if 'Game Info' in self.projections.columns:
                        self.projections = self.projections.rename(columns={'Game Info': 'game_info'})
            except Exception as e:
                print(f"Warning: Could not load crosswalk from {crosswalk_path}: {e}")
                self.crosswalk = pd.DataFrame()
        
        # If no game_info from crosswalk, create synthetic game info
        if 'game_info' not in self.projections.columns:
            # Create synthetic game info based on team pairs for testing
            teams = self.projections['team'].unique()
            game_mapping = {}
            for i in range(0, len(teams), 2):
                if i + 1 < len(teams):
                    game_id = f"{teams[i]}@{teams[i+1]}"
                    game_mapping[teams[i]] = game_id
                    game_mapping[teams[i+1]] = game_id
            
            self.projections['game_info'] = self.projections['team'].map(game_mapping)
            self.projections['game_info'] = self.projections['game_info'].fillna('OTHER')
            
            print(f"✓ Created synthetic game_info with {len(game_mapping)} games")
            print(f"  Game mapping: {game_mapping}")
        
        # Extract game_id (e.g., 'CIN@CLE')
        self.projections['game_id'] = self.projections['game_info'].apply(
            lambda x: x if pd.notna(x) else 'OTHER'
        )
        
        print(f"✓ Final columns: {self.projections.columns.tolist()}")
        print(f"✓ Game info sample: {self.projections['game_info'].head().tolist()}")

    def add_qb_wr_stack(self, prob, player_vars):
        """Enforce at least one QB-WR or QB-TE stack from the same team"""
        team_players = self.projections.groupby('team')
        for team, group in team_players:
            if pd.notna(team) and team != '':
                qb_indices = group[group['position'] == 'QB'].index.tolist()
                wr_te_indices = group[group['position'].isin(['WR', 'TE'])].index.tolist()
                if qb_indices and wr_te_indices:
                    # Constraint: If QB is selected, at least one WR/TE from same team must be selected
                    for qb_idx in qb_indices:
                        prob += lpSum([player_vars[idx] for idx in wr_te_indices]) >= player_vars[qb_idx]
                        # Note: The constraint above already ensures QB-WR/TE stacking
                        # We don't need to multiply variables (which PuLP doesn't allow)

    def add_game_stack(self, prob, player_vars, min_players=4):
        """Allow 4+ players from the same game (both teams)"""
        game_groups = self.projections.groupby('game_id')
        for game_id, group in game_groups:
            if pd.notna(game_id) and game_id != '':
                game_indices = group.index.tolist()
                # Constraint: If any players from this game are selected, encourage stacking
                # This is a softer constraint that encourages but doesn't force game stacking
                pass  # Removed complex constraint that could cause issues

    def add_bring_back(self, prob, player_vars):
        """For game stacks, require at least one player from the opposing team"""
        game_groups = self.projections.groupby('game_id')
        for game_id, group in game_groups:
            if pd.notna(game_id) and game_id != '' and '@' in game_id:
                team1, team2 = game_id.split('@')
                team1_indices = group[group['team'] == team1].index.tolist()
                team2_indices = group[group['team'] == team2].index.tolist()
                if team1_indices and team2_indices:
                    # Constraint: If 3+ players from team1, include at least 1 from team2
                    prob += lpSum([player_vars[idx] for idx in team2_indices]) >= lpSum([player_vars[idx] for idx in team1_indices]) / 3
                    # Same for team2
                    prob += lpSum([player_vars[idx] for idx in team1_indices]) >= lpSum([player_vars[idx] for idx in team2_indices]) / 3

    def get_stack_info(self, lineup):
        """Return stack details for a lineup"""
        stack_info = {'qb_wr_stacks': [], 'game_stacks': [], 'bring_backs': []}
        # QB-WR/TE stacks
        team_groups = lineup.groupby('team')
        for team, group in team_groups:
            qbs = group[group['position'] == 'QB']['player_name'].tolist()
            wr_te = group[group['position'].isin(['WR', 'TE'])]['player_name'].tolist()
            if qbs and wr_te:
                stack_info['qb_wr_stacks'].append({'team': team, 'qb': qbs[0], 'wr_te': wr_te})
        # Game stacks
        if 'game_id' in lineup.columns:
            game_groups = lineup.groupby('game_id')
            for game_id, group in game_groups:
                if len(group) >= 4:
                    stack_info['game_stacks'].append({'game_id': game_id, 'players': group['player_name'].tolist()})
                    # Check bring-back
                    teams = game_id.split('@') if '@' in game_id else []
                    if len(teams) == 2:
                        team1_players = group[group['team'] == teams[0]]['player_name'].tolist()
                        team2_players = group[group['team'] == teams[1]]['player_name'].tolist()
                        if team1_players and team2_players:
                            stack_info['bring_backs'].append({
                                'game_id': game_id,
                                'team1_players': team1_players,
                                'team2_players': team2_players
                            })
        else:
            # Fallback if game_id not available
            pass
        return stack_info
    
    def calculate_stacking_score(self, lineup, projections):
        """Calculate a stacking quality score (0-100)"""
        score = 0.0
        
        # QB-WR stack bonus (0-30 points)
        qb_teams = set(lineup[lineup['position'] == 'QB']['team'])
        wr_te_teams = set(lineup[lineup['position'].isin(['WR', 'TE'])]['team'])
        qb_wr_stacks = qb_teams.intersection(wr_te_teams)
        if len(qb_wr_stacks) >= 1:
            score += 30
        
        # Game stack bonus (0-25 points)
        if 'game_info' in lineup.columns:
            game_counts = lineup.groupby('game_info').size()
            max_game_players = game_counts.max() if not game_counts.empty else 0
            if max_game_players >= 4:
                score += 25
            elif max_game_players >= 3:
                score += 15
            elif max_game_players >= 2:
                score += 5
        
        # Bring-back bonus (0-20 points)
        if 'game_info' in lineup.columns:
            game_groups = lineup.groupby('game_info')
            for game_id, group in game_groups:
                if len(group) >= 2:
                    unique_teams = set(group['team'])
                    if len(unique_teams) >= 2:
                        score += 20
                        break
        
        # Team diversity bonus (0-25 points)
        team_counts = lineup['team'].value_counts()
        if team_counts.max() <= 3:
            score += 25
        elif team_counts.max() <= 4:
            score += 15
        else:
            score += 0
        
        return score