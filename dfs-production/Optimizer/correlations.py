# optimizer/correlations.py
import pandas as pd
import nfl_data_py as nfl
import json
import os

class CorrelationCalculator:
    def __init__(self, data_dir='data', crosswalk_path='../../data/processed/crosswalk_2025.csv', master_sheet_path='../../data/processed/master_sheet_2025.csv'):
        """Initialize with directory and paths to crosswalk/master sheet"""
        self.data_dir = data_dir
        self.correlations_file = os.path.join(data_dir, 'correlations.json')
        os.makedirs(data_dir, exist_ok=True)
        self.correlations = {}
        self.crosswalk = pd.read_csv(crosswalk_path) if os.path.exists(crosswalk_path) else pd.DataFrame()
        self.master_sheet = pd.read_csv(master_sheet_path) if os.path.exists(master_sheet_path) else pd.DataFrame()
        print(f"Loaded crosswalk with {len(self.crosswalk)} entries")
        print(f"Loaded master sheet with {len(self.master_sheet)} entries")

    def clean_name(self, name):
        """Standardize name to match crosswalk join_key format"""
        if pd.isna(name):
            return ''
        return name.lower().replace("'", "").replace(".", "").replace("-", " ").strip()

    def fetch_historical_data(self, years=[2022, 2023, 2024]):
        """Fetch play-by-play and roster data using nfl_data_py"""
        weekly_data = nfl.import_weekly_data(years)
        rosters = nfl.import_rosters(years)
        return weekly_data, rosters

    def calculate_dk_points(self, row):
        """Calculate DraftKings fantasy points from raw stats"""
        points = 0.0
        # Passing
        passing_yards = row.get('passing_yards', 0)
        points += passing_yards * 0.04
        if passing_yards >= 300:
            points += 3
        points += row.get('passing_tds', 0) * 4
        points -= row.get('interceptions', 0) * 1
        points += row.get('passing_2pt_conversions', 0) * 2
        # Rushing
        rushing_yards = row.get('rushing_yards', 0)
        points += rushing_yards * 0.1
        if rushing_yards >= 100:
            points += 3
        points += row.get('rushing_tds', 0) * 6
        points += row.get('rushing_2pt_conversions', 0) * 2
        points -= row.get('rushing_fumbles_lost', 0) * 1
        # Receiving
        receiving_yards = row.get('receiving_yards', 0)
        points += receiving_yards * 0.1
        if receiving_yards >= 100:
            points += 3
        points += row.get('receptions', 0) * 1  # Full PPR
        points += row.get('receiving_tds', 0) * 6
        points += row.get('receiving_2pt_conversions', 0) * 2
        points -= row.get('receiving_fumbles_lost', 0) * 1
        # Misc
        points += row.get('special_teams_tds', 0) * 6
        points -= row.get('sack_fumbles_lost', 0) * 1
        return points

    def calculate_correlations(self, weekly_data, rosters):
        """Calculate correlations using standardized join_key"""
        # Standardize historical data
        if 'recent_team' in weekly_data.columns:
            weekly_data = weekly_data.rename(columns={'recent_team': 'team'})
        display_name_col = 'player_name' if 'player_name' in weekly_data.columns else 'player_display_name'
        weekly_data['clean_name'] = weekly_data[display_name_col].apply(self.clean_name)
        rosters['clean_name'] = rosters['player_name'].apply(self.clean_name)
        
        # Create join_key
        weekly_data['join_key'] = weekly_data['clean_name'] + '|' + weekly_data['team'] + '|' + weekly_data['position']
        rosters['join_key'] = rosters['clean_name'] + '|' + rosters['team'] + '|' + rosters['position']
        
        # Merge with crosswalk for additional data (crosswalk has 'join_key', 'Name', 'Position', 'TeamAbbrev')
        weekly_data = weekly_data.merge(self.crosswalk[['join_key', 'Name', 'Position', 'TeamAbbrev']], on='join_key', how='left')
        rosters = rosters.merge(self.crosswalk[['join_key', 'Name', 'Position', 'TeamAbbrev']], on='join_key', how='left')
        
        # Merge with master_sheet for player_id
        weekly_data = weekly_data.merge(self.master_sheet[['join_key', 'player_id']], on='join_key', how='left')
        rosters = rosters.merge(self.master_sheet[['join_key', 'player_id']], on='join_key', how='left')
        
        # Filter positions and calculate DK points
        data = weekly_data[weekly_data['position'].isin(['QB', 'RB', 'WR', 'TE'])]
        data['dk_fantasy_points'] = data.apply(self.calculate_dk_points, axis=1)
        print("Sample DK points calculated:")
        print(data[['Name', 'position', 'dk_fantasy_points', 'join_key']].head())
        
        # Pivot using join_key (fallback to Name if NaN)
        data['pivot_key'] = data['Name'].fillna(data['join_key'])
        pivot_data = data.pivot_table(
            index=['season', 'week', 'team'],
            columns='pivot_key',
            values='dk_fantasy_points',
            fill_value=0
        )
        
        # Calculate team correlations
        team_correlations = {}
        for team in data['team'].unique():
            team_data = data[data['team'] == team]
            team_pivot = team_data.pivot_table(
                index=['season', 'week'],
                columns='pivot_key',
                values='dk_fantasy_points',
                fill_value=0
            )
            corr_matrix = team_pivot.corr()
            
            # Store using join_key as key
            team_players = team_data[['join_key', 'Name', 'position', 'player_id']].drop_duplicates()
            for idx1, row1 in team_players.iterrows():
                for idx2, row2 in team_players.iterrows():
                    if row1['join_key'] != row2['join_key']:
                        key = f"{row1['join_key']}_{row2['join_key']}"
                        try:
                            corr_value = corr_matrix.loc[row1['pivot_key'], row2['pivot_key']]
                        except (KeyError, IndexError):
                            corr_value = 0
                        team_correlations[key] = {
                            'player1_join_key': row1['join_key'],
                            'player1_name': row1['Name'],
                            'player1_pos': row1['position'],
                            'player1_id': row1['player_id'],
                            'player2_join_key': row2['join_key'],
                            'player2_name': row2['Name'],
                            'player2_pos': row2['position'],
                            'player2_id': row2['player_id'],
                            'correlation': corr_value if not pd.isna(corr_value) else 0
                        }
        
        self.correlations = team_correlations
        self.save_correlations()
        return self.correlations

    def save_correlations(self):
        """Save correlations to JSON file"""
        with open(self.correlations_file, 'w') as f:
            json.dump(self.correlations, f, indent=4)
        print(f"Correlations saved to {self.correlations_file}")

    def load_correlations(self):
        """Load correlations from JSON file"""
        if os.path.exists(self.correlations_file):
            with open(self.correlations_file, 'r') as f:
                self.correlations = json.load(f)
            print(f"Loaded correlations from {self.correlations_file}")
            return self.correlations
        return {}

    def apply_correlation_adjustments(self, projections_df):
        """Adjust projections using join_key-matched correlations, filtered to current slate via crosswalk"""
        projections = projections_df.copy()
        projections['adjusted_projection'] = projections['projection']
        
        # Standardize projections to match crosswalk format
        projections['clean_name'] = projections['player_name'].apply(self.clean_name)
        projections['join_key'] = projections['clean_name'] + '|' + projections['team'] + '|' + projections['position']
        
        # Merge with crosswalk to confirm slate players and get game_info
        # Check if crosswalk has game_info column, if not create a placeholder
        crosswalk_cols = ['join_key']
        if 'game_info' in self.crosswalk.columns:
            crosswalk_cols.append('game_info')
        if 'Name' in self.crosswalk.columns:
            crosswalk_cols.append('Name')
            
        projections = projections.merge(
            self.crosswalk[crosswalk_cols],
            on='join_key',
            how='inner'
        )
        
        # Add game_info if it doesn't exist (create synthetic game info for testing)
        if 'game_info' not in projections.columns:
            # Create synthetic game info based on team pairs for testing
            teams = projections['team'].unique()
            game_mapping = {}
            for i in range(0, len(teams), 2):
                if i + 1 < len(teams):
                    game_id = f"{teams[i]}@{teams[i+1]}"
                    game_mapping[teams[i]] = game_id
                    game_mapping[teams[i+1]] = game_id
            
            projections['game_info'] = projections['team'].map(game_mapping)
            projections['game_info'] = projections['game_info'].fillna('OTHER')
        
        projections = projections.merge(self.master_sheet[['join_key', 'player_id']], on='join_key', how='left')
        
        # Log unmatched projections
        if len(projections) < len(projections_df):
            unmatched_count = len(projections_df) - len(projections)
            print(f"⚠️ {unmatched_count} players in projections not found in crosswalk (skipped for adjustments)")
        
        # Get set of current join_keys
        current_join_keys = set(projections['join_key'].dropna().unique())

        for idx, row in projections.iterrows():
            join_key = row['join_key']
            team = row['team']
            if pd.isna(join_key) or pd.isna(team):
                continue
            
            # Find related correlations
            related_players = [
                corr for key, corr in self.correlations.items()
                if (key.startswith(f"{join_key}_") or key.endswith(f"_{join_key}"))
                and corr['correlation'] > 0
            ]
            
            # Apply only for same-team, slate-active pairs
            for corr in related_players:
                other_join_key = corr['player2_join_key'] if corr['player1_join_key'] == join_key else corr['player1_join_key']
                if other_join_key in current_join_keys:
                    other_row = projections[projections['join_key'] == other_join_key]
                    if not other_row.empty and other_row['team'].values[0] == team:
                        adjustment_factor = corr['correlation'] * 0.1
                        projections.loc[idx, 'adjusted_projection'] += row['projection'] * adjustment_factor

        # Merge adjustments back to original projections_df
        projections_df = projections_df.merge(
            projections[['player_name', 'team', 'position', 'adjusted_projection', 'game_info']],
            on=['player_name', 'team', 'position'],
            how='left'
        )
        projections_df['adjusted_projection'] = projections_df['adjusted_projection'].fillna(projections_df['projection'])
        return projections_df