import nfl_data_py as nfl
import pandas as pd

# Load 2024 data
weekly_data = nfl.import_weekly_data([2024])

# Filter for Jahmyr Gibbs
gibbs_data = weekly_data[
    (weekly_data['player_name'] == 'J.Gibbs') & 
    (weekly_data['recent_team'] == 'DET') & 
    (weekly_data['position'] == 'RB')
].copy()

print("Jahmyr Gibbs 2024 Game Log:")
print("=" * 100)

# Show game log
game_log = gibbs_data[['week', 'opponent_team', 'fantasy_points', 'rushing_yards', 'rushing_tds', 'carries', 'receptions', 'receiving_yards', 'receiving_tds']].sort_values('week')
print(game_log.to_string(index=False))

print("\n" + "=" * 100)
print("SEASON SUMMARY:")
print("=" * 100)

print(f"Games Played: {len(gibbs_data)}")
print(f"Total Fantasy Points: {gibbs_data['fantasy_points'].sum():.1f}")
print(f"Average Fantasy Points: {gibbs_data['fantasy_points'].mean():.1f}")
print(f"Best Game: Week {gibbs_data.loc[gibbs_data['fantasy_points'].idxmax(), 'week']} vs {gibbs_data.loc[gibbs_data['fantasy_points'].idxmax(), 'opponent_team']} ({gibbs_data['fantasy_points'].max():.1f} pts)")
print(f"Worst Game: Week {gibbs_data.loc[gibbs_data['fantasy_points'].idxmin(), 'week']} vs {gibbs_data.loc[gibbs_data['fantasy_points'].idxmin(), 'opponent_team']} ({gibbs_data['fantasy_points'].min():.1f} pts)")

print(f"\nRushing: {gibbs_data['rushing_yards'].sum():.0f} yards, {gibbs_data['rushing_tds'].sum():.0f} TDs")
print(f"Receiving: {gibbs_data['receiving_yards'].sum():.0f} yards, {gibbs_data['receiving_tds'].sum():.0f} TDs")
print(f"Total Yards: {gibbs_data['rushing_yards'].sum() + gibbs_data['receiving_yards'].sum():.0f}")
print(f"Total TDs: {gibbs_data['rushing_tds'].sum() + gibbs_data['receiving_tds'].sum():.0f}")

print(f"\nGames with 20+ points: {(gibbs_data['fantasy_points'] >= 20).sum()}")
print(f"Games with 15+ points: {(gibbs_data['fantasy_points'] >= 15).sum()}")
print(f"Games under 10 points: {(gibbs_data['fantasy_points'] < 10).sum()}")
