import nfl_data_py as nfl
import pandas as pd

# Load weekly data
weekly_data = nfl.import_weekly_data([2023, 2024])

# Check Drew Lock's data
lock_id = '00-0035704'
lock_data = weekly_data[
    (weekly_data['player_id'] == lock_id) & 
    (weekly_data['position'] == 'QB')
].copy()

print("Drew Lock Historical Data Analysis:")
print("=" * 60)
print(f"Player ID: {lock_id}")
print(f"Records found: {len(lock_data)}")

if len(lock_data) > 0:
    print("\nGame Log:")
    print(lock_data[['season', 'week', 'recent_team', 'fantasy_points', 'passing_yards', 'passing_tds', 'rushing_yards']].sort_values(['season', 'week']).to_string(index=False))
    
    print(f"\nAverage Fantasy Points: {lock_data['fantasy_points'].mean():.2f}")
    print(f"Total Games: {len(lock_data)}")
else:
    print("\n❌ NO HISTORICAL DATA FOUND")
    print("This explains why he's getting league average fallback!")

# Check what happens with league average fallback
print("\n" + "=" * 60)
print("League Average Fallback Analysis:")
print("=" * 60)

qb_avg = weekly_data[weekly_data['position'] == 'QB'].mean(numeric_only=True)
print(f"League Average QB Fantasy Points: {qb_avg['fantasy_points']:.2f}")
print(f"League Average QB Passing Yards: {qb_avg['passing_yards']:.2f}")
print(f"League Average QB Passing TDs: {qb_avg['passing_tds']:.2f}")

# Check if there are backup QBs with low fantasy points
backup_qbs = weekly_data[
    (weekly_data['position'] == 'QB') & 
    (weekly_data['fantasy_points'] < 10)
].copy()

print(f"\nBackup QBs (<10 pts) sample:")
print(backup_qbs[['player_name', 'fantasy_points', 'passing_yards', 'passing_tds']].head(10).to_string(index=False))

print(f"\nBackup QB Average Fantasy Points: {backup_qbs['fantasy_points'].mean():.2f}")
