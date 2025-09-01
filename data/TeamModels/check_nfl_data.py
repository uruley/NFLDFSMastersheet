#!/usr/bin/env python3
"""
Check what NFL data sources are available and their column structures
"""

import nfl_data_py as nfl
import pandas as pd

print("=== NFL Data Sources Available ===\n")

# 1. Weekly player data
print("1. WEEKLY PLAYER DATA (nfl.import_weekly_data)")
try:
    weekly = nfl.import_weekly_data([2024], downcast=True)
    print(f"   Shape: {weekly.shape}")
    print(f"   Columns: {len(weekly.columns)}")
    team_cols = [col for col in weekly.columns if 'team' in col.lower()]
    print(f"   Team-related columns: {team_cols}")
    print(f"   Sample columns: {weekly.columns[:10].tolist()}")
    print()
except Exception as e:
    print(f"   Error: {e}\n")

# 2. Play-by-play data
print("2. PLAY-BY-PLAY DATA (nfl.import_pbp_data)")
try:
    pbp = nfl.import_pbp_data([2024], downcast=True)
    print(f"   Shape: {pbp.shape}")
    print(f"   Columns: {len(pbp.columns)}")
    team_cols = [col for col in pbp.columns if 'team' in col.lower()]
    print(f"   Team-related columns: {team_cols}")
    home_cols = [col for col in pbp.columns if 'home' in col.lower()]
    print(f"   Home-related columns: {home_cols}")
    print(f"   Sample columns: {pbp.columns[:10].tolist()}")
    print()
except Exception as e:
    print(f"   Error: {e}\n")

# 3. Schedule data
print("3. SCHEDULE DATA (nfl.import_schedules)")
try:
    sched = nfl.import_schedules([2024])
    print(f"   Shape: {sched.shape}")
    print(f"   Columns: {len(sched.columns)}")
    print(f"   All columns: {sched.columns.tolist()}")
    print()
except Exception as e:
    print(f"   Error: {e}\n")

# 4. Team data
print("4. TEAM DATA (nfl.import_team_desc)")
try:
    teams = nfl.import_team_desc()
    print(f"   Shape: {teams.shape}")
    print(f"   Columns: {len(teams.columns)}")
    print(f"   All columns: {teams.columns.tolist()}")
    print()
except Exception as e:
    print(f"   Error: {e}\n")

# 5. Game data
print("5. GAME DATA (nfl.import_game_data)")
try:
    games = nfl.import_game_data([2024])
    print(f"   Shape: {games.shape}")
    print(f"   Columns: {len(games.columns)}")
    print(f"   All columns: {games.columns.tolist()}")
    print()
except Exception as e:
    print(f"   Error: {e}\n")

print("=== RECOMMENDATION ===")
print("The schedule data (nfl.import_schedules) has home_team and away_team columns.")
print("We can merge this with the weekly player data to get home/away information.")
print("This would give us the missing home/away feature for our model.")

