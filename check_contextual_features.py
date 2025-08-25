import nfl_data_py as nfl
import pandas as pd

print("=== EXPLORING CONTEXTUAL FEATURES FOR QB MODELING ===")
print()

# Get QB weekly data
weekly_data = nfl.import_weekly_data([2023, 2024])
qb_weekly = weekly_data[weekly_data['position'] == 'QB'].copy()

print("=== AVAILABLE CONTEXTUAL COLUMNS ===")
print(f"Total columns: {len(qb_weekly.columns)}")
print()

# Categorize available features
contextual_features = []
game_features = []
opponent_features = []
weather_features = []
venue_features = []

for col in qb_weekly.columns:
    col_lower = col.lower()
    
    # Game context
    if any(x in col_lower for x in ['week', 'season', 'season_type', 'game_id']):
        game_features.append(col)
    
    # Opponent info
    elif any(x in col_lower for x in ['opponent', 'away', 'home']):
        opponent_features.append(col)
    
    # Weather/Venue
    elif any(x in col_lower for x in ['weather', 'temp', 'wind', 'humidity', 'stadium', 'venue']):
        weather_features.append(col)
    
    # Team context
    elif any(x in col_lower for x in ['team', 'recent_team']):
        contextual_features.append(col)
    
    # Advanced metrics
    elif any(x in col_lower for x in ['epa', 'dakota', 'pacr', 'racr', 'wopr']):
        contextual_features.append(col)

print("GAME CONTEXT FEATURES:")
print(game_features)
print()

print("OPPONENT FEATURES:")
print(opponent_features)
print()

print("TEAM/CONTEXTUAL FEATURES:")
print(contextual_features)
print()

print("WEATHER/VENUE FEATURES:")
print(weather_features)
print()

# Check if we have opponent team data
if 'opponent_team' in qb_weekly.columns:
    print("=== OPPONENT TEAM ANALYSIS ===")
    print(f"Unique opponent teams: {qb_weekly['opponent_team'].nunique()}")
    print(f"Sample opponent teams: {qb_weekly['opponent_team'].dropna().unique()[:10]}")
    print()
    
    # Check if we can get opponent defensive stats
    print("Can we get opponent defensive stats?")
    print("This would require additional data sources beyond weekly player stats")
    print()

# Check for home/away information
if 'home_away' in qb_weekly.columns:
    print("=== HOME/AWAY ANALYSIS ===")
    print(qb_weekly['home_away'].value_counts())
    print()
else:
    print("No explicit home/away column found")
    print("Might need to derive from game_id or other sources")
    print()

# Check for weather data
weather_cols = [col for col in qb_weekly.columns if any(x in col.lower() for x in ['weather', 'temp', 'wind'])]
if weather_cols:
    print("=== WEATHER DATA AVAILABLE ===")
    print(f"Weather columns: {weather_cols}")
    print()
else:
    print("No weather data found in weekly stats")
    print("Weather data might be available from other NFL API functions")
    print()

# Check for advanced metrics
print("=== ADVANCED METRICS AVAILABLE ===")
advanced_cols = [col for col in qb_weekly.columns if any(x in col.lower() for x in ['epa', 'dakota', 'pacr', 'racr', 'wopr'])]
if advanced_cols:
    print(f"Advanced metrics: {advanced_cols}")
    print("These could be valuable features for QB modeling")
    print()
else:
    print("No advanced metrics found")

print("=== RECOMMENDATION FOR QB MODELING ===")
print("Based on available data, your QB model can consider:")
print()

if 'opponent_team' in qb_weekly.columns:
    print("✅ Opponent team (can add defensive stats later)")
if 'week' in qb_weekly.columns:
    print("✅ Week of season (early/mid/late season patterns)")
if 'season' in qb_weekly.columns:
    print("✅ Season (year-over-year trends)")
if advanced_cols:
    print(f"✅ Advanced metrics: {', '.join(advanced_cols)}")

print()
print("To add more contextual factors, you could:")
print("1. Import opponent defensive stats from NFL API")
print("2. Add weather data if available")
print("3. Add home/away context")
print("4. Add team strength metrics")
print("5. Add rest days between games")

