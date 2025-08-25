import nfl_data_py as nfl
import pandas as pd

print("=== EXPLORING QB DATA IN NFL API ===")
print()

# Check what functions are available
print("Available NFL API functions:")
print(dir(nfl)[:20])  # Show first 20 functions
print()

# Let's look at weekly data for QBs
print("=== WEEKLY QB DATA STRUCTURE ===")
try:
    # Get recent weekly data for QBs
    weekly_data = nfl.import_weekly_data([2023, 2024])
    qb_weekly = weekly_data[weekly_data['position'] == 'QB']
    
    print(f"QB weekly data shape: {qb_weekly.shape}")
    print(f"Available columns: {list(qb_weekly.columns)}")
    print()
    
    # Show sample QB data
    print("Sample QB weekly data:")
    print(qb_weekly[['player_name', 'position', 'recent_team', 'week', 'season'] + 
                    [col for col in qb_weekly.columns if 'passing' in col.lower() or 'rushing' in col.lower() or 'fantasy' in col.lower()]][:5])
    print()
    
    # Check fantasy points
    if 'fantasy_points_dk' in qb_weekly.columns:
        print("DraftKings fantasy points available!")
        print(f"QB fantasy points range: {qb_weekly['fantasy_points_dk'].min():.2f} to {qb_weekly['fantasy_points_dk'].max():.2f}")
        print(f"QB fantasy points mean: {qb_weekly['fantasy_points_dk'].mean():.2f}")
    else:
        print("No DraftKings fantasy points found")
        print("Available fantasy columns:", [col for col in qb_weekly.columns if 'fantasy' in col.lower()])
    
except Exception as e:
    print(f"Error getting weekly data: {e}")

print()
print("=== SEASONAL QB DATA STRUCTURE ===")
try:
    # Get seasonal data for QBs
    seasonal_data = nfl.import_seasonal_data([2023, 2024])
    qb_seasonal = seasonal_data[seasonal_data['position'] == 'QB']
    
    print(f"QB seasonal data shape: {qb_seasonal.shape}")
    print(f"Available columns: {list(qb_seasonal.columns)}")
    print()
    
    # Show sample seasonal QB data
    print("Sample QB seasonal data:")
    print(qb_seasonal[['player_name', 'position', 'recent_team', 'season'] + 
                      [col for col in qb_seasonal.columns if 'passing' in col.lower() or 'rushing' in col.lower() or 'fantasy' in col.lower()]][:5])
    
except Exception as e:
    print(f"Error getting seasonal data: {e}")

print()
print("=== PLAYER STATS DATA STRUCTURE ===")
try:
    # Get player stats data
    player_stats = nfl.import_player_stats([2023, 2024])
    qb_stats = player_stats[player_stats['position'] == 'QB']
    
    print(f"QB player stats shape: {qb_stats.shape}")
    print(f"Available columns: {list(qb_stats.columns)}")
    print()
    
    # Show sample player stats
    print("Sample QB player stats:")
    print(qb_stats[['player_name', 'position', 'team', 'season'] + 
                   [col for col in qb_stats.columns if 'passing' in col.lower() or 'rushing' in col.lower() or 'fantasy' in col.lower()]][:5])
    
except Exception as e:
    print(f"Error getting player stats: {e}")

print()
print("=== SUMMARY OF AVAILABLE QB FEATURES ===")
try:
    # Combine all data sources to see what we have
    all_qb_data = qb_weekly if 'qb_weekly' in locals() else pd.DataFrame()
    
    if not all_qb_data.empty:
        # Categorize available features
        passing_features = [col for col in all_qb_data.columns if 'passing' in col.lower()]
        rushing_features = [col for col in all_qb_data.columns if 'rushing' in col.lower()]
        fantasy_features = [col for col in all_qb_data.columns if 'fantasy' in col.lower()]
        game_features = [col for col in all_qb_data.columns if any(x in col.lower() for x in ['week', 'season', 'team', 'opponent'])]
        
        print(f"Passing features: {len(passing_features)}")
        print(f"Rushing features: {len(rushing_features)}")
        print(f"Fantasy features: {len(fantasy_features)}")
        print(f"Game context features: {len(game_features)}")
        print()
        
        print("Key passing features:", passing_features[:10])
        print("Key rushing features:", rushing_features[:10])
        print("Fantasy features:", fantasy_features)
        
except Exception as e:
    print(f"Error summarizing features: {e}")

