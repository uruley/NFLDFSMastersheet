import nfl_data_py as nfl
import pandas as pd

# Try with completed seasons first
seasons = [2020,2021,2022,2023,2024]
try:
    df = nfl.import_weekly_data(seasons)
    print("row_count:", len(df))
    cols = sorted(df.columns.tolist())
    print("columns:", cols)
    keep = ['season','week','player_id','player_name','team','position',
            'game_id','game_date','fantasy_points_dk','passing_yards','passing_tds','interceptions',
            'rushing_yards','rushing_tds','receptions','receiving_yards','receiving_tds','fumbles_lost']
    sample = [c for c in keep if c in df.columns]
    print("sample_columns_present:", sample)
    print(df[sample].head(10).to_string(index=False))
    # sanity: show if player_id format matches our master (00-00xxxxx)
    ids = df['player_id'].dropna().astype(str)
    print("player_id_sample:", ids.head(5).tolist())
except Exception as e:
    print(f"Error loading data: {e}")
    print("Trying individual seasons...")
    
    # Try individual seasons
    for season in seasons:
        try:
            print(f"\nTrying season {season}...")
            df_season = nfl.import_weekly_data([season])
            print(f"Season {season}: {len(df_season)} rows")
            if 'player_id' in df_season.columns:
                ids = df_season['player_id'].dropna().astype(str)
                print(f"Sample IDs: {ids.head(3).tolist()}")
        except Exception as e2:
            print(f"Season {season} failed: {e2}")
