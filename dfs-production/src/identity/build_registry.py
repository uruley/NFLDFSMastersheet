"""
Step 1: Build the foundation - Player Registry
This solves identity once and forever
"""

import nfl_data_py as nfl
import pandas as pd
import uuid
from pathlib import Path

def create_player_uid(row):
    """Create deterministic UUID for each player"""
    # Use name + birthdate for unique ID
    namespace = uuid.NAMESPACE_DNS
    # Try different possible name columns
    name = row.get('display_name') or row.get('name') or row.get('player_name') or 'unknown'
    birth_date = row.get('birth_date') or 'unknown'
    value = f"{name}:{birth_date}"
    return str(uuid.uuid5(namespace, value))

def build_player_registry(years=[2021, 2022, 2023, 2024, 2025]):
    """Build the master player registry"""
    
    print("Fetching rosters...")
    all_rosters = []
    for year in years:
        try:
            rosters = nfl.import_seasonal_rosters([year])
            rosters['season'] = year
            all_rosters.append(rosters)
            print(f"  {year}: {len(rosters)} players")
        except Exception as e:
            print(f"  {year}: Failed - {e}")
    
    # Combine all years
    df = pd.concat(all_rosters, ignore_index=True)
    
    # Debug: show available columns
    print(f"\nAvailable columns in roster data:")
    print(f"  {list(df.columns)}")
    
    # Show sample data
    print(f"\nSample roster data:")
    print(f"  {df.head(1).to_dict('records')[0]}")
    
    # Get ID mappings (if available)
    print("Fetching ID mappings...")
    try:
        ids = nfl.import_ids()
        # Check if gsis_id exists in roster data
        if 'gsis_id' in df.columns and 'gsis_id' in ids.columns:
            df = df.merge(
                ids[['gsis_id', 'espn_id', 'yahoo_id', 'sleeper_id']], 
                on='gsis_id', 
                how='left'
            )
            print(f"  Merged {len(ids)} ID mappings")
        else:
            print("  Skipping ID merge - gsis_id not available")
            # Add placeholder columns
            df['espn_id'] = None
            df['yahoo_id'] = None
            df['sleeper_id'] = None
    except Exception as e:
        print(f"  Warning: Could not fetch ID mappings: {e}")
        # Add placeholder columns
        df['espn_id'] = None
        df['yahoo_id'] = None
        df['sleeper_id'] = None
    
    # Create master player records (most recent info per player)
    # Use a unique identifier that exists in the data
    if 'gsis_id' in df.columns:
        registry = df.sort_values('season').groupby('gsis_id').last().reset_index()
    elif 'player_id' in df.columns:
        registry = df.sort_values('season').groupby('player_id').last().reset_index()
    else:
        # Fallback: group by name + position + team
        registry = df.sort_values('season').groupby(['display_name', 'position', 'team']).last().reset_index()
    
    # Add our UUID
    registry['player_uid'] = registry.apply(create_player_uid, axis=1)
    
    # Add normalized name for matching
    # Try different possible name columns
    name_col = None
    for col in ['display_name', 'name', 'player_name']:
        if col in registry.columns:
            name_col = col
            break
    
    if name_col:
        registry['name_normalized'] = registry[name_col].str.lower().str.replace(r'[^a-z\s]', '', regex=True)
    else:
        print("Warning: No name column found for normalization")
        registry['name_normalized'] = 'unknown'
    
    # Save
    output_path = Path('data/registry')
    output_path.mkdir(parents=True, exist_ok=True)
    
    registry.to_parquet(output_path / 'dim_players.parquet')
    print(f"\nSaved {len(registry)} players to dim_players.parquet")
    
    # Show sample
    print("\nSample players:")
    # Use the actual name column that exists
    name_col = 'player_name' if 'player_name' in registry.columns else 'name'
    print(registry[['player_uid', name_col, 'position', 'team']].head())
    
    return registry

if __name__ == "__main__":
    registry = build_player_registry()
    print(f"\nRegistry built with {len(registry)} unique players")
    print(f"Positions: {registry['position'].value_counts().to_dict()}")