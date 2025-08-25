import pandas as pd
from pathlib import Path
from dk_mapper import DraftKingsMapper  # Your mapper from step 2

# 1. Load a real DraftKings CSV
dk_slate = pd.read_csv('data/samples/dk_week10.csv')  # You need to download this from DK

# 2. Load your registry
registry = pd.read_parquet('data/registry/dim_players.parquet')

# 3. Try to match each DK player
mapper = DraftKingsMapper()
results = mapper.match_slate(dk_slate)

# 4. Check results
matched = results[results['player_uid'].notna()]
unmatched = results[results['player_uid'].isna()]

print(f"Matched: {len(matched)}/{len(dk_slate)} = {len(matched)/len(dk_slate)*100:.1f}%")
print(f"\nUnmatched players:")
# Use the actual column names that exist
available_cols = [col for col in ['dk_player_name', 'dk_position', 'dk_team'] if col in unmatched.columns]
if available_cols:
    print(unmatched[available_cols].head(10))
else:
    print("Column names not found. Available columns:", list(unmatched.columns))

# 5. Save for review
unmatched.to_csv('data/staging/unmatched_players.csv', index=False)
