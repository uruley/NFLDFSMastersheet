# scripts/verify_id_consistency.py
import nfl_data_py as nfl
import pandas as pd

seasons = [2024]

# A) Pull from the API (ground truth)
rost_api = nfl.import_seasonal_rosters(seasons)
wk_api   = nfl.import_weekly_data(seasons)

print("API roster player_id sample:", rost_api['player_id'].dropna().astype(str).head().tolist())
print("API weekly player_id sample:", wk_api['player_id'].dropna().astype(str).head().tolist())

rost_ids_api = set(rost_api['player_id'].dropna().astype(str))
wk_ids_api   = set(wk_api['player_id'].dropna().astype(str))
overlap_api  = len(rost_ids_api & wk_ids_api)
print(f"[API] Overlap player_id: {overlap_api} / roster {len(rost_ids_api)}")

# B) Compare your master sheet (what our pipeline outputs) to the API weekly
master = pd.read_csv('data/processed/history/2025/master_sheet_2025_20250824.csv', dtype=str)
print("Master columns:", list(master.columns)[:20])
print("Master player_id sample:", master['player_id'].dropna().head().tolist()[:5])

master_ids = set(master['player_id'].dropna().astype(str))
overlap_master_weekly = len(master_ids & wk_ids_api)
print(f"[Master vs API weekly] Overlap player_id: {overlap_master_weekly} / master {len(master_ids)}")

# Show a few that don't overlap to diagnose
if overlap_master_weekly < len(master_ids):
    missing = list(master_ids - wk_ids_api)[:10]
    print("Example master IDs not in weekly:", missing)
