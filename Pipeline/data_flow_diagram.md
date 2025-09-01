# Data Flow Diagram

## Pipeline Flow

```
DKSalaries.csv (Raw DraftKings Data)
           ↓
build_master_sheet.py
           ↓
master_sheet_2025.csv (Player mappings + current slate)
           ↓
Position Models (QB/RB/WR/TE)
           ↓
Individual Prediction Files
           ↓
playermaster_from_projections.py ⭐
           ↓
PlayerMaster_current.csv ⭐ (USE THIS FOR OPTIMIZER)
           ↓
Dashboard.py (OPTIONAL - BROKEN)
           ↓
PlayerMaster_v2_2025w01.csv (DON'T USE - Historical features lost)
```

## File Details

### Input Files
- `data/raw/DKSalaries.csv` - Raw DraftKings salary file

### Intermediate Files
- `data/processed/master_sheet_2025.csv` - Player mappings and crosswalk
- `PositionModel/*/predictions/*.csv` - Individual position predictions

### Output Files

#### ✅ PlayerMaster_current.csv (GOOD)
- **Created by**: `playermaster_from_projections.py`
- **Size**: ~160KB, 746 lines
- **Columns**: 23 columns
- **Historical Features**: ✅ WORKING
- **Use**: ✅ **USE THIS FOR OPTIMIZER**

#### ❌ PlayerMaster_v2_2025w01.csv (BROKEN)
- **Created by**: `Dashboard.py`
- **Size**: ~200KB, 784 lines
- **Columns**: 41 columns
- **Historical Features**: ❌ ZEROED OUT
- **Use**: ❌ **DON'T USE**

#### ❌ PlayerMaster_unified.csv (OLD)
- **Created by**: Old version of pipeline
- **Size**: ~122KB, 784 lines
- **Columns**: 12 columns only
- **Historical Features**: ❌ NOT PRESENT
- **Use**: ❌ **DON'T USE**

## Historical Features Flow

```
RB Model (rb_inference.py)
           ↓
Calculates: targets_l3, rush_att_l3, etc.
           ↓
Outputs to: PositionModel/RB/predictions/
           ↓
playermaster_from_projections.py
           ↓
Preserves historical features ✅
           ↓
PlayerMaster_current.csv
           ↓
Dashboard.py
           ↓
Overwrites with zeros ❌
           ↓
PlayerMaster_v2_2025w01.csv
```

## Quick Commands

### Check Which File Has Historical Features
```bash
# Check PlayerMaster_current.csv
python -c "import pandas as pd; df = pd.read_csv('data/DFSDashboard/PlayerMaster_current.csv'); rb = df[df['position']=='RB']; print(f'RB players: {len(rb)}'); print(f'targets_l3 non-zero: {(rb[\"targets_l3\"] != 0).sum()}')"

# Check PlayerMaster_v2_2025w01.csv
python -c "import pandas as pd; df = pd.read_csv('data/DFSDashboard/PlayerMaster_v2_2025w01.csv'); rb = df[df['position']=='RB']; print(f'RB players: {len(rb)}'); print(f'targets_l3 non-zero: {(rb[\"targets_l3\"] != 0).sum()}')"
```

### Run Just The Working Part
```bash
python data/DFSDashboard/playermaster_from_projections.py --proj-dir PositionModel
```

## Summary

**ALWAYS USE**: `data/DFSDashboard/PlayerMaster_current.csv`

**NEVER USE**: 
- `data/DFSDashboard/PlayerMaster_v2_2025w01.csv`
- `data/DFSDashboard/PlayerMaster_unified.csv`
