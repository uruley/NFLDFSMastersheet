# DFS Pipeline Data Flow

## Overview
This document shows the exact flow of data through the DFS pipeline and what each CSV file contains.

## Pipeline Steps

### 1. Raw Data Input
- **File**: `data/raw/DKSalaries.csv`
- **What it is**: Raw DraftKings salary file with current slate
- **Columns**: Name, Position, TeamAbbrev, Salary, etc.

### 2. Master Sheet Creation
- **Script**: `scripts/build_master_sheet.py`
- **Input**: `data/raw/DKSalaries.csv`
- **Output**: `data/processed/master_sheet_2025.csv`
- **What it does**: Creates master sheet with player mappings and crosswalk
- **Key**: Filters to current slate players only

### 3. Position Model Predictions
- **Scripts**: 
  - `PositionModel/QB/qb_inference.py`
  - `PositionModel/RB/rb_inference.py`
  - `PositionModel/WR/wr_inference.py`
  - `PositionModel/TE/te_inference.py`
- **Input**: Training data + current week data
- **Output**: Individual prediction files in `PositionModel/*/predictions/`
- **Key**: RB model calculates historical features (targets_l3, rush_att_l3, etc.)

### 4. Team Model Predictions
- **Script**: `data/TeamModels/train_teamtotal.py`
- **Input**: Historical team data
- **Output**: `data/outputs/predictions_2025w01.csv`
- **What it contains**: Team totals, spreads, win probabilities

### 5. PlayerMaster Creation (THE IMPORTANT STEP)
- **Script**: `data/DFSDashboard/playermaster_from_projections.py`
- **Input**: 
  - `data/processed/master_sheet_2025.csv`
  - Position model predictions
- **Output**: `data/DFSDashboard/PlayerMaster_current.csv` ⭐
- **What it contains**: 
  - All player projections
  - Historical features (targets_l3, rush_att_l3, etc.)
  - Current slate players only
- **Status**: ✅ THIS IS THE FILE YOU WANT FOR OPTIMIZER

### 6. Dashboard Enrichment (OPTIONAL - BROKEN)
- **Script**: `data/DFSDashboard/Dashboard.py`
- **Input**: `data/DFSDashboard/PlayerMaster_current.csv`
- **Output**: `data/DFSDashboard/PlayerMaster_v2_2025w01.csv`
- **What it does**: Adds team context (spreads, totals, etc.)
- **Status**: ❌ BROKEN - Overwrites historical features with zeros

## File Comparison

### PlayerMaster_current.csv (GOOD)
- **Size**: ~160KB, 746 lines
- **Columns**: 23 columns including historical features
- **Historical Features**: ✅ WORKING (targets_l3, rush_att_l3, etc.)
- **Use**: ✅ USE THIS FOR OPTIMIZER

### PlayerMaster_v2_2025w01.csv (BROKEN)
- **Size**: ~200KB, 784 lines  
- **Columns**: 41 columns including team context
- **Historical Features**: ❌ ZEROED OUT (all 0.0)
- **Use**: ❌ DON'T USE - Historical features lost

### PlayerMaster_unified.csv (OLD)
- **Size**: ~122KB, 784 lines
- **Columns**: 12 columns only
- **Historical Features**: ❌ NOT PRESENT
- **Use**: ❌ OLD VERSION - DON'T USE

## Quick Commands

### Run Complete Pipeline
```bash
python Pipeline/run_pipeline.py
```

### Run Just PlayerMaster Creation (RECOMMENDED)
```bash
python data/DFSDashboard/playermaster_from_projections.py --proj-dir PositionModel
```

### Check Historical Features
```bash
python -c "import pandas as pd; df = pd.read_csv('data/DFSDashboard/PlayerMaster_current.csv'); rb = df[df['position']=='RB']; print(f'RB players: {len(rb)}'); print(f'targets_l3 non-zero: {(rb[\"targets_l3\"] != 0).sum()}')"
```

## Troubleshooting

### Historical Features Missing
1. Check if `PlayerMaster_current.csv` has historical features
2. If not, run `playermaster_from_projections.py` again
3. **DO NOT** run Dashboard.py as it overwrites historical features

### Wrong File Being Used
- Always use `PlayerMaster_current.csv` for optimizer
- Ignore `PlayerMaster_v2_2025w01.csv` (broken)
- Ignore `PlayerMaster_unified.csv` (old version)

### Pipeline Issues
- Check that position models are running correctly
- Verify that `master_sheet_2025.csv` exists
- Ensure current slate is being processed correctly


