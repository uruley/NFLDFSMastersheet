#!/usr/bin/env python3
"""
Check the status of all pipeline files and show which ones have historical features
"""

import pandas as pd
import os
from pathlib import Path

def check_file_status():
    print("=" * 80)
    print("DFS PIPELINE STATUS CHECK")
    print("=" * 80)
    
    # Check input files
    print("\n📥 INPUT FILES:")
    dk_file = Path('data/raw/DKSalaries.csv')
    if dk_file.exists():
        df = pd.read_csv(dk_file)
        print(f"✅ DKSalaries.csv: {len(df)} players, {len(df.columns)} columns")
    else:
        print("❌ DKSalaries.csv: NOT FOUND")
    
    # Check intermediate files
    print("\n🔄 INTERMEDIATE FILES:")
    master_file = Path('data/processed/master_sheet_2025.csv')
    if master_file.exists():
        df = pd.read_csv(master_file)
        print(f"✅ master_sheet_2025.csv: {len(df)} players, {len(df.columns)} columns")
    else:
        print("❌ master_sheet_2025.csv: NOT FOUND")
    
    # Check position model outputs
    print("\n🎯 POSITION MODEL OUTPUTS:")
    position_dirs = ['QB', 'RB', 'WR', 'TE']
    for pos in position_dirs:
        pred_dir = Path(f'PositionModel/{pos}/predictions')
        if pred_dir.exists():
            pred_files = list(pred_dir.glob('*.csv'))
            if pred_files:
                latest_file = max(pred_files, key=lambda x: x.stat().st_mtime)
                df = pd.read_csv(latest_file)
                print(f"✅ {pos} predictions: {len(df)} players, {len(df.columns)} columns")
            else:
                print(f"❌ {pos} predictions: NO FILES FOUND")
        else:
            print(f"❌ {pos} predictions: DIRECTORY NOT FOUND")
    
    # Check team model output
    print("\n🏈 TEAM MODEL OUTPUT:")
    team_file = Path('data/outputs/predictions_2025w01.csv')
    if team_file.exists():
        df = pd.read_csv(team_file)
        print(f"✅ Team predictions: {len(df)} games, {len(df.columns)} columns")
    else:
        print("❌ Team predictions: NOT FOUND")
    
    # Check PlayerMaster files
    print("\n📊 PLAYERMASTER FILES:")
    
    # PlayerMaster_current.csv
    current_file = Path('data/DFSDashboard/PlayerMaster_current.csv')
    if current_file.exists():
        df = pd.read_csv(current_file)
        rb_df = df[df['position'] == 'RB']
        targets_nonzero = (rb_df['targets_l3'] != 0).sum() if 'targets_l3' in df.columns else 0
        rush_nonzero = (rb_df['rush_att_l3'] != 0).sum() if 'rush_att_l3' in df.columns else 0
        
        print(f"✅ PlayerMaster_current.csv: {len(df)} players, {len(df.columns)} columns")
        print(f"   📈 Historical features: targets_l3 non-zero: {targets_nonzero}/{len(rb_df)} RBs")
        print(f"   📈 Historical features: rush_att_l3 non-zero: {rush_nonzero}/{len(rb_df)} RBs")
        print(f"   🎯 STATUS: {'✅ WORKING' if targets_nonzero > 0 else '❌ BROKEN'}")
    else:
        print("❌ PlayerMaster_current.csv: NOT FOUND")
    
    # PlayerMaster_v2_2025w01.csv
    v2_file = Path('data/DFSDashboard/PlayerMaster_v2_2025w01.csv')
    if v2_file.exists():
        df = pd.read_csv(v2_file)
        rb_df = df[df['position'] == 'RB']
        targets_nonzero = (rb_df['targets_l3'] != 0).sum() if 'targets_l3' in df.columns else 0
        rush_nonzero = (rb_df['rush_att_l3'] != 0).sum() if 'rush_att_l3' in df.columns else 0
        
        print(f"⚠️  PlayerMaster_v2_2025w01.csv: {len(df)} players, {len(df.columns)} columns")
        print(f"   📈 Historical features: targets_l3 non-zero: {targets_nonzero}/{len(rb_df)} RBs")
        print(f"   📈 Historical features: rush_att_l3 non-zero: {rush_nonzero}/{len(rb_df)} RBs")
        print(f"   🎯 STATUS: {'❌ BROKEN' if targets_nonzero == 0 else '✅ WORKING'}")
    else:
        print("❌ PlayerMaster_v2_2025w01.csv: NOT FOUND")
    
    # PlayerMaster_unified.csv
    unified_file = Path('data/DFSDashboard/PlayerMaster_unified.csv')
    if unified_file.exists():
        df = pd.read_csv(unified_file)
        has_historical = 'targets_l3' in df.columns and 'rush_att_l3' in df.columns
        print(f"⚠️  PlayerMaster_unified.csv: {len(df)} players, {len(df.columns)} columns")
        print(f"   📈 Historical features: {'❌ NOT PRESENT' if not has_historical else '✅ PRESENT'}")
        print(f"   🎯 STATUS: {'❌ OLD VERSION' if not has_historical else '⚠️  CHECK'}")
    else:
        print("❌ PlayerMaster_unified.csv: NOT FOUND")
    
    # Summary
    print("\n" + "=" * 80)
    print("📋 SUMMARY")
    print("=" * 80)
    
    if current_file.exists():
        df = pd.read_csv(current_file)
        rb_df = df[df['position'] == 'RB']
        targets_nonzero = (rb_df['targets_l3'] != 0).sum() if 'targets_l3' in df.columns else 0
        
        if targets_nonzero > 0:
            print("🎉 SUCCESS: PlayerMaster_current.csv has historical features!")
            print("✅ USE THIS FILE FOR YOUR OPTIMIZER")
        else:
            print("❌ PROBLEM: PlayerMaster_current.csv missing historical features")
            print("🔄 Run: python data/DFSDashboard/playermaster_from_projections.py --proj-dir PositionModel")
    else:
        print("❌ PROBLEM: PlayerMaster_current.csv not found")
        print("🔄 Run: python data/DFSDashboard/playermaster_from_projections.py --proj-dir PositionModel")
    
    print("\n🚫 DON'T USE:")
    print("   - PlayerMaster_v2_2025w01.csv (historical features lost)")
    print("   - PlayerMaster_unified.csv (old version)")

if __name__ == "__main__":
    check_file_status()
