#!/usr/bin/env python3
"""
Validate that PlayerMaster only contains players from the current slate
"""

import pandas as pd
import sys
import os

def validate_slate_consistency():
    """Ensure PlayerMaster only contains current slate players"""
    
    print("\n" + "="*60)
    print("SLATE CONSISTENCY VALIDATION")
    print("="*60)
    
    # Load current slate
    current_slate_path = 'data/processed/current_slate.csv'
    if not os.path.exists(current_slate_path):
        print(f"❌ ERROR: No current slate found at {current_slate_path}")
        print("Run build_master_sheet.py first to process a slate")
        return False
    
    current_slate = pd.read_csv(current_slate_path)
    slate_players = set(current_slate['Name'].unique())
    print(f"✓ Current slate loaded: {len(slate_players)} unique players")
    
    # Load master sheet
    master_sheet_path = 'data/processed/master_sheet_2025.csv'
    if os.path.exists(master_sheet_path):
        master_sheet = pd.read_csv(master_sheet_path)
        master_players = set(master_sheet['Name'].unique())
        print(f"✓ Master sheet loaded: {len(master_players)} unique players")
        
        # Check for extra players in master sheet
        extra_in_master = master_players - slate_players
        if extra_in_master:
            print(f"\n❌ ERROR: {len(extra_in_master)} players in master sheet NOT in current slate!")
            print("Examples of invalid players in master sheet:")
            for player in list(extra_in_master)[:10]:
                print(f"  - {player}")
            return False
        else:
            print("✓ Master sheet contains only current slate players")
    else:
        print(f"⚠ Master sheet not found at {master_sheet_path}")
    
    # Load PlayerMaster current
    pm_current_path = 'data/DFSDashboard/PlayerMaster_current.csv'
    if os.path.exists(pm_current_path):
        pm = pd.read_csv(pm_current_path)
        pm_players = set(pm['Name'].unique())
        print(f"✓ PlayerMaster_current loaded: {len(pm_players)} unique players")
        
        # Check for extra players in PlayerMaster
        extra_in_pm = pm_players - slate_players
        if extra_in_pm:
            print(f"\n❌ ERROR: {len(extra_in_pm)} players in PlayerMaster NOT in current slate!")
            print("Examples of invalid players in PlayerMaster:")
            for player in list(extra_in_pm)[:10]:
                print(f"  - {player}")
            return False
        else:
            print("✓ PlayerMaster contains only current slate players")
    else:
        print(f"⚠ PlayerMaster_current not found at {pm_current_path}")
    
    # Load and check crosswalk
    crosswalk_path = 'data/processed/crosswalk_2025.csv'
    if os.path.exists(crosswalk_path):
        crosswalk = pd.read_csv(crosswalk_path)
        print(f"✓ Crosswalk loaded: {len(crosswalk)} total entries (cumulative)")
        
        # Crosswalk SHOULD have more players than current slate (it's cumulative)
        crosswalk_players = set(crosswalk['Name'].unique())
        if len(crosswalk_players) >= len(slate_players):
            print(f"✓ Crosswalk is cumulative ({len(crosswalk_players)} total players across all slates)")
        else:
            print(f"⚠ WARNING: Crosswalk has fewer players than current slate")
    
    print("\n" + "="*60)
    print("✅ VALIDATION PASSED: All files consistent with current slate")
    print("="*60)
    return True

if __name__ == "__main__":
    success = validate_slate_consistency()
    sys.exit(0 if success else 1)
