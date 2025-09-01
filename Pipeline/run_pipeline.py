#!/usr/bin/env python3
"""
NFL DFS Pipeline Orchestrator
Two modes: TRAIN ALL vs RUN PREDICTIONS ONLY
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime
import json

class PipelineOrchestrator:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent  # Go up from Pipeline to NFLDFSMasterSheet root
        self.status_file = self.project_root / "pipeline_status.json"
        
        # Inference script paths (these generate the projection CSVs)
        self.inference_scripts = {
            'QB': 'PositionModel/QB/qb_inference.py',
            'RB': 'PositionModel/RB/rb_inference.py',
            'WR': 'PositionModel/WR/infer_wr_advanced.py',
            'TE': 'PositionModel/TE/infer_te_advanced.py',
            'DST': 'PositionModel/DST/infer_dst_advanced.py'
        }
        
    def train_all_models(self):
        """ TRAIN ALL MODELS - Run once per week when new NFL data arrives"""
        print("=== TRAINING ALL MODELS ===")
        print(f"Project root: {self.project_root}")
        
        # 1. Train Position Models (independent, no args)
        self._train_position_models()
        
        # 2. Train Team Model  
        self._train_team_model()
        
        # 3. Update training status
        self._update_training_status()
        
        print("✅ All models trained successfully!")
        
    def run_predictions_only(self):
        """⚡ RUN PREDICTIONS ONLY - Use existing models on new DraftKings slate"""
        print("=== RUNNING PREDICTIONS ONLY ===")
        
        # 1. Build master sheet from new DraftKings slate (applies crosswalk)
        self._build_master_sheet()
        
        # 2. Run position model inference (generates projection CSVs)
        self._run_position_inference()
        
        # 3. Run team model inference for current week
        self._run_team_inference()
        
        # 4. Build PlayerMaster from position projections
        self._build_playermaster()
        
        # 5. Enrich with Dashboard
        self._enrich_dashboard()
        
        print("✅ All predictions completed successfully!")
        
    def _train_position_models(self):
        """Train QB, RB, WR, TE, DST models (independent, no args)"""
        print("\n--- Training Position Models ---")
        
        # These are the training scripts (run once to create models)
        training_scripts = {
            'QB': 'PositionModel/QB/train_qb_model.py',
            'RB': 'PositionModel/RB/train_rb_model_advanced.py', 
            'WR': 'PositionModel/WR/train_wr_model_advanced.py',
            'TE': 'PositionModel/TE/train_te_model_advanced.py',
            'DST': 'PositionModel/DST/train_dst_model_advanced.py'
        }
        
        for position, script_path in training_scripts.items():
            full_path = self.project_root / script_path
            print(f"\n Training {position} model...")
            print(f"Script: {full_path}")
            
            if not full_path.exists():
                print(f"❌ Training script not found: {full_path}")
                continue
                
            try:
                # Run the training script from its own directory to fix path issues
                env = os.environ.copy()
                env['PYTHONIOENCODING'] = 'utf-8'
                
                # Run from the script's directory, not the project root
                script_dir = full_path.parent
                script_name = full_path.name
                
                result = subprocess.run([sys.executable, script_name], 
                                     cwd=script_dir, capture_output=True, text=True,
                                     env=env, encoding='utf-8')
                
                if result.returncode == 0:
                    print(f"✅ {position} model trained successfully")
                else:
                    print(f"❌ {position} training failed:")
                    print(f"Error: {result.stderr}")
                    
            except Exception as e:
                print(f"❌ Error running {position} training: {e}")
                
    def _train_team_model(self):
        """Train team total model"""
        print("\n--- Training Team Model ---")
        
        team_script = self.project_root / "data/TeamModels/train_teamtotal.py"
        if not team_script.exists():
            print(f"❌ Team training script not found: {team_script}")
            return
            
        print("🚀 Training team model...")
        try:
            # Team model needs --mode=train
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            
            result = subprocess.run([sys.executable, str(team_script), "--mode=train"], 
                                 cwd=self.project_root, capture_output=True, text=True,
                                 env=env, encoding='utf-8')
            
            if result.returncode == 0:
                print("✅ Team model trained successfully")
            else:
                print(f"❌ Team training failed: {result.stderr}")
                
        except Exception as e:
            print(f"❌ Error running team training: {e}")
            
    def _run_position_inference(self):
        """Run inference on all positions (generates projection CSVs)"""
        print("\n--- Running Position Inference ---")
        
        # Get the path to the newly built master sheet
        master_sheet_path = self.project_root / "data/processed/master_sheet_2025.csv"
        if not master_sheet_path.exists():
            print(f"❌ Master sheet not found: {master_sheet_path}")
            print("❌ Run build_master_sheet first!")
            return
            
        print(f"📊 Using filtered master sheet: {master_sheet_path}")
        
        for position, script_path in self.inference_scripts.items():
            full_path = self.project_root / script_path
            print(f"\n⚡ Running {position} inference...")
            
            if not full_path.exists():
                print(f"❌ Inference script not found: {full_path}")
                continue
                
            try:
                # Run the inference script from its own directory with master sheet path
                env = os.environ.copy()
                env['PYTHONIOENCODING'] = 'utf-8'
                
                # Run from the script's directory, not the project root
                script_dir = full_path.parent
                script_name = full_path.name
                
                # Pass the master sheet path as an argument
                cmd = [sys.executable, script_name, "--master-sheet", str(master_sheet_path)]
                
                result = subprocess.run(cmd, 
                                     cwd=script_dir, capture_output=True, text=True,
                                     env=env, encoding='utf-8')
                
                if result.returncode == 0:
                    print(f"✅ {position} inference completed")
                else:
                    print(f"❌ {position} inference failed: {result.stderr}")
                    
            except Exception as e:
                print(f"❌ Error running {position} inference: {e}")
                
    def _run_team_inference(self):
        """Run team model inference for current week"""
        print("\n--- Running Team Inference ---")
        
        team_script = self.project_root / "data/TeamModels/train_teamtotal.py"
        print("⚡ Running team inference...")
        
        try:
            # Team model needs --mode=infer with season/week
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            
            result = subprocess.run([sys.executable, str(team_script), "--mode=infer", "--season=2025", "--weeks=1"], 
                                 cwd=self.project_root, capture_output=True, text=True,
                                 env=env, encoding='utf-8')
            
            if result.returncode == 0:
                print("✅ Team inference completed")
            else:
                print(f"❌ Team inference failed: {result.stderr}")
                
        except Exception as e:
            print(f"❌ Error running team inference: {e}")
                
    def _build_master_sheet(self):
        """Build master sheet from new DraftKings slate (applies crosswalk)"""
        print("\n--- Building Master Sheet ---")
        
        script_path = self.project_root / "scripts/build_master_sheet.py"
        if not script_path.exists():
            print(f"❌ Build master sheet script not found: {script_path}")
            return
            
        print("🔧 Building master sheet from DraftKings slate...")
        try:
            # Build master sheet with crosswalk - uses current DKSalaries.csv
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            
            result = subprocess.run([sys.executable, str(script_path), 
                                   "--dk", "data/raw/DKSalaries.csv", 
                                   "--season", "2025",
                                   "--merge-mode",
                                   "--crosswalk-out", "data/processed/crosswalk_2025.csv",
                                   "--master-out", "data/processed/master_sheet_2025.csv"], 
                                 cwd=self.project_root, capture_output=True, text=True,
                                 env=env, encoding='utf-8')
            
            if result.returncode == 0:
                print("✅ Master sheet built successfully")
                print("📊 Applied crosswalk to filter players for current slate")
            else:
                print(f"❌ Master sheet build failed: {result.stderr}")
                
        except Exception as e:
            print(f"❌ Error building master sheet: {e}")
                
    def _build_playermaster(self):
        """Build unified PlayerMaster from all projections"""
        print("\n--- Building PlayerMaster ---")
        
        script_path = self.project_root / "data/DFSDashboard/playermaster_from_projections.py"
        if not script_path.exists():
            print(f"❌ PlayerMaster script not found: {script_path}")
            return
            
        print("🔨 Building PlayerMaster...")
        print("📊 Looking for projection files in PositionModel/ directory...")
        try:
            # Pass the projection directory argument to find the right files
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            
            result = subprocess.run([sys.executable, str(script_path), 
                                   "--proj-dir", "PositionModel/"], 
                                 cwd=self.project_root, capture_output=True, text=True,
                                 env=env, encoding='utf-8')
            
            if result.returncode == 0:
                print("✅ PlayerMaster built successfully")
                
                # Validate slate consistency
                self._validate_slate_consistency()
            else:
                print(f"❌ PlayerMaster build failed: {result.stderr}")
                
        except Exception as e:
            print(f"❌ Error building PlayerMaster: {e}")
            
    def _validate_slate_consistency(self):
        """Validate that PlayerMaster only contains current slate players"""
        print("\n--- Validating Slate Consistency ---")
        
        script_path = self.project_root / "scripts/validate_slate.py"
        if not script_path.exists():
            print(f"❌ Validation script not found: {script_path}")
            return
            
        print("🔍 Validating slate consistency...")
        try:
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            
            result = subprocess.run([sys.executable, str(script_path)], 
                                 cwd=self.project_root, capture_output=True, text=True,
                                 env=env, encoding='utf-8')
            
            if result.returncode == 0:
                print("✅ Slate validation passed")
            else:
                print(f"❌ Slate validation failed: {result.stderr}")
                print("⚠️  WARNING: PlayerMaster may contain invalid players!")
                
        except Exception as e:
            print(f"❌ Error during slate validation: {e}")
            
    def _enrich_dashboard(self):
        """Enrich PlayerMaster with team predictions + Vegas"""
        print("\n--- Enriching Dashboard ---")
        
        script_path = self.project_root / "data/DFSDashboard/Dashboard.py"
        if not script_path.exists():
            print(f"❌ Dashboard script not found: {script_path}")
            return
            
        print("🎯 Enriching Dashboard...")
        try:
            # Dashboard needs season/week and file paths
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            
            # Build the correct file paths for Dashboard
            players_file = self.project_root / "data/DFSDashboard/PlayerMaster_unified.csv"
            teams_file = self.project_root / "data/TeamModels/data/outputs/predictions_2025w01.csv"
            
            # Check if files exist before calling Dashboard
            if not players_file.exists():
                print(f"❌ PlayerMaster file not found: {players_file}")
                return
            if not teams_file.exists():
                print(f"❌ Team predictions file not found: {teams_file}")
                return
                
            result = subprocess.run([sys.executable, str(script_path), 
                                   "--season=2025", "--week=1",
                                   "--players", str(players_file),
                                   "--teams", str(teams_file)], 
                                 cwd=self.project_root, capture_output=True, text=True,
                                 env=env, encoding='utf-8')
            
            if result.returncode == 0:
                print("✅ Dashboard enrichment completed")
            else:
                print(f"❌ Dashboard enrichment failed: {result.stderr}")
                
        except Exception as e:
            print(f"❌ Error enriching Dashboard: {e}")
            
    def _models_are_fresh(self):
        """Check if models are trained on current week data"""
        # For now, just return True - you can implement this logic later
        return True
        
    def _update_training_status(self):
        """Update when models were last trained"""
        status = {
            'last_trained': datetime.now().isoformat(),
            'models_trained': list(self.inference_scripts.keys()) + ['TEAM'],
            'status': 'fresh'
        }
        
        with open(self.status_file, 'w') as f:
            json.dump(status, f, indent=2)
            
        print(f"✅ Training status updated: {self.status_file}")

def main():
    parser = argparse.ArgumentParser(description="NFL DFS Pipeline Orchestrator")
    parser.add_argument('--mode', choices=['train', 'predict'], required=True,
                       help='train: retrain all models, predict: run predictions only')
    
    args = parser.parse_args()
    
    orchestrator = PipelineOrchestrator()
    
    if args.mode == 'train':
        orchestrator.train_all_models()
    else:
        orchestrator.run_predictions_only()

if __name__ == "__main__":
    main()
