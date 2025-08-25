# QB Model - NFL DFS Prediction System

## 🏈 Overview
The QB Model is a machine learning system that predicts fantasy football points for NFL quarterbacks using historical performance data, salary information, and team context. The system automatically identifies backup QBs and applies conservative projections to prevent over-prediction.

## 📁 File Structure

### Core Model Files
- **`qb_model.pkl`** (200KB) - Trained LightGBM model for QB predictions
- **`encoders.pkl`** (753B) - Categorical encoders for team and opponent data
- **`feature_columns.json`** (705B) - List of 33 features used by the model

### Inference Scripts
- **`qb_inference_fixed.py`** (27KB) - Main inference script with backup QB detection
- **`qb_predictions_FIXED.csv`** (6.2KB) - Latest predictions output

## 🔧 Model Architecture

### Model Type
- **Algorithm**: LightGBM (Gradient Boosting)
- **Training Data**: NFL API weekly data (2023-2024 seasons)
- **Features**: 33 total features including passing, rushing, receiving stats

### Key Features
- **Passing Stats**: Yards, TDs, completions, attempts, EPA, air yards
- **Rushing Stats**: Yards, TDs, carries, fumbles, EPA
- **Receiving Stats**: Yards, TDs, targets, air yards, EPA
- **Context**: Team encoding, opponent encoding, season/week progression
- **Advanced Metrics**: PACR, RACR, WOPR, DAKOTA ratings

## 🚀 Workflow

### 1. Data Loading
- **DK Slate**: Loads DraftKings salary data for current week
- **Master Sheet**: Matches players using NFL API IDs
- **NFL Data**: Caches weekly performance data (2023-2024)

### 2. Player Processing
- **Name Matching**: Uses cleaned names and team abbreviations
- **Join Keys**: Creates unique identifiers: `clean_name|team|position`
- **Feature Extraction**: Pulls historical performance data

### 3. Backup QB Detection
The system automatically identifies backup QBs using:
- **Known Backups**: Drew Lock, Jameis Winston, Mason Rudolph, etc.
- **Salary Threshold**: QBs under $5,000 salary
- **Team Depth Charts**: Week 1 specific backup situations

### 4. Feature Engineering
- **Historical Averages**: Last 3 games performance
- **Derived Features**: Completion rate, yards per attempt, total touches
- **Season Context**: Early/mid/late season patterns

### 5. Prediction & Output
- **Model Inference**: LightGBM prediction with clipping (0-50 points)
- **Value Calculation**: Points per $1000 salary
- **CSV Export**: Complete predictions with metadata

## 🎯 Key Features

### Backup QB Handling
```python
# Automatic backup detection prevents over-prediction
if is_backup_qb(qb_name, salary, team, current_week):
    return get_backup_baseline_features(qb_name, salary, team, backup_info)
```

### Varied Fallbacks
- **No Identical Values**: Each player gets unique baseline projections
- **Salary-Based Tiers**: Different expectations by salary range
- **Deterministic Randomness**: Consistent but varied projections

### Team/Opponent Encoding
- **Team Mapping**: Handles DK vs NFL API team abbreviation differences
- **Encoder Fallbacks**: Graceful handling of missing team data
- **Opponent Context**: Game-specific matchup information

## 📊 Output Format

### Prediction Columns
- **Name, Team, Opponent, Position** - Player identification
- **Salary** - DraftKings salary for value calculation
- **Predicted_Points** - Model output (0-50 range)
- **Value** - Points per $1000 salary
- **Recent_Avg_Points** - Historical baseline
- **Match_Status** - Success of player matching

### Example Output
```
Name: Drew Lock
Team: SEA
Opponent: SF
Salary: 4300
Predicted_Points: 3.82
Value: 0.888
Recent_Avg_Points: 4.0
Match_Status: Matched
```

## 🚀 Usage

### Running Predictions
```bash
cd PositionModel/QB_Model
python qb_inference_fixed.py
```

### Input Requirements
- **DK Slate**: `../../data/raw/dk_slates/2025/DKSalaries_20250824.csv`
- **Master Sheet**: `../../data/processed/master_sheet_2025.csv`
- **Model Files**: All `.pkl` and `.json` files in directory

### Output Files
- **`qb_predictions_FIXED.csv`** - Complete predictions
- **Console Output** - Detailed processing logs and top performers

## 🔍 Model Performance

### Current Capabilities
- **Backup QB Detection**: Prevents over-prediction of non-starters
- **Historical Data Integration**: Uses NFL API for accurate player stats
- **Salary-Aware Projections**: Realistic expectations based on cost
- **Team Context**: Considers opponent and team situations

### Recent Improvements
- ✅ **Fixed Backup QB Logic**: Drew Lock now projects 3.82 vs 20.26
- ✅ **Eliminated Identical Projections**: Each backup gets unique baseline
- ✅ **Enhanced Fallback System**: Varied projections instead of league averages

## 🛠️ Technical Details

### Dependencies
- **pandas, numpy** - Data manipulation
- **nfl_data_py** - NFL API integration
- **pickle** - Model serialization
- **lightgbm** - Machine learning model

### Data Sources
- **DraftKings**: Current week salaries and matchups
- **NFL API**: Historical player performance data
- **Master Sheet**: Player ID mapping and metadata

### Performance Considerations
- **Cached Data**: NFL weekly data loaded once per session
- **Efficient Matching**: Join key optimization for player lookup
- **Memory Management**: Streaming processing for large datasets

## 🔮 Future Enhancements

### Planned Improvements
- **Ensemble Models**: Multiple algorithm predictions (see QB_Model_Advanced)
- **Injury Reports**: Integration with injury status data
- **Weather Data**: Game day weather impact on passing
- **Vegas Lines**: Point spread and over/under integration

### Advanced Features
- **Player Props**: Integration with betting market data
- **Real-time Updates**: Live injury and roster changes
- **Multi-week Projections**: Season-long fantasy planning

## 📝 Notes

- **Current Season**: 2025
- **Current Week**: 1
- **Model Version**: Fixed (with backup QB logic)
- **Last Updated**: Latest backup QB detection implementation
- **Status**: Production ready with comprehensive backup handling

---

*This QB model represents a production-ready system that automatically handles the complexities of NFL DFS, including backup QB detection, historical data integration, and realistic projection generation.*
