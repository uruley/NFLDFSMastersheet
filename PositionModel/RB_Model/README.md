# 🏃‍♂️ RB Model - Running Back Fantasy Football Predictions

This directory contains the complete RB (Running Back) prediction model for fantasy football DFS.

## 📁 Files

- **`train_rb_model.py`** - Basic script to train the RB prediction model (Random Forest only)
- **`train_rb_model_advanced.py`** - Advanced script to train multiple ML models with hyperparameter optimization
- **`requirements_advanced.txt`** - Dependencies for advanced training (LightGBM, CatBoost, Optuna)
- **`rb_inference.py`** - Script to make RB predictions using the trained model
- **`rb_model.pkl`** - Trained machine learning model (created after training)
- **`encoders.pkl`** - Label encoders for categorical features (created after training)
- **`feature_columns.json`** - List of features used by the model (created after training)
- **`rb_predictions.csv`** - Output predictions (created after running inference)

## 🚀 Quick Start

### Basic Training (Random Forest Only)
```bash
cd PositionModel/RB_Model
python train_rb_model.py
```

This will:
- Load 5 seasons of RB data (2020-2024) from NFL API
- Create RB-specific features (rushing, receiving, efficiency)
- Train a Random Forest model
- Save all artifacts to the RB_Model directory

### Advanced Training (Multiple ML Frameworks)
```bash
cd PositionModel/RB_Model
pip install -r requirements_advanced.txt
python train_rb_model_advanced.py
```

This will:
- Test multiple ML algorithms (LightGBM, CatBoost, Random Forest, Gradient Boosting)
- Use Optuna for hyperparameter optimization
- Compare all models and select the best performer
- Save detailed metrics and model comparisons

### 2. Make RB Predictions
```bash
python rb_inference.py
```

This will:
- Load the trained model and encoders
- Process the current DraftKings slate
- Match RBs to historical data
- Generate predictions for all RBs
- Save results to `rb_predictions.csv`

## 🏈 RB-Specific Features

### Core Rushing Features
- **carries** - Number of rushing attempts
- **rushing_yards** - Total rushing yards
- **rushing_tds** - Rushing touchdowns
- **rushing_epa** - Expected Points Added for rushing
- **rushing_first_downs** - First downs gained via rushing
- **yards_per_carry** - Average yards per rushing attempt

### Core Receiving Features
- **receptions** - Number of catches
- **receiving_yards** - Total receiving yards
- **receiving_tds** - Receiving touchdowns
- **targets** - Number of passes thrown to the RB
- **receiving_epa** - Expected Points Added for receiving
- **yards_per_reception** - Average yards per catch

### Derived Features
- **total_touches** - carries + receptions
- **total_yards** - rushing_yards + receiving_yards
- **total_tds** - rushing_tds + receiving_tds
- **yards_per_touch** - Average yards per touch (rush or catch)
- **touchdown_rate** - TDs per touch

### Context Features
- **week** - Week of the season
- **season** - NFL season
- **early_season** - Binary flag for weeks 1-4
- **mid_season** - Binary flag for weeks 5-12
- **late_season** - Binary flag for weeks 13+
- **high_scoring_game** - Binary flag for games with total > 45
- **favorite/underdog** - Game spread context

## 📊 Model Performance

The RB model is trained on **5 seasons of data (2020-2024)** to capture:
- **Workload patterns** - How touches are distributed
- **Efficiency trends** - Yards per carry/catch over time
- **Role changes** - Starter vs backup usage
- **Matchup impact** - Performance vs different defenses

## 🔧 Customization

### Modify Training Seasons
Edit `train_rb_model.py`:
```python
seasons = [2020, 2021, 2022, 2023, 2024]  # Change seasons here
```

### Add New Features
Edit `create_rb_features()` function in `train_rb_model.py`:
```python
# Add your custom feature
feature_data['my_custom_feature'] = feature_data['existing_feature'] * 2
```

### Change Model Type
Edit `train_rb_model()` function in `train_rb_model.py`:
```python
# Replace RandomForest with your preferred model
from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor(n_estimators=100, random_state=42)
```

## 📈 Understanding Predictions

The model predicts **fantasy points** based on:
1. **Historical performance** - Recent rushing/receiving stats
2. **Efficiency metrics** - Yards per touch, touchdown rate
3. **Context factors** - Week of season, game script
4. **Matchup data** - Opponent defensive strength

## 🎯 Key Insights

- **Dual-threat RBs** (good at both rushing and receiving) typically score higher
- **Touchdown efficiency** is crucial for fantasy production
- **Workload consistency** matters more than raw talent
- **Game script** affects rushing vs receiving opportunities

## 🚨 Troubleshooting

### Common Issues
1. **"Model files not found"** - Run `train_rb_model.py` first
2. **"No RBs matched"** - Check master sheet and DK slate compatibility
3. **"Error loading NFL data"** - Check internet connection and nfl_data_py installation

### Data Requirements
- **NFL API access** via nfl_data_py
- **DraftKings slate** in expected format
- **Master sheet** with proper join keys
- **Python packages**: pandas, numpy, scikit-learn, joblib

## 🔄 Workflow

1. **Train model** (weekly or when new data available)
2. **Update DK slate** (new week's player list)
3. **Run inference** (generate predictions)
4. **Analyze results** (check feature importance, top picks)
5. **Use for DFS** (lineup construction, player selection)

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify all required files exist
3. Ensure proper data formats
4. Check Python package versions

---

**Happy RB modeling! 🏈📊**
