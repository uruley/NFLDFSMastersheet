# 🏈 Advanced TE Position Model

This is an advanced Tight End (TE) fantasy football prediction model that follows the same pattern as your successful WR model.

## 🚀 Features

- **Ensemble Model**: Combines LightGBM, CatBoost, Random Forest, and Gradient Boosting
- **Optuna Hyperparameter Tuning**: Automatic optimization of model parameters
- **NFL API Integration**: Uses `nfl_data_py` to load data directly from NFL API
- **Feature Engineering**: Creates TE-specific derived features
- **Next Week Prediction**: Predicts fantasy points for the upcoming week
- **Multiple Seasons**: Trained on data from 2020-2024

## 📁 Files

- `train_te_model_advanced.py` - Main training script
- `infer_te_advanced.py` - Inference script for making predictions
- `requirements.txt` - Python dependencies
- `README.md` - This documentation

## 🎯 Training the Model

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Training
```bash
python train_te_model_advanced.py
```

**What happens during training:**
- Loads TE data from NFL API (2020-2024)
- Creates derived features (catch_rate, yards_per_reception, etc.)
- Performs feature selection (top 20 features)
- Trains 4 models with Optuna hyperparameter tuning
- Creates ensemble predictions
- Saves all models and artifacts to `TEModel_Advanced/`

## 🔮 Making Predictions

### 1. Run Inference
```bash
python infer_te_advanced.py --seasons 2024 --output te_predictions.csv
```

**Options:**
- `--seasons`: Seasons to predict (default: 2024)
- `--output`: Output CSV file (default: te_predictions.csv)
- `--model-dir`: Model directory (default: TEModel_Advanced)

### 2. Output Format
The inference script generates:
- Individual model predictions (LightGBM, CatBoost, Random Forest, Gradient Boosting)
- Ensemble predictions (average of all models)
- Sorted by highest predicted points

## 🏗️ Model Architecture

### **Feature Engineering:**
- **Efficiency Metrics**: catch_rate, yards_per_reception, yards_per_target, yards_per_rush
- **Total Metrics**: total_yards, total_tds, total_touches
- **Season Progression**: early_season, mid_season, late_season, week_progression
- **Categorical**: team_encoded, opponent_encoded, season_type_encoded

### **Model Ensemble:**
1. **LightGBM**: Gradient boosting with Optuna tuning
2. **CatBoost**: Advanced gradient boosting with Optuna tuning
3. **Random Forest**: Ensemble of decision trees
4. **Gradient Boosting**: Traditional gradient boosting

### **Target Variable:**
- **Next Week's Fantasy Points**: Uses `shift(-1)` to predict future performance
- **Realistic Predictions**: Not current week performance

## 📊 Expected Results

- **Training Data**: ~6,775 TE weekly records
- **Features**: 20 selected from 50+ available
- **Performance**: R² around 0.13-0.15 (appropriate for next-week prediction)
- **Output**: CSV with predictions for all TEs in specified seasons

## 🔧 Customization

### **Adding New Features:**
Edit `create_derived_features()` function in training script

### **Changing Model Parameters:**
Modify hyperparameter ranges in `objective_lightgbm()` and `objective_catboost()`

### **Different Seasons:**
Change `seasons` parameter in `load_te_data()` function

## 🚨 Troubleshooting

### **Common Issues:**
1. **Missing Dependencies**: Install with `pip install -r requirements.txt`
2. **Model Not Found**: Ensure training completed successfully
3. **Feature Mismatch**: Check that feature schema matches training data

### **Data Issues:**
- Verify NFL API access
- Check season availability (2020-2024)
- Ensure TE position data exists

## 📈 Performance Notes

- **Lower R² is Expected**: Next-week prediction is inherently difficult
- **Ensemble Approach**: Reduces overfitting and improves stability
- **Feature Selection**: Focuses on most predictive features
- **Hyperparameter Tuning**: Optimizes each model individually

## 🎯 Next Steps

After training:
1. **Validate Model**: Check feature importance and model performance
2. **Run Inference**: Generate predictions for current season
3. **Monitor Performance**: Track prediction accuracy over time
4. **Iterate**: Retrain with new data or adjusted features

---

**Model follows the exact same successful pattern as your WR model!** 🚀
