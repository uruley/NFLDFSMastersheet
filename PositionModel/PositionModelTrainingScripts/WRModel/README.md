# WR Model Training and Inference

This directory contains the advanced WR (Wide Receiver) fantasy football model training and inference pipeline.

## Files

- `train_wr_model_advanced.py` - Main training script for the WR model
- `infer_wr.py` - Inference pipeline for making predictions
- `example_inference.py` - Example usage of the inference pipeline
- `README.md` - This documentation file

## Model Overview

The WR model is designed to predict next-week fantasy points for wide receivers using:
- **No target leakage**: All features come from prior weeks only
- **Time-ordered splits**: Train/validation/test split by chronological order
- **Opponent context**: "WR-allowed" metrics from prior weeks
- **Robust feature engineering**: Handles missing data gracefully

## Training the Model

### Prerequisites

Install required packages:
```bash
pip install pandas numpy scikit-learn lightgbm catboost
```

### Basic Training

```bash
python train_wr_model_advanced.py --stats-csv path/to/weekly_stats.csv --outdir WRModel
```

### Training Options

```bash
python train_wr_model_advanced.py \
    --stats-csv path/to/weekly_stats.csv \
    --outdir WRModel \
    --train-frac 0.70 \
    --val-frac 0.15 \
    --n-estimators 1200 \
    --learning-rate 0.03 \
    --max-depth -1 \
    --verbose
```

### Training Parameters

- `--stats-csv`: Path to weekly WR stats CSV (or omit to use nfl_data_py)
- `--outdir`: Directory to save model artifacts (default: WRModel)
- `--train-frac`: Fraction of weeks for training (default: 0.70)
- `--val-frac`: Fraction of weeks for validation (default: 0.15)
- `--n-estimators`: Number of boosting rounds (default: 1200)
- `--learning-rate`: Learning rate for gradient boosting (default: 0.03)
- `--max-depth`: Maximum tree depth (default: -1 for unlimited)
- `--verbose`: Enable verbose training output

### Training Output

The training script saves:
- `lightgbm_model.pkl` - Trained LightGBM model
- `encoders.pkl` - Label encoders for categorical variables
- `feature_schema.json` - Feature column names and types
- `metrics.json` - Training/validation/test metrics
- `feature_importance.csv` - Feature importance rankings
- `summary.json` - Complete training summary

## Making Predictions

### Using the Inference Script

```bash
python infer_wr.py \
    --model-dir WRModel \
    --weekly-stats path/to/new_weekly_stats.csv \
    --salaries path/to/dk_salaries.csv \
    --output predictions.csv
```

### Required Input Data

#### Weekly Stats CSV
Must contain these columns:
- `season`, `week`, `player_id`, `name`, `position`
- `fantasy_points`, `receptions`, `targets`, `receiving_yards`
- `snap_count`, `opponent_team`

#### DraftKings Salaries CSV
Must contain:
- `player_id`, `team`, `opponent`, `salary`

### Inference Output

The inference script produces:
- `predictions.csv` - Player predictions with predicted fantasy points
- `inference_diagnostics.txt` - Feature validation diagnostics

## Example Usage

### Quick Test

```bash
# Run the example inference with sample data
python example_inference.py
```

This will:
1. Check if model artifacts exist
2. Create sample data files
3. Run inference
4. Display results
5. Clean up sample files

### Production Inference

```bash
# Load your actual data and make predictions
python infer_wr.py \
    --model-dir /path/to/trained/model \
    --weekly-stats /path/to/current_week_stats.csv \
    --salaries /path/to/dk_slate.csv \
    --output week_18_predictions.csv
```

## Data Requirements

### Training Data Format

The training script expects weekly WR statistics with:
- Historical data spanning multiple seasons
- Consistent player IDs across weeks
- Fantasy points calculated (or raw stats to calculate from)
- Position filtering for WR only

### Inference Data Format

For inference, you need:
- Current week's WR statistics
- Same column structure as training data
- DraftKings salary information
- Player ID consistency with training data

## Troubleshooting

### Common Issues

1. **Missing Model Artifacts**
   - Ensure you've trained the model first
   - Check that all required files exist in the model directory

2. **Column Mismatches**
   - Verify your data has the same column names as training data
   - Check for typos in column names

3. **Feature Alignment Errors**
   - Ensure all required features can be generated
   - Check that rolling features have sufficient historical data

4. **Encoding Errors**
   - Verify categorical variables match training data
   - Check for new teams/opponents not seen during training

### Validation

The inference script includes validation for:
- Feature completeness
- Data quality checks
- Player team consistency
- Feature distribution analysis

## Model Performance

The model typically achieves:
- **R²**: 0.15-0.25 on test data
- **MAE**: 3-5 fantasy points
- **RMSE**: 4-6 fantasy points

Performance varies by season and data quality.

## Advanced Features

### Multiple Model Support
The training script also saves:
- CatBoost model (if available)
- Random Forest model
- Gradient Boosting model

### Quantile Regression
Support for quantile regression objectives:
```bash
python train_wr_model_advanced.py \
    --objective quantile \
    --quantile-alpha 0.75
```

### Custom Splits
Adjust train/validation/test proportions:
```bash
python train_wr_model_advanced.py \
    --train-frac 0.80 \
    --val-frac 0.10
```

## Integration

This pipeline integrates with:
- NFL API data sources
- DraftKings salary data
- Fantasy football optimization tools
- Automated lineup generation systems

## Support

For issues or questions:
1. Check the inference diagnostics file
2. Verify data format matches requirements
3. Ensure model artifacts are complete
4. Review training logs for errors
