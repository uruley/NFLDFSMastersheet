#!/usr/bin/env python3
"""
Analyze TE Performance from 2024 Season
Compare actual averages to model predictions
"""
import pandas as pd
import numpy as np
import nfl_data_py as nfl

def load_te_2024_data():
    """Load all TE data from 2024 season."""
    print("Loading 2024 TE data...")
    
    # Load weekly data for 2024
    weekly_data = nfl.import_weekly_data([2024])
    te_data = weekly_data[weekly_data['position'] == 'TE'].copy()
    
    print(f"Loaded {len(te_data)} TE weekly records for 2024")
    return te_data

def calculate_te_averages(te_data):
    """Calculate season averages for each TE."""
    print("Calculating season averages...")
    
    # Calculate PPR fantasy points
    te_data['fantasy_points_ppr'] = (
        te_data['receptions'] * 1.0 +
        te_data['receiving_yards'] * 0.1 +
        te_data['rushing_yards'] * 0.1 +
        te_data['receiving_tds'] * 6.0 +
        te_data['rushing_tds'] * 6.0 +
        te_data.get('return_tds', 0) * 6.0 +
        te_data.get('two_point_conversions', 0) * 2.0 -
        te_data.get('fumbles_lost', 0) * 2.0
    ).fillna(0)
    
    # Group by player and calculate averages
    te_averages = te_data.groupby(['player_id', 'player_name', 'recent_team']).agg({
        'fantasy_points_ppr': ['mean', 'std', 'min', 'max', 'count'],
        'receptions': 'mean',
        'targets': 'mean',
        'receiving_yards': 'mean',
        'receiving_tds': 'mean',
        'rushing_yards': 'mean',
        'rushing_tds': 'mean'
    }).round(2)
    
    # Flatten column names
    te_averages.columns = ['_'.join(col).strip() for col in te_averages.columns]
    te_averages = te_averages.reset_index()
    
    # Filter for players with at least 5 games
    te_averages = te_averages[te_averages['fantasy_points_ppr_count'] >= 5].copy()
    
    # Sort by average fantasy points
    te_averages = te_averages.sort_values('fantasy_points_ppr_mean', ascending=False)
    
    print(f"Calculated averages for {len(te_averages)} TEs with 5+ games")
    return te_averages

def analyze_top_tes(te_averages, top_n=20):
    """Analyze top performing TEs."""
    print(f"\n=== TOP {top_n} TE AVERAGES (2024 SEASON) ===")
    
    top_tes = te_averages.head(top_n).copy()
    
    # Display key stats
    display_cols = [
        'player_name', 'recent_team', 'fantasy_points_ppr_mean', 
        'fantasy_points_ppr_std', 'fantasy_points_ppr_count',
        'receptions_mean', 'targets_mean', 'receiving_yards_mean', 'receiving_tds_mean'
    ]
    
    print(top_tes[display_cols].to_string(index=False))
    
    # Summary statistics
    print(f"\n=== SUMMARY STATISTICS ===")
    print(f"Top 10 TE Average: {top_tes.head(10)['fantasy_points_ppr_mean'].mean():.2f} points")
    print(f"Top 20 TE Average: {top_tes.head(20)['fantasy_points_ppr_mean'].mean():.2f} points")
    print(f"Overall TE Average: {te_averages['fantasy_points_ppr_mean'].mean():.2f} points")
    
    return top_tes

def compare_to_predictions(te_averages, predictions_file):
    """Compare actual 2024 averages to model predictions."""
    print(f"\n=== COMPARING ACTUAL 2024 AVERAGES TO MODEL PREDICTIONS ===")
    
    try:
        predictions = pd.read_csv(predictions_file)
        print(f"Loaded predictions from {predictions_file}")
        
        # Filter predictions for 2024 season
        pred_2024 = predictions[predictions['season'] == 2024].copy()
        
        if len(pred_2024) == 0:
            print("No 2024 predictions found in file")
            return
        
        # Merge actual averages with predictions
        comparison = te_averages.merge(
            pred_2024[['player_id', 'predicted_points_ensemble', 'actual_points']], 
            on='player_id', 
            how='inner'
        )
        
        if len(comparison) == 0:
            print("No matching players found between averages and predictions")
            return
        
        # Calculate prediction accuracy
        comparison['prediction_error'] = comparison['predicted_points_ensemble'] - comparison['fantasy_points_ppr_mean']
        comparison['abs_error'] = abs(comparison['prediction_error'])
        
        print(f"\nFound {len(comparison)} matching players for comparison")
        
        # Show comparison for top performers
        top_comparison = comparison.head(15)[[
            'player_name', 'recent_team', 'fantasy_points_ppr_mean', 
            'predicted_points_ensemble', 'prediction_error', 'abs_error'
        ]].copy()
        
        print("\nTop 15 TEs - Actual vs Predicted:")
        print(top_comparison.to_string(index=False))
        
        # Summary of prediction accuracy
        print(f"\n=== PREDICTION ACCURACY ===")
        print(f"Mean Absolute Error: {comparison['abs_error'].mean():.2f} points")
        print(f"Mean Error: {comparison['prediction_error'].mean():.2f} points")
        print(f"Root Mean Square Error: {np.sqrt((comparison['prediction_error']**2).mean()):.2f} points")
        
        # Check for systematic bias
        high_performers = comparison[comparison['fantasy_points_ppr_mean'] >= 10]
        if len(high_performers) > 0:
            print(f"\nHigh Performers (10+ avg points) - Mean Error: {high_performers['prediction_error'].mean():.2f}")
        
        low_performers = comparison[comparison['fantasy_points_ppr_mean'] < 10]
        if len(low_performers) > 0:
            print(f"Low Performers (<10 avg points) - Mean Error: {low_performers['prediction_error'].mean():.2f}")
            
    except Exception as e:
        print(f"Error comparing to predictions: {e}")

def main():
    """Main analysis function."""
    print("🔍 TE 2024 Season Analysis")
    print("=" * 50)
    
    # Load and analyze 2024 data
    te_data = load_te_2024_data()
    te_averages = calculate_te_averages(te_data)
    
    # Analyze top performers
    top_tes = analyze_top_tes(te_averages, top_n=20)
    
    # Compare to model predictions if available
    predictions_file = "te_predictions_advanced_new.csv"
    compare_to_predictions(te_averages, predictions_file)
    
    # Save results
    output_file = "te_2024_averages.csv"
    te_averages.to_csv(output_file, index=False)
    print(f"\n✅ TE averages saved to {output_file}")
    
    # Additional insights
    print(f"\n=== KEY INSIGHTS ===")
    print(f"• Top TE average: {te_averages.iloc[0]['fantasy_points_ppr_mean']:.2f} points")
    print(f"• 10th best TE average: {te_averages.iloc[9]['fantasy_points_ppr_mean']:.2f} points")
    print(f"• 20th best TE average: {te_averages.iloc[19]['fantasy_points_ppr_mean']:.2f} points")
    
    # Check for high variance players
    high_variance = te_averages[te_averages['fantasy_points_ppr_std'] > 8].head(5)
    if len(high_variance) > 0:
        print(f"\n• High variance TEs (std > 8): {len(high_variance)} players")
        print("  These players may be harder to predict consistently")

if __name__ == "__main__":
    main()
