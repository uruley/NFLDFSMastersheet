import nfl_data_py as nfl
import pandas as pd

print("=== COMPARING FANTASY SCORING SYSTEMS ===")
print()

# Get QB weekly data
weekly_data = nfl.import_weekly_data([2023, 2024])
qb_weekly = weekly_data[weekly_data['position'] == 'QB'].copy()

# Calculate DraftKings fantasy points manually
def calculate_dk_points(row):
    """Calculate DraftKings fantasy points using official scoring rules."""
    dk_points = 0
    
    # Passing
    dk_points += (row['passing_yards'] / 25)  # 1 pt per 25 passing yards
    dk_points += (row['passing_tds'] * 4)     # 4 pts per passing TD
    dk_points += (row['interceptions'] * -1)  # -1 pt per interception
    
    # Rushing
    dk_points += (row['rushing_yards'] / 10)  # 1 pt per 10 rushing yards
    dk_points += (row['rushing_tds'] * 6)     # 6 pts per rushing TD
    
    # Receiving (QBs can catch their own passes)
    dk_points += row['receptions']             # 1 pt per reception
    dk_points += (row['receiving_yards'] / 10) # 1 pt per 10 receiving yards
    dk_points += (row['receiving_tds'] * 6)   # 6 pts per receiving TD
    
    # 2-point conversions
    dk_points += (row['passing_2pt_conversions'] * 2)
    dk_points += (row['rushing_2pt_conversions'] * 2)
    
    # Fumbles lost
    dk_points += (row['rushing_fumbles_lost'] * -1)
    dk_points += (row['receiving_fumbles_lost'] * -1)
    
    # Yardage bonuses
    if row['passing_yards'] >= 300:
        dk_points += 3  # +3 pts for 300+ passing yards
    if row['rushing_yards'] >= 100:
        dk_points += 3  # +3 pts for 100+ rushing yards
    if row['receiving_yards'] >= 100:
        dk_points += 3  # +3 pts for 100+ receiving yards
    
    return round(dk_points, 2)

# Calculate DK points for each QB
qb_weekly['fantasy_points_dk_calculated'] = qb_weekly.apply(calculate_dk_points, axis=1)

# Compare the scoring systems
print("=== SCORING COMPARISON ===")
print(f"Sample size: {len(qb_weekly)} QB weekly performances")
print()

# Show correlation between scoring systems
correlation = qb_weekly['fantasy_points'].corr(qb_weekly['fantasy_points_dk_calculated'])
print(f"Correlation between standard and DK scoring: {correlation:.4f}")
print()

# Show scoring differences
qb_weekly['scoring_diff'] = qb_weekly['fantasy_points_dk_calculated'] - qb_weekly['fantasy_points']

print("=== SCORING DIFFERENCES ===")
print(f"Average difference (DK - Standard): {qb_weekly['scoring_diff'].mean():.2f} points")
print(f"Standard deviation of difference: {qb_weekly['scoring_diff'].std():.2f} points")
print(f"Min difference: {qb_weekly['scoring_diff'].min():.2f} points")
print(f"Max difference: {qb_weekly['scoring_diff'].max():.2f} points")
print()

# Show examples with different difference ranges
print("=== EXAMPLES BY DIFFERENCE RANGE ===")

# Small differences (0-1 point)
small_diff = qb_weekly[abs(qb_weekly['scoring_diff']) <= 1].head(5)
print("SMALL DIFFERENCES (0-1 point):")
print(small_diff[['player_name', 'recent_team', 'week', 'season', 
                  'passing_yards', 'passing_tds', 'rushing_yards', 'rushing_tds',
                  'fantasy_points', 'fantasy_points_dk_calculated', 'scoring_diff']].to_string(index=False))
print()

# Medium differences (1-3 points)
medium_diff = qb_weekly[(abs(qb_weekly['scoring_diff']) > 1) & (abs(qb_weekly['scoring_diff']) <= 3)].head(5)
print("MEDIUM DIFFERENCES (1-3 points):")
print(medium_diff[['player_name', 'recent_team', 'week', 'season', 
                   'passing_yards', 'passing_tds', 'rushing_yards', 'rushing_tds',
                   'fantasy_points', 'fantasy_points_dk_calculated', 'scoring_diff']].to_string(index=False))
print()

# Large differences (3+ points)
large_diff = qb_weekly[abs(qb_weekly['scoring_diff']) > 3].head(5)
print("LARGE DIFFERENCES (3+ points):")
print(large_diff[['player_name', 'recent_team', 'week', 'season', 
                  'passing_yards', 'passing_tds', 'rushing_yards', 'rushing_tds',
                  'fantasy_points', 'fantasy_points_dk_calculated', 'scoring_diff']].to_string(index=False))
print()

# Check if standard scoring is close enough
close_enough = abs(qb_weekly['scoring_diff']) <= 2  # Within 2 points
print(f"Performances within 2 points: {close_enough.sum()} / {len(qb_weekly)} ({close_enough.mean()*100:.1f}%)")
print()

print("=== RECOMMENDATION ===")
if correlation > 0.95:
    print("✅ Standard fantasy points are very highly correlated with DK points")
    print("   You could use standard fantasy_points as a proxy target")
elif correlation > 0.90:
    print("🟡 Standard fantasy points are highly correlated with DK points")
    print("   Consider using standard fantasy_points as a proxy target")
else:
    print("❌ Standard fantasy points are not well correlated with DK points")
    print("   You should calculate DK points manually")

print(f"\nCorrelation threshold: 0.95+ = excellent proxy, 0.90+ = good proxy, <0.90 = poor proxy")
