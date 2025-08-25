# DraftKings Sample Data

## How to Get a Real DraftKings CSV:

1. **Go to DraftKings.com**
2. **Enter any NFL contest**
3. **Click "Export to CSV"**
4. **Save as `dk_week10.csv` in this directory**

## Expected CSV Format:
The CSV should have columns like:
- `Name` (player name)
- `Position` (QB, RB, WR, TE, K, DST)
- `Team` (team abbreviation)

## File Structure:
```
data/samples/
├── dk_week10.csv          # Your real DK export (required)
└── README.md              # This file
```

## Testing:
Once you have the CSV file, run:
```bash
python src/identity/test_matching.py
```

**Success = 95%+ match rate**

