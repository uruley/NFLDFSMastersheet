# NFL DFS Production System

## Identity System Testing

This system solves the player identity problem once and for all by creating a master registry and mapping DraftKings players to it.

### 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Test with sample data:**
   ```bash
   python test_identity.py
   ```

3. **Test with real DraftKings data:**
   ```bash
   python test_identity.py --dk-csv path/to/your/dk_file.csv
   ```

### 📊 What Gets Tested

#### Test 1: Player Registry Build
- Fetches rosters from 2021-2025 using `nfl-data-py`
- Creates deterministic UUIDs for each player
- Validates data quality and completeness
- Saves to `data/registry/dim_players.parquet`

#### Test 2: DraftKings Mapping
- Loads DK player data (real or sample)
- Maps DK players to registry using multiple strategies:
  - **Exact match**: Name + Position + Team (100% confidence)
  - **Fuzzy match**: Similar names with position + team (85%+ confidence)
  - **Position-only match**: Name similarity with position only (75%+ confidence)
- Calculates match success rate
- **Target: 95%+ match rate**

### 🔍 Test Results

The system generates comprehensive test results:

- **Console output**: Real-time progress and results
- **Log file**: `identity_test.log` with detailed information
- **Test results**: `data/staging/current_week/identity_test_results.json`
- **Unmatched players**: `data/staging/current_week/unmatched_dk_players.csv` (for manual review)

### 📁 File Structure

```
src/identity/
├── build_registry.py      # Builds master player registry
├── dk_mapper.py          # Maps DK players to registry
├── test_identity_system.py # Comprehensive testing framework
└── __init__.py

data/
├── registry/
│   ├── dim_players.parquet    # Master player registry
│   └── map_dk.csv            # DK player mappings
└── staging/current_week/     # Test results and logs
```

### 🎯 Success Criteria

- ✅ Registry builds successfully with 1000+ players
- ✅ DK mapping achieves 95%+ match rate
- ✅ All critical data quality checks pass
- ✅ Unmatched players documented for manual review

### 🔧 Customization

#### Using Your Own DK Data
1. Ensure your CSV has these columns (or similar):
   - Player name
   - Position
   - Team
   - Player ID (optional, will be auto-generated)

2. Run the test:
   ```bash
   python test_identity.py --dk-csv your_file.csv
   ```

#### Adjusting Match Thresholds
Edit `src/identity/dk_mapper.py`:
- `fuzzy_match()`: Change `threshold=0.85` for stricter/flexible matching
- `position_team_match()`: Change `threshold=0.75` for position-only matches

### 📈 Performance

- **Registry build**: Typically 30-60 seconds for 5 seasons
- **DK mapping**: 1-5 seconds for 1000+ players
- **Memory usage**: ~100-200MB for full registry

### 🐛 Troubleshooting

#### Common Issues

1. **Import errors**: Make sure you're in the project root directory
2. **Missing packages**: Run `pip install -r requirements.txt`
3. **Low match rate**: Check unmatched players CSV for patterns
4. **Registry build fails**: Verify internet connection for NFL data

#### Getting Help

- Check `identity_test.log` for detailed error messages
- Review `unmatched_dk_players.csv` for mapping failures
- Verify your DK data format matches expected columns

### 🔄 Continuous Testing

For production use, consider:
- Running tests before each data update
- Monitoring match rate trends over time
- Automating tests in your CI/CD pipeline
- Setting up alerts for match rate drops

---

**Next Steps**: Once testing passes, integrate the identity system into your data pipeline for automatic player matching!

