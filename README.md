# DFS Master Sheet Pipeline

A robust system for merging DraftKings salary data with roster information using normalized matching keys.

## Pipeline Overview

This system takes DraftKings salaries and roster data as inputs, applies consistent cleaning rules, and produces matched datasets with high accuracy rates.

### Inputs

- **`data/raw/DKSalaries.csv`** - DraftKings salary data (Name, Position, TeamAbbrev, Salary, Game Info)
- **`data/raw/rosters_2025_season.csv`** - NFL roster data (player_id, player_name, team, position, etc.)

### Cleaning Rules

All matching uses normalized data with these transformations:

- **`clean_name`**: lowercase → remove punctuation → collapse spaces → drop suffix (jr, sr, ii, iii, iv, v)
- **`norm_team`**: map {JAX→JAC, WSH→WAS, OAK→LV, SD→LAC, LA→LAR, STL→LAR}
- **`norm_pos`**: D/ST or DEF → DST, else UPPERCASE
- **`join_key`**: `clean_name|norm_team|norm_pos` for players; `TEAM-<abbr>|<abbr>|DST` for defenses

### Data Sources

- **`data/xwalk/aliases.csv`** - Manual name mappings (dk_name → roster_full_name)
- **`data/xwalk/synthetic_dst_roster_rows.csv`** - Generated defense team entries
- **`data/xwalk/xwalk_manual.csv`** - Manual conflict resolution overrides

### Outputs

- **`data/processed/master_sheet.csv`** - All DK rows + matched player_id
- **`data/processed/crosswalk.csv`** - DK key columns + match method (exact/dst/unmatched)
- **`data/processed/unmatched.csv`** - Unmatched DK rows sorted by salary
- **`data/processed/match_report.txt`** - Summary statistics and validation log

## Usage

```bash
python scripts/build_master_sheet.py \
  --dk data/raw/DKSalaries.csv \
  --rosters data/raw/rosters_2025_season.csv \
  --master-out data/processed/master_sheet.csv \
  --crosswalk-out data/processed/crosswalk.csv \
  --unmatched-out data/processed/unmatched.csv
```

## Expected Performance

- **First run**: ~90-94% matched
- **After DST rows + aliases**: 98-100% matched  
- **Each new slate**: Usually only 1-5 new aliases needed

## Project Rules

**Repository Structure:**
```
data/raw/           (input CSVs)
data/processed/     (outputs)
data/xwalk/         (aliases + dst + manual tie-breaks)
scripts/            (python scripts)
```

**Allowed Files Only:**
- `data/xwalk/aliases.csv`
- `data/xwalk/synthetic_dst_roster_rows.csv`
- `data/xwalk/xwalk_manual.csv`
- `scripts/build_master_sheet.py`

Never rename or move files. Never add new folders.

## Match Methods

- **`exact`** - Direct join_key match between DK and roster
- **`dst`** - Defense team matched via synthetic roster entry
- **`unmatched`** - No match found (candidates for aliases)

## Master Sheet Pipeline

### Inputs
- **`data/raw/DKSalaries.csv`** - DraftKings salary data with columns: Name, Position, TeamAbbrev, Salary, Game Info
- **`data/raw/rosters_2025_season.csv`** - NFL roster data with columns: player_id, player_name, team, position

### Memory Files
- **`data/xwalk/aliases.csv`** - Name mappings from DK to roster (dk_name → roster_full_name)
- **`data/xwalk/synthetic_dst_roster_rows.csv`** - Generated defense team entries for DST matching
- **`data/xwalk/roster_additions.csv`** - Missing players added with stable EXT- IDs
- **`data/xwalk/xwalk_manual.csv`** - Manual conflict resolution overrides for join_key conflicts

### Outputs
- **`data/processed/master_sheet.csv`** - All DK rows + matched player_id + method
- **`data/processed/crosswalk.csv`** - DK key columns + match method (exact/dst/fallback_name_team/unmatched)
- **`data/processed/unmatched.csv`** - Unmatched DK rows sorted by salary descending

### Matching Rules
1. **Primary matching**: `join_key` = `clean_name|norm_team|norm_pos`
2. **DST handling**: Special `TEAM-<abbr>|<abbr>|DST` format for defense teams
3. **Alias replacement**: Apply aliases to DK Name before cleaning/normalization
4. **Specialist fallback**: For unmatched rows, try name+team matching on LS/K/FB positions (exact candidate match only)
5. **Roster additions**: Union synthetic DST rows and missing player entries before matching
