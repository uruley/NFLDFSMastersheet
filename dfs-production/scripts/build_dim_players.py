import re, uuid
from pathlib import Path
import pandas as pd
import nfl_data_py as nfl
from unidecode import unidecode

IN_ROSTERS = Path("data/facts/rosters/fct_rosters_weekly.parquet")
OUT_DIM    = Path("data/dim"); OUT_DIM.mkdir(parents=True, exist_ok=True)

# --- helpers ---
SUFFIXES = {"jr","sr","ii","iii","iv","v"}
def normalize_name(s: str) -> str:
    if not s: return ""
    s = unidecode(s).lower().replace("-", " ")
    s = re.sub(r"[.,'’`]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return " ".join(t for t in s.split() if t not in SUFFIXES)

def stable_uid(gsis_id: str | None, name_norm: str, birth: str | None) -> str:
    """Prefer GSIS; else name+birth; else name-only."""
    if pd.notna(gsis_id) and str(gsis_id):
        key = f"gsis:{gsis_id}"
    elif pd.notna(birth) and str(birth):
        key = f"namebirth:{name_norm}|{birth}"
    else:
        key = f"name:{name_norm}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))

def first_non_null(series: pd.Series):
    for v in series:
        if pd.notna(v) and v != "":
            return v
    return pd.NA

# --- load weekly rosters you built in Step 1 ---
df = pd.read_parquet(IN_ROSTERS)

# basic columns we need
need_cols = {"season","week","team_dk","team_raw","position","player_name",
             "first_name","last_name","birth_date",
             "gsis_id","espn_id","pfr_id","nfl_id","sportradar_id","yahoo_id",
             "rotowire_id","pff_id","fantasy_data_id","sleeper_id",
             "entry_year","rookie_year","years_exp"}
for c in need_cols:
    if c not in df.columns:
        df[c] = pd.NA

# normalized name for registry
df["name_norm"] = df["player_name"].map(normalize_name)

# choose a canonical display name = most recent occurrence
df = df.sort_values(["season","week"], ascending=[True, True])

# build an entity key to aggregate: prefer GSIS when present, else name+birth
entity_key = df.apply(
    lambda r: f"gsis:{r['gsis_id']}" if pd.notna(r["gsis_id"]) and str(r["gsis_id"]) else
              (f"namebirth:{r['name_norm']}|{r['birth_date']}" if pd.notna(r["birth_date"]) and str(r["birth_date"])
               else f"name:{r['name_norm']}"),
    axis=1
)
df["entity_key"] = entity_key

# aggregate to one row per entity_key
agg_ordered = df.sort_values(["season","week"], ascending=[False, False])  # prefer latest values
grouped = agg_ordered.groupby("entity_key")

players = grouped.agg({
    "player_name": first_non_null,
    "name_norm": "first",
    "birth_date": first_non_null,
    "first_name": first_non_null,
    "last_name": first_non_null,
    "gsis_id": first_non_null,
    "espn_id": first_non_null,
    "pfr_id": first_non_null,
    "nfl_id": first_non_null,
    "sportradar_id": first_non_null,
    "yahoo_id": first_non_null,
    "rotowire_id": first_non_null,
    "pff_id": first_non_null,
    "fantasy_data_id": first_non_null,
    "sleeper_id": first_non_null,
    "entry_year": first_non_null,
    "rookie_year": first_non_null,
    "years_exp": first_non_null,
}).reset_index(drop=True)

# --- aggregate to one row per entity_key (KEEP entity_key as a column) ---
agg_latest = df.sort_values(["season","week"], ascending=[False, False])  # prefer latest
grouped = agg_latest.groupby("entity_key")

players = grouped.agg({
    "player_name": first_non_null,
    "name_norm": "first",
    "birth_date": first_non_null,
    "first_name": first_non_null,
    "last_name": first_non_null,
    "gsis_id": first_non_null,
    "espn_id": first_non_null,
    "pfr_id": first_non_null,
    "nfl_id": first_non_null,
    "sportradar_id": first_non_null,
    "yahoo_id": first_non_null,
    "rotowire_id": first_non_null,
    "pff_id": first_non_null,
    "fantasy_data_id": first_non_null,
    "sleeper_id": first_non_null,
    "entry_year": first_non_null,
    "rookie_year": first_non_null,
    "years_exp": first_non_null,
}).reset_index()  # <— keep entity_key

# --- choose a primary position: prefer non-DST mode; fallback to latest ---
pos_non_dst = (
    agg_latest[agg_latest["position"] != "DST"]
    .groupby("entity_key")["position"]
    .agg(lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0])
)
pos_latest = grouped["position"].first()
pos_series = pos_non_dst.combine_first(pos_latest)

pos_df = pos_series.rename("position").reset_index()
players = players.merge(pos_df, on="entity_key", how="left")  # <— merge on same dtype

# compute player_uid
players["player_uid"] = players.apply(lambda r: stable_uid(r["gsis_id"], r["name_norm"], r["birth_date"]), axis=1)

# --- reorder columns ---
cols = [
    "player_uid","player_name","name_norm","first_name","last_name","birth_date","position",
    "gsis_id","espn_id","pfr_id","nfl_id","sportradar_id","yahoo_id","rotowire_id","pff_id",
    "fantasy_data_id","sleeper_id","entry_year","rookie_year","years_exp","entity_key"
]
players = players[cols].drop_duplicates()

# ---- attach additional IDs from nflverse crosswalk (fill blanks) ----
ids = nfl.import_ids()

# pick a name column that actually exists in your version
name_col = None
for c in ("player_name", "full_name", "display_name", "name"):
    if c in ids.columns:
        name_col = c
        break
if name_col is None:
    raise RuntimeError(f"No name column found in ids: {list(ids.columns)}")

ids["name_norm"] = ids[name_col].astype(str).map(normalize_name)

# only keep ID columns that exist to avoid KeyErrors
id_cols = [c for c in ["pfr_id", "espn_id", "nfl_id"] if c in ids.columns]

# 1) merge by GSIS first (strongest key)
join1_cols = ["gsis_id"] + id_cols
players = players.merge(ids[join1_cols].drop_duplicates(), on="gsis_id", how="left", suffixes=("", "_x"))
for col in id_cols:
    if f"{col}_x" in players.columns:
        players[col] = players[col].fillna(players[f"{col}_x"])
        players.drop(columns=[f"{col}_x"], inplace=True)

# 2) fill remaining gaps by normalized name (best-effort)
join2_cols = ["name_norm"] + id_cols
players = players.merge(ids[join2_cols].drop_duplicates(), on="name_norm", how="left", suffixes=("", "_byname"))
for col in id_cols:
    if f"{col}_byname" in players.columns:
        players[col] = players[col].fillna(players[f"{col}_byname"])
        players.drop(columns=[f"{col}_byname"], inplace=True)

# --- ensure all ID columns are string type for parquet compatibility ---
id_columns = ["gsis_id", "espn_id", "pfr_id", "nfl_id", "sportradar_id", "yahoo_id", 
              "rotowire_id", "pff_id", "fantasy_data_id", "sleeper_id"]
for col in id_columns:
    if col in players.columns:
        # Convert to string, replacing NaN with empty string
        players[col] = players[col].astype(str).replace('nan', '').replace('None', '')

# include DSTs as separate entities (one per team)
dsts = df[df["position"]=="DST"][["team_dk"]].dropna().drop_duplicates()
if not dsts.empty:
    dst_rows = []
    for team in sorted(dsts["team_dk"].unique()):
        nm = f"{team} DST"
        nm_norm = f"{team.lower()} dst"
        puid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"dst:{team}"))
        dst_rows.append({
            "player_uid": puid, "player_name": nm, "name_norm": nm_norm, "first_name": team,
            "last_name":"DST","birth_date": pd.NA, "position":"DST",
            "gsis_id": pd.NA,"espn_id": pd.NA,"pfr_id": pd.NA,"nfl_id": pd.NA,
            "sportradar_id": pd.NA,"yahoo_id": pd.NA,"rotowire_id": pd.NA,"pff_id": pd.NA,
            "fantasy_data_id": pd.NA,"sleeper_id": pd.NA,"entry_year": pd.NA,"rookie_year": pd.NA,"years_exp": pd.NA
        })
    players = pd.concat([players, pd.DataFrame(dst_rows)], ignore_index=True)

# write dim_players
players = players.sort_values(["position","player_name"])
players.to_parquet(OUT_DIM / "dim_players.parquet", index=False)
players.to_csv(OUT_DIM / "dim_players.csv", index=False)
print(f"✅ dim_players rows: {len(players):,}  → {OUT_DIM/'dim_players.parquet'}")

# -------- dim_player_season (season-level attributes) --------
# primary team per season = most frequent team_dk that year, plus first/last seen week
wk = df.copy()
wk["player_uid"] = wk.apply(lambda r: stable_uid(r["gsis_id"], normalize_name(r["player_name"]), r["birth_date"]), axis=1)
g = wk.groupby(["player_uid","season"])

def mode_team(s: pd.Series):
    m = s.mode()
    return m.iloc[0] if not m.empty else first_non_null(s)

dim_ps = pd.DataFrame({
    "primary_team_dk": g["team_dk"].apply(mode_team),
    "first_week": g["week"].min(),
    "last_week": g["week"].max(),
    "any_dst": g["position"].apply(lambda s: (s=="DST").any()),
    "any_k": g["position"].apply(lambda s: (s=="K").any()),
}).reset_index()

# rookie flag if rookie_year == season (from any weekly row we saw)
rookie_info = wk.groupby(["player_uid","season"])["rookie_year"].agg(first_non_null).reset_index()
dim_ps = dim_ps.merge(rookie_info, on=["player_uid","season"], how="left")
dim_ps["rookie_flag"] = dim_ps.apply(lambda r: bool(pd.notna(r["rookie_year"]) and r["rookie_year"]==r["season"]), axis=1)
dim_ps.drop(columns=["rookie_year"], inplace=True)

dim_ps.to_parquet(OUT_DIM / "dim_player_season.parquet", index=False)
dim_ps.to_csv(OUT_DIM / "dim_player_season.csv", index=False)
print(f"✅ dim_player_season rows: {len(dim_ps):,}  → {OUT_DIM/'dim_player_season.parquet'}")
