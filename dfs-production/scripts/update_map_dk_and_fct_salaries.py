import os, re, argparse, pandas as pd
from pathlib import Path
from unidecode import unidecode
from rapidfuzz import process, fuzz

DIM_PLAYERS = Path("data/dim/dim_players.parquet")
FCT_ROSTERS = Path("data/facts/rosters/fct_rosters_weekly.parquet")
MAP_DK      = Path("data/dim/map_dk.parquet")
FCT_SAL     = Path("data/facts/salaries/fct_salaries.parquet")
OUTDIR      = Path("output"); OUTDIR.mkdir(parents=True, exist_ok=True)

ALT2DK = {
  "KCC":"KC","KAN":"KC","SFO":"SF","STL":"LAR","SD":"LAC","SDG":"LAC","OAK":"LV",
  "WSH":"WAS","WDC":"WAS","GNB":"GB","TAM":"TB","TBB":"TB","NOR":"NO","NWE":"NE",
  "CLV":"CLE","JAC":"JAX","ARZ":"ARI","PHL":"PHI","LA":"LAR"
}
def to_dk(code:str)->str:
    c=(code or "").upper()
    return ALT2DK.get(c, c)

POS_EQUIV = {
    "RB": {"RB","FB"},
    "FB": {"RB","FB","TE"},
    "TE": {"TE","LS","FB"},
    "WR": {"WR","RB"},      # Velus Jones Jr., Brady Russell, etc. weird DK slots
    "QB": {"QB","WR"},      # John Rhys Plumlee type position switches
    "LS": {"LS","TE"},
    "DST":{"DST"}
}
def pos_equivalent(dk_pos, roster_pos):
    # Ensure both are strings and handle None/NaN values
    dk_pos_str = str(dk_pos) if pd.notna(dk_pos) else ""
    roster_pos_str = str(roster_pos) if pd.notna(roster_pos) else ""
    
    if not dk_pos_str or not roster_pos_str:
        return False
    
    s = POS_EQUIV.get(dk_pos_str.upper(), {dk_pos_str.upper()})
    return roster_pos_str.upper() in s

SUFFIXES = {"jr","sr","ii","iii","iv","v"}
def norm_name(s:str)->str:
    if not s: return ""
    s = unidecode(s).lower().replace("-", " ")
    s = re.sub(r"[.,'’`]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return " ".join(t for t in s.split() if t not in SUFFIXES)

def load_dk_csv(path:str)->pd.DataFrame:
    dk = pd.read_csv(path)
    name_col = "Name" if "Name" in dk.columns else ("name" if "name" in dk.columns else None)
    team_col = "TeamAbbrev" if "TeamAbbrev" in dk.columns else ("Team" if "Team" in dk.columns else "team")
    pos_col  = "Position" if "Position" in dk.columns else ("Pos" if "Pos" in dk.columns else "position")
    id_col   = "ID" if "ID" in dk.columns else ("Id" if "Id" in dk.columns else "dk_player_id")
    sal_col  = "Salary" if "Salary" in dk.columns else ("salary" if "salary" in dk.columns else None)
    if not all([name_col, team_col, pos_col, id_col]):
        raise SystemExit(f"DK CSV missing core columns. Got: {dk.columns.tolist()}")
    dk = dk.rename(columns={
        name_col:"dk_name", team_col:"team_raw", pos_col:"position",
        id_col:"dk_player_id"
    })
    if sal_col: dk = dk.rename(columns={sal_col:"salary"})
    dk["name_norm"] = dk["dk_name"].map(norm_name)
    dk["team_dk"]   = dk["team_raw"].map(to_dk)
    
    # strip whitespace junk in DK names
    dk["dk_name"] = dk["dk_name"].astype(str).str.strip()
    
    # make DST names canonical: "{team_dk} dst" (e.g., "den dst")
    mask_dst = dk["position"].str.upper().eq("DST")
    dk.loc[mask_dst, "name_norm"] = dk.loc[mask_dst, "team_dk"].str.lower() + " dst"
    
    return dk

def roster_latest_for_2025() -> pd.DataFrame:
    ro = pd.read_parquet(FCT_ROSTERS)
    ro25 = ro[ro["season"]==2025].copy()
    if ro25.empty:
        raise SystemExit("No 2025 rows in fct_rosters_weekly.parquet. Build Step 1 first.")
    # prefer latest week for team tie-breakers
    ro25 = ro25.sort_values(["season","week"]).dropna(subset=["name_norm"])
    # get a latest snapshot per (name_norm, position)
    snap = ro25.sort_values(["week"]).groupby(["name_norm","position"], as_index=False).last()
    # also keep player_uid if it exists later (we'll compute it from dim_players)
    snap = snap[["name_norm","position","team_dk"]].drop_duplicates()
    return snap

def dim_players_min() -> pd.DataFrame:
    dp = pd.read_parquet(DIM_PLAYERS)
    keep = ["player_uid","name_norm","position","gsis_id","espn_id","pfr_id","nfl_id"]
    for c in keep:
        if c not in dp.columns:
            dp[c] = pd.NA
    return dp[keep].drop_duplicates()

def load_map_dk() -> pd.DataFrame:
    if MAP_DK.exists():
        return pd.read_parquet(MAP_DK)
    return pd.DataFrame(columns=["dk_player_id","player_uid","first_seen","last_seen",
                                 "name_first_seen","team_first_seen","position_first_seen"])

def save_map_dk(df: pd.DataFrame):
    MAP_DK.parent.mkdir(parents=True, exist_ok=True)
    df.drop_duplicates(subset=["dk_player_id"], keep="last").to_parquet(MAP_DK, index=False)

def save_fct_salaries(rows: pd.DataFrame):
    FCT_SAL.parent.mkdir(parents=True, exist_ok=True)
    if FCT_SAL.exists():
        base = pd.read_parquet(FCT_SAL)
        rows = pd.concat([base, rows], ignore_index=True)
    rows.to_parquet(FCT_SAL, index=False)

def match_new_ids(dk_new: pd.DataFrame, roster25: pd.DataFrame, dimp: pd.DataFrame) -> pd.DataFrame:
    # Stage 1: exact name+team+position
    res = dk_new.merge(roster25, on=["name_norm","position"], how="left", suffixes=("","_r"))
    res = res.merge(dimp, on=["name_norm","position"], how="left", suffixes=("","_p"))

    # Stage 1b: exact name+team (ignore position) with pos-equivalence filter
    miss = res["player_uid"].isna()
    if miss.any():
        # candidates by name only
        cand = dimp.merge(roster25[["name_norm","team_dk"]].drop_duplicates(), on="name_norm", how="left")
        for i in res[miss].index:
            dk_row = res.loc[i]
            pool = cand[cand["name_norm"].eq(dk_row["name_norm"])]
            if pool.empty: continue
            # prefer same team; then accept pos-equivalent; else any
            same_team = pool[pool["team_dk"].eq(dk_row["team_dk"])]
            picked = None
            if not same_team.empty:
                # if any roster position for this name is compatible with DK position, take it
                compat = same_team[same_team["position"].apply(lambda rp: pos_equivalent(dk_row["position"], rp))]
                picked = compat.iloc[0] if not compat.empty else same_team.iloc[0]
            else:
                compat = pool[pool["position"].apply(lambda rp: pos_equivalent(dk_row["position"], rp))]
                picked = compat.iloc[0] if not compat.empty else pool.iloc[0]
            res.at[i, "player_uid"] = picked["player_uid"]

    # Stage 2: fuzzy within TEAM first (ignore position), then within POSITION
    miss = res["player_uid"].isna()
    if miss.any():
        from rapidfuzz import process, fuzz
        # prebuild dictionaries
        by_team = {
            t: dimp.merge(roster25[roster25["team_dk"].eq(t)][["name_norm"]].drop_duplicates(),
                          on="name_norm", how="inner")[["name_norm","player_uid","position"]]
            for t in roster25["team_dk"].dropna().unique()
        }
        for i in res[miss].index:
            dk_row = res.loc[i]
            # try team-constrained candidates first
            pool = by_team.get(dk_row["team_dk"], dimp[["name_norm","player_uid","position"]])
            hits = process.extract(dk_row["name_norm"], pool["name_norm"].tolist(), scorer=fuzz.WRatio, limit=3)
            hits = [h for h in hits if h[1] >= 93]
            if hits:
                # prefer a hit with pos-equivalence
                best_uid = None
                for nm, score, _ in hits:
                    cand = pool[pool["name_norm"].eq(nm)].iloc[0]
                    if pos_equivalent(dk_row["position"], cand["position"]):
                        best_uid = cand["player_uid"]; break
                if best_uid is None:
                    best_uid = pool[pool["name_norm"].eq(hits[0][0])]["player_uid"].iloc[0]
                res.at[i, "player_uid"] = best_uid

    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dk", required=True, help="Path to DraftKings Salary CSV")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--slate_id", default="main")
    args = ap.parse_args()

    dk = load_dk_csv(args.dk)
    
    # optional alias hook
    alias_path = Path("data/dim/aliases.csv")
    if alias_path.exists():
        alias = pd.read_csv(alias_path)
        alias["dk_team_dk"] = alias["dk_team_dk"].astype(str)
        alias["target_name_norm"] = alias["target_name_norm"].astype(str)
        dk = dk.merge(alias[["dk_player_id","target_name_norm"]], on="dk_player_id", how="left")
        dk["name_norm"] = dk["target_name_norm"].fillna(dk["name_norm"])
        dk.drop(columns=["target_name_norm"], inplace=True)
    
    dp = dim_players_min()
    roster25 = roster_latest_for_2025()
    mapdk = load_map_dk()

    # Attach existing mappings fast by dk_player_id
    dk = dk.merge(mapdk[["dk_player_id","player_uid"]], on="dk_player_id", how="left")

    # Resolve new dk ids
    new_mask = dk["player_uid"].isna()
    if new_mask.any():
        new_res = match_new_ids(dk[new_mask].copy(), roster25, dp)[["dk_player_id","player_uid","dk_name","team_dk","position","name_norm"]]
        # Merge back
        dk = dk.merge(new_res[["dk_player_id","player_uid"]], on="dk_player_id", how="left", suffixes=("","_new"))
        dk["player_uid"] = dk["player_uid"].fillna(dk["player_uid_new"]); dk.drop(columns=["player_uid_new"], inplace=True)

        # Append new rows to map_dk
        add = new_res.dropna(subset=["player_uid"]).copy()
        if not add.empty:
            add["first_seen"] = pd.Timestamp.utcnow().date()
            add["last_seen"]  = add["first_seen"]
            add["name_first_seen"] = add["dk_name"]
            add["team_first_seen"] = add["team_dk"]
            add["position_first_seen"] = add["position"]
            mapdk = pd.concat([mapdk, add[["dk_player_id","player_uid","first_seen","last_seen","name_first_seen","team_first_seen","position_first_seen"]]], ignore_index=True)
            save_map_dk(mapdk)

    # Coverage report
    hit = dk["player_uid"].notna()
    overall = round(hit.mean()*100, 1)
    by_pos = dk.groupby("position")["player_uid"].apply(lambda s: round(s.notna().mean()*100,1)).to_dict()

    # Save unmatched for review
    unmatched = dk.loc[~hit, ["dk_player_id","dk_name","team_dk","position","name_norm"]]
    unmatched_path = OUTDIR / "dk_unmatched.csv"
    unmatched.to_csv(unmatched_path, index=False)

    print(f"Players: {len(dk)} | Coverage: {overall}% | By pos: {by_pos}")
    print(f"Unmatched written → {unmatched_path} (rows={len(unmatched)})")

    # Append fct_salaries for this slate (only matched rows)
    sal_cols = ["dk_player_id","player_uid","team_dk","position","salary"] if "salary" in dk.columns else ["dk_player_id","player_uid","team_dk","position"]
    fct = dk[hit][sal_cols].copy()
    fct["season"] = args.season
    fct["week"]   = args.week
    fct["slate_id"] = args.slate_id
    fct["ingested_at"] = pd.Timestamp.utcnow()
    save_fct_salaries(fct)

    print(f"✅ map_dk saved to {MAP_DK}")
    print(f"✅ fct_salaries appended to {FCT_SAL}")

if __name__ == "__main__":
    main()
