import re, sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import nfl_data_py as nfl
from unidecode import unidecode

# -------- settings --------
OUT_DIR = Path("data/facts/rosters")
YEARS = list(range(2020, 2026))  # 2020–2025 inclusive
ADD_DST = True                   # add synthetic DST rows per team-week
# --------------------------

# DK canonical mapping for team codes
ALT2DK = {
    "KCC":"KC","KAN":"KC","SFO":"SF","STL":"LAR","SD":"LAC","SDG":"LAC","OAK":"LV",
    "WSH":"WAS","WDC":"WAS","GNB":"GB","TAM":"TB","TBB":"TB","NOR":"NO","NWE":"NE",
    "CLV":"CLE","JAC":"JAX","ARZ":"ARI","PHL":"PHI","LA":"LAR"
}
def to_dk(code: str) -> str:
    c = (str(code) or "").upper()
    return ALT2DK.get(c, c)

SUFFIXES = {"jr","sr","ii","iii","iv","v"}
def name_norm(s: str) -> str:
    if not s: return ""
    s = unidecode(s).lower().replace("-", " ")
    s = re.sub(r"[.,'’`]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return " ".join(t for t in s.split() if t not in SUFFIXES)

TARGET_COLS = [
    "season","week","game_type",
    "team","position","depth_chart_position","jersey_number","status",
    "player_name","first_name","last_name","birth_date","height","weight","college",
    "player_id","espn_id","sportradar_id","yahoo_id","rotowire_id","pff_id","pfr_id",
    "fantasy_data_id","sleeper_id","years_exp","headshot_url","ngs_position",
    "status_description_abbr","football_name","esb_id","gsis_it_id","smart_id",
    "entry_year","rookie_year","draft_club","draft_number","age"
]

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"→ Pulling weekly rosters for years: {YEARS}")

    # 1) Pull weekly rosters 2020–2025
    df = nfl.import_weekly_rosters(YEARS)

    # 2) Keep/Pad columns to a consistent schema
    for col in TARGET_COLS:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[TARGET_COLS].copy()

    # 3) Rename player_id -> gsis_id (clearer)
    df = df.rename(columns={"player_id":"gsis_id"})

    # 4) Canonicalize teams/positions + normalize names
    df["team_raw"] = df["team"]
    df["team_dk"] = df["team"].map(to_dk)
    df["position"] = df["position"].replace({"FB":"RB"})  # minor cleanup
    df["name_norm"] = df["player_name"].map(name_norm)

    # 5) Optional: add DST rows per team *per season-week*
    if ADD_DST:
        # unique (season, week, team_dk) combos present in data
        uniq = df[["season","week","team_dk"]].dropna().drop_duplicates()
        dst_rows = []
        for _, row in uniq.iterrows():
            season, week, team = int(row["season"]), int(row["week"]), row["team_dk"]
            if not team or team == "": continue
            dst_rows.append({
                "season": season, "week": week, "game_type": None,
                "team": team, "team_raw": team, "team_dk": team,
                "position": "DST", "depth_chart_position": "DST",
                "jersey_number": None, "status": "ACT",
                "player_name": f"{team} DST", "first_name": team, "last_name": "DST",
                "birth_date": None, "height": None, "weight": None, "college": "NFL Team",
                "gsis_id": None, "espn_id": None, "sportradar_id": None, "yahoo_id": None,
                "rotowire_id": None, "pff_id": None, "pfr_id": None, "fantasy_data_id": None,
                "sleeper_id": None, "years_exp": None, "headshot_url": None, "ngs_position": "DST",
                "status_description_abbr": "ACT", "football_name": f"{team} DST",
                "esb_id": None, "gsis_it_id": None, "smart_id": None,
                "entry_year": season, "rookie_year": None, "draft_club": None, "draft_number": None, "age": None,
                "name_norm": f"{team.lower()} dst"
            })
        if dst_rows:
            dst_df = pd.DataFrame(dst_rows)
            # align columns
            for col in df.columns:
                if col not in dst_df.columns: dst_df[col] = pd.NA
            # put DST at same column order
            dst_df = dst_df[df.columns]
            df = pd.concat([df, dst_df], ignore_index=True)

    # 6) Stamp ingestion time
    df["ingested_at"] = pd.Timestamp.utcnow()

    # 7) Sort & save
    df = df.sort_values(["season","week","team_dk","position","player_name"], na_position="last")

    # Single files
    out_parquet = OUT_DIR / "fct_rosters_weekly.parquet"
    out_csv     = OUT_DIR / "fct_rosters_weekly.csv"
    df.to_parquet(out_parquet, index=False)
    df.to_csv(out_csv, index=False)

    # Also split by season (optional, handy for debugging)
    for y in YEARS:
        sub = df[df["season"]==y]
        if not sub.empty:
            sub.to_parquet(OUT_DIR / f"fct_rosters_weekly_{y}.parquet", index=False)

    print(f"✅ wrote {out_parquet}  (rows={len(df):,}, cols={df.shape[1]})")
    print(f"✅ wrote {out_csv}")
    print("Tip: quick peek →",
          "python -c \"import pandas as pd; d=pd.read_parquet(r'{}'); print(d[['season','week','team_dk','position']].head())\"".format(out_parquet))
    return 0

if __name__ == "__main__":
    sys.exit(main())
