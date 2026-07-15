"""
01_collect_data.py
==================
Downloads everything needed for the ELO-based World Cup model.

Outputs
-------
data/matches.csv          All international results 1872–2026 (martj42)
data/elo_ratings.csv      Historical team ELO ratings (eloratings.net)
data/wc2026_squads.csv    2026 World Cup squads (Wikipedia) — for team name mapping
"""

import requests
import pandas as pd
import io
import os
from bs4 import BeautifulSoup

os.makedirs("data", exist_ok=True)

# ─────────────────────────────────────────────
# 1. Match results — martj42 (GitHub)
# ─────────────────────────────────────────────

def download_match_results() -> pd.DataFrame:
    print("Downloading match results (martj42)...")
    url = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
    df = pd.read_csv(url)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year

    # Clean outcome label
    df["home_goals"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_goals"] = pd.to_numeric(df["away_score"], errors="coerce")
    df = df.dropna(subset=["home_goals", "away_goals"])
    df["outcome"] = df.apply(
        lambda r: 1 if r.home_goals > r.away_goals
        else (-1 if r.home_goals < r.away_goals else 0),
        axis=1,
    )

    # Tournament weight — heavier for competitive matches
    WEIGHTS = {
        "FIFA World Cup":              1.00,
        "FIFA World Cup qualification": 0.70,
        "UEFA Euro":                   0.85,
        "Copa América":                0.85,
        "AFC Asian Cup":               0.75,
        "Africa Cup of Nations":       0.75,
        "Friendly":                    0.30,
    }
    df["match_weight"] = df["tournament"].map(WEIGHTS).fillna(0.50)

    df = df[["date", "year", "tournament", "match_weight",
             "home_team", "away_team", "home_goals", "away_goals",
             "outcome", "neutral"]].copy()

    print(f"  → {len(df):,} matches  |  "
          f"{df['tournament'].nunique()} tournaments  |  "
          f"years {df['year'].min()}–{df['year'].max()}")
    df.to_csv("data/matches.csv", index=False)
    return df


# ─────────────────────────────────────────────
# 2. ELO ratings — eloratings.net
# ─────────────────────────────────────────────

def download_elo_ratings() -> pd.DataFrame:
    """
    eloratings.net publishes a full history CSV.
    Columns: date, team, elo_rating (rounded to nearest int)
    """
    print("Downloading ELO ratings (eloratings.net)...")
    url = "https://eloratings.net/World.tsv"
    headers = {"User-Agent": "Mozilla/5.0 (WC ML Research)"}

    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        # TSV format: rank, team, elo, +/-, matches
        df = pd.read_csv(io.StringIO(resp.text), sep="\t",
                         names=["rank", "team", "elo", "change", "matches"])
        df = df[["team", "elo"]].copy()
        df["elo"] = pd.to_numeric(df["elo"], errors="coerce")
        df = df.dropna(subset=["elo"])
        print(f"  → Current ELO snapshot: {len(df)} teams")
        df.to_csv("data/elo_current.csv", index=False)
        return df

    except Exception as e:
        print(f"  ✗ Live fetch failed ({e}), trying historical CSV...")

    # Fallback: full history from the GitHub mirror maintained by the community
    url2 = "https://raw.githubusercontent.com/martj42/international_results/master/elo.csv"
    try:
        df = pd.read_csv(url2)
        df["date"] = pd.to_datetime(df["date"])
        print(f"  → Historical ELO: {len(df):,} snapshots")
        df.to_csv("data/elo_ratings.csv", index=False)
        return df
    except Exception as e2:
        print(f"  ✗ Fallback also failed ({e2})")
        print("  → Manual download: https://eloratings.net  (click 'Download')")
        return pd.DataFrame()


# ─────────────────────────────────────────────
# 3. 2026 World Cup squads — Wikipedia
#    Used only for canonical team name mapping,
#    not as model features in the ELO pipeline.
# ─────────────────────────────────────────────

def download_2026_squads() -> pd.DataFrame:
    print("Scraping 2026 squad list (Wikipedia)...")
    url = "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_squads"
    headers = {"User-Agent": "Mozilla/5.0 (WC ML Research)"}
    resp = requests.get(url, headers=headers, timeout=30)
    soup = BeautifulSoup(resp.text, "html.parser")

    teams = []
    for h3 in soup.find_all("h3"):
        span = h3.find("span", {"class": "mw-headline"})
        if span:
            teams.append(span.get_text(strip=True))

    # Deduplicate and remove non-team headings
    seen = set()
    clean_teams = []
    for t in teams:
        if t not in seen and len(t) > 2:
            seen.add(t)
            clean_teams.append(t)

    df = pd.DataFrame({"team_wiki": clean_teams})
    print(f"  → {len(df)} teams found")
    df.to_csv("data/wc2026_squads.csv", index=False)
    return df


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("World Cup 2026 — ELO Data Collection")
    print("=" * 50)

    matches = download_match_results()
    elo     = download_elo_ratings()
    squads  = download_2026_squads()

    print("\nDone. Files written to data/")
    print("  data/matches.csv      ", f"{len(matches):>7,} rows")
    if not elo.empty:
        print("  data/elo_ratings.csv  ", f"{len(elo):>7,} rows")
    print("  data/wc2026_squads.csv", f"{len(squads):>7,} rows")
    print("\nNext: run  python 02_build_features.py")
