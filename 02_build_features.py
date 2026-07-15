"""
02_build_features.py
====================
Computes ELO ratings for every team at the time of every match,
then joins them to produce the training-ready dataset.

How ELO works here
------------------
Rather than using the pre-computed eloratings.net snapshots as static
features, we *recompute* ELO dynamically from the full match history.
This gives us the exact ELO of each team on the day of each match —
which is what the model needs.

ELO update formula (standard):
  E  = 1 / (1 + 10^((R_away - R_home) / 400))   # expected score for home
  R' = R + K * (S - E)                            # updated rating

where S = 1 (win), 0.5 (draw), 0 (loss), and K controls how fast
ratings change. We use a higher K for World Cup matches than friendlies.

Outputs
-------
data/matches_with_elo.csv     Training-ready: every match + elo features
data/elo_timeline.csv         ELO of every team after every match (audit trail)
data/team_elo_current.csv     Most recent ELO per team (used for 2026 prediction)
"""

import pandas as pd
import numpy as np
import os

os.makedirs("data", exist_ok=True)

# ─────────────────────────────────────────────
# ELO config
# ─────────────────────────────────────────────

INITIAL_ELO   = 1500          # starting rating for any new team
HOME_ADVANTAGE = 100          # added to home team's effective ELO (neutral=0)

# K-factor by tournament importance — bigger K = faster rating change
K_FACTORS = {
    "FIFA World Cup":               60,
    "FIFA World Cup qualification": 40,
    "UEFA Euro":                    50,
    "Copa América":                 50,
    "AFC Asian Cup":                40,
    "Africa Cup of Nations":        40,
    "Friendly":                     20,
}
K_DEFAULT = 30


# ─────────────────────────────────────────────
# Core ELO engine
# ─────────────────────────────────────────────

def expected_score(r_home: float, r_away: float, neutral: bool) -> float:
    """Expected score (0–1) for home team."""
    advantage = 0 if neutral else HOME_ADVANTAGE
    return 1.0 / (1.0 + 10 ** ((r_away - r_home - advantage) / 400))


def k_factor(tournament: str) -> float:
    return K_FACTORS.get(tournament, K_DEFAULT)


def compute_elo(matches: pd.DataFrame):
    """
    Walk through every match chronologically, maintaining a live ELO
    dict. Returns:
      - matches_with_elo: original df + pre-match ELO columns + elo_diff
      - elo_timeline: full history of every rating change
      - final_elo: dict of team → latest ELO
    """
    matches = matches.sort_values("date").reset_index(drop=True)

    elo: dict[str, float] = {}          # team → current ELO
    records   = []                      # one row per match (pre-match ELOs)
    timeline  = []                      # full audit trail

    for _, row in matches.iterrows():
        ht, at = row["home_team"], row["away_team"]
        neutral = bool(row.get("neutral", False))

        r_home = elo.get(ht, INITIAL_ELO)
        r_away = elo.get(at, INITIAL_ELO)

        E = expected_score(r_home, r_away, neutral)

        # Actual score from home team's perspective
        if row["outcome"] == 1:
            S = 1.0
        elif row["outcome"] == -1:
            S = 0.0
        else:
            S = 0.5

        K = k_factor(row["tournament"])
        delta = K * (S - E)

        # Record pre-match state
        records.append({
            "date":          row["date"],
            "year":          row["year"],
            "tournament":    row["tournament"],
            "match_weight":  row["match_weight"],
            "home_team":     ht,
            "away_team":     at,
            "home_goals":    row["home_goals"],
            "away_goals":    row["away_goals"],
            "outcome":       row["outcome"],
            "neutral":       neutral,
            "home_elo":      round(r_home, 1),
            "away_elo":      round(r_away, 1),
            "elo_diff":      round(r_home - r_away, 1),   # main feature
            "home_elo_adj":  round(r_home + (0 if neutral else HOME_ADVANTAGE), 1),
            "expected_home": round(E, 4),                 # model sanity check
            "K":             K,
            "elo_delta":     round(delta, 2),
        })

        # Update ratings
        elo[ht] = r_home + delta
        elo[at] = r_away - delta

        timeline.append({"date": row["date"], "team": ht, "elo": round(elo[ht], 1)})
        timeline.append({"date": row["date"], "team": at, "elo": round(elo[at], 1)})

    df_matches   = pd.DataFrame(records)
    df_timeline  = pd.DataFrame(timeline)
    return df_matches, df_timeline, elo


# ─────────────────────────────────────────────
# Post-processing helpers
# ─────────────────────────────────────────────

def add_form_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Add rolling recent-form ELO change for each team over last N matches.
    This captures momentum beyond the raw ELO level.
    """
    df = df.sort_values("date").copy()

    # Build a long-form series of elo_delta per team per match
    home_deltas = df[["date","home_team","elo_delta"]].rename(
        columns={"home_team":"team","elo_delta":"delta"})
    away_deltas = df[["date","away_team","elo_delta"]].rename(
        columns={"away_team":"team"})
    away_deltas["delta"] = -df["elo_delta"]   # away team gains the opposite delta

    all_deltas = pd.concat([home_deltas, away_deltas]).sort_values("date")

    # Rolling sum of deltas (momentum)
    all_deltas["form"] = all_deltas.groupby("team")["delta"].transform(
        lambda x: x.rolling(window, min_periods=1).sum()
    )
    form_lookup = all_deltas.set_index(["date","team"])["form"].to_dict()

    df["home_form"] = df.apply(
        lambda r: form_lookup.get((r["date"], r["home_team"]), 0), axis=1)
    df["away_form"] = df.apply(
        lambda r: form_lookup.get((r["date"], r["away_team"]), 0), axis=1)
    df["form_diff"] = df["home_form"] - df["away_form"]

    return df


def filter_wc_only(df: pd.DataFrame) -> pd.DataFrame:
    """Subset to just World Cup matches for evaluation."""
    return df[df["tournament"] == "FIFA World Cup"].copy()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Load
    matches_path = "data/matches.csv"
    if not os.path.exists(matches_path):
        raise FileNotFoundError("Run 01_collect_data.py first.")

    print("Loading matches...")
    matches = pd.read_csv(matches_path, parse_dates=["date"])
    print(f"  {len(matches):,} matches loaded")

    # Compute ELO across full history
    print("Computing ELO ratings across full history...")
    df_features, df_timeline, final_elo = compute_elo(matches)
    print(f"  Done. {len(df_features):,} matches processed.")

    # Add form features
    print("Adding rolling form features (window=5)...")
    df_features = add_form_features(df_features, window=5)

    # Save full dataset (all tournaments — for richer training)
    df_features.to_csv("data/matches_with_elo.csv", index=False)
    print(f"  → Saved data/matches_with_elo.csv  ({len(df_features):,} rows)")

    # Save WC-only subset for inspection
    df_wc = filter_wc_only(df_features)
    df_wc.to_csv("data/wc_matches_elo.csv", index=False)
    print(f"  → Saved data/wc_matches_elo.csv    ({len(df_wc):,} WC matches)")

    # Save ELO timeline (audit trail)
    df_timeline.to_csv("data/elo_timeline.csv", index=False)
    print(f"  → Saved data/elo_timeline.csv      ({len(df_timeline):,} snapshots)")

    # Save current ELO per team (used to build 2026 predictions)
    current_elo = (
        pd.DataFrame({"team": list(final_elo.keys()),
                      "elo":  list(final_elo.values())})
        .sort_values("elo", ascending=False)
        .reset_index(drop=True)
    )
    current_elo["rank"] = current_elo.index + 1
    current_elo.to_csv("data/team_elo_current.csv", index=False)
    print(f"  → Saved data/team_elo_current.csv  ({len(current_elo)} teams)")

    # Print top 20 current ELO
    print("\nTop 20 teams by current ELO:")
    print(current_elo.head(20).to_string(index=False))

    # Feature summary for the training set
    print(f"\nFeature columns in matches_with_elo.csv:")
    feature_cols = ["elo_diff", "home_elo_adj", "away_elo",
                    "form_diff", "neutral", "match_weight"]
    print("  Target  : outcome  (1=home win, 0=draw, -1=away win)")
    print("  Features:", ", ".join(feature_cols))
    print("\nNext: run  python 03_train_model.py")
