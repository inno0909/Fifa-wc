# World Cup 2026 — ELO Pipeline

## File Structure
```
wc_elo_pipeline/
├── requirements.txt
├── 01_collect_data.py      # Download matches + ELO + 2026 squads
├── 02_build_features.py    # Compute ELO per match, add form features
└── data/
    ├── matches.csv                 All intl results 1872–2026
    ├── matches_with_elo.csv        Training-ready: all matches + ELO features
    ├── wc_matches_elo.csv          WC-only subset for inspection
    ├── elo_timeline.csv            Full ELO history per team
    ├── team_elo_current.csv        Latest ELO per team (for 2026 predictions)
    └── wc2026_squads.csv           2026 team names (for name mapping)
```

## Setup
```bash
pip install -r requirements.txt
```

## Run
```bash
python 01_collect_data.py     # ~1 min  — downloads ~49k matches + ELO
python 02_build_features.py   # ~1 min  — computes ELO for every match
```

## No manual downloads required
Everything is fetched automatically from:
- martj42 GitHub (match results)
- eloratings.net (ELO snapshot) with fallback to martj42 ELO CSV
- Wikipedia (2026 squad list / team names)

## Feature schema: matches_with_elo.csv

| Column | Description |
|--------|-------------|
| date, year | Match date |
| tournament | Competition name |
| match_weight | 0.3 (friendly) → 1.0 (World Cup) |
| home_team, away_team | Team names |
| home_goals, away_goals | Full-time score |
| outcome | **Target** — 1=home win, 0=draw, -1=away win |
| home_elo | Home team ELO *before* this match |
| away_elo | Away team ELO *before* this match |
| elo_diff | home_elo − away_elo — **primary feature** |
| home_elo_adj | home_elo + 100 if not neutral (home advantage baked in) |
| expected_home | Model-implied win probability pre-match |
| form_diff | Rolling 5-match ELO delta home − away (**momentum**) |
| neutral | True if played at neutral venue |

## ELO parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Initial rating | 1500 | All new teams start here |
| Home advantage | +100 ELO | Ignored at neutral venues |
| K — World Cup | 60 | Fastest rating change |
| K — WC qualifying | 40 | |
| K — Continental | 40–50 | |
| K — Friendly | 20 | Slowest change |
| Form window | 5 matches | Rolling ELO delta sum |
