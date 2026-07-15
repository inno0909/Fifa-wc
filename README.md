# FIFA World Cup 2026 — Match Outcome Prediction

This notebook builds a machine learning model to predict match outcomes for the 2026 FIFA World Cup, hosted across the United States, Canada, and Mexico. The model uses ELO ratings as its core signal trained on over 150 years of international football results.

## Project Overview
The goal is to produce win/draw/loss probabilities for any matchup between the 48 qualified teams. These probabilities can then be used to simulate the full tournament bracket and estimate each team's likelihood of advancing through each round.

The approach is deliberately kept simple for this baseline version: a strong, well-calibrated ELO-based feature set fed into a multiclass classifier. Squad-level features (player ratings, xG, age profiles) are planned as a future enhancement once the baseline is validated.
