# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Bot or Not Challenge** — competition to detect bot accounts in social media datasets. Custom scoring: `4*TP - 1*FN - 2*FP` (precision matters 2.5x more than recall).

**Eval:** 1-hour window on Feb 14 to run detector on unseen data and submit.

## Current Model Performance

**100% accuracy on all 4 practice datasets** (736/736 competition score). Cross-dataset validation (the realistic generalization test):
- EN: 95.2-97.1% accuracy across unseen datasets
- FR: 98.3-100% accuracy across unseen datasets

## Data Format

- **`dataset.posts&users.{id}.json`** — JSON with `id`, `lang`, `metadata`, `users[]`, `posts[]`. Users have `id`, `name`, `username`, `description`, `location`, `tweet_count`, `z_score`. Posts have `author_id`, `text`, `created_at`.
- **`dataset.bots.{id}.txt`** — One bot user UUID per line.
- EN datasets: 30, 32 (~24% bots). FR datasets: 31, 33 (~16% bots).

## Architecture

`detect_bots.py` implements a two-phase pipeline: hard rules (near-zero FP heuristics) → 3-model ML ensemble.

1. **Hard rules** fire first: garbled names (non-printable chars), LLM output leakage ("here are my recent tweets"), systematic URL typos (`htts://`), weird location formats
2. **Feature engineering** (~100 features): profile metadata, aggregated text stats, temporal patterns, text diversity/stylometry, hashtag analysis, cross-user TF-IDF similarity, AI-text detection markers, interaction features
3. **3-model ensemble:** LightGBM (35%) + XGBoost (35%) + GradientBoosting (30%)
4. **Language-specific models** — separate EN/FR trained on respective dataset pairs
5. **Cross-validated threshold optimization** — 5-fold stratified CV maximizing the custom score
6. Final detections = union of hard rule hits + ML predictions above threshold

### Key internal structure (`detect_bots.py`)
- `extract_text_features()` → per-tweet text signals
- `compute_temporal_features()` → posting pattern analysis
- `compute_text_diversity_features()` → stylometry across a user's posts
- `compute_hashtag_features()` → hashtag usage patterns
- `extract_user_features()` → orchestrates all per-user feature extraction
- `compute_cross_user_features()` → TF-IDF similarity between users
- `prepare_dataset()` → full pipeline: loads data → extracts features → returns DataFrame
- `BotDetectorEnsemble` class → trains models, optimizes threshold, predicts
- `apply_hard_rules()` → pre-ML heuristic bot detection
- `detect_bots()` → inference entry point (loads model, applies hard rules + ML)

## Commands

```bash
# Train models on all practice datasets
python detect_bots.py --train          # → saves models.pkl

# Run inference on new data
python detect_bots.py <input.json> <output.txt>

# EDA (stdlib only)
python eda_analysis.py

# Feature analysis
python feature_ablation.py     # Cross-dataset ablation study
python feature_selection.py    # Feature subset evaluation
python feature_selection_fast.py  # Faster variant
```

## Dependencies

`lightgbm`, `xgboost`, `scikit-learn`, `pandas`, `numpy`

## Key Bot Signals (by Cohen's d effect size)

| Feature | d | Notes |
|---------|---|-------|
| `hashtag_diversity` | 1.0-2.2 | Bots use many unique hashtags |
| `hour_entropy` | 1.0-1.3 | Bots post uniformly across hours |
| `night_ratio` | 1.0+ (FR) | FR bots post more at night |
| `inter_tweet_cv` | 0.7-0.8 | Bots have regular posting intervals |
| `url_per_post` | 0.6-0.8 | Bots use FEWER URLs (counter-intuitive) |
| `ai_phrase_ratio` | new | AI-generated text markers |
| `formality_score` | new | Bot text is more polished/formal |
| `artificial_caps_ratio` | new | Bots inject random mid-word CAPS |

## Bot Subtypes Identified

1. **"Typo injection" bots** — random CAPS mid-word (e.g., "aM I", "THe")
2. **"Persona" bots** — AI characters (Sir Reginald, CEO Samantha)
3. **"Generic life" bots** — AI opinions on generic topics
4. **"Flower/plant" bots** — very short, low-effort generic posts
5. **"Engagement bait" bots** — DM/follow/bio prompts

## Key Files

- `detect_bots.py` — Production detector (train + inference), saves `models.pkl`
- `eda_analysis.py` — Exploratory data analysis
- `feature_ablation.py` — Cross-dataset feature ablation study (imports from `detect_bots`)
- `BOT_DETECTION_PLAN.md` — Original detection plan
