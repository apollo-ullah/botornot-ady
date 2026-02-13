# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Bot or Not Challenge** project — a competition to detect bot accounts in social media datasets. The goal is to build a classifier that identifies bot users from their profile metadata and post content, optimized for a custom scoring function where precision matters 2.5x more than recall.

**Scoring:** `4×TP - 1×FN - 2×FP` — false positives are very costly. Optimal threshold is likely 0.6–0.85 (not default 0.5).

**Eval constraint:** 1-hour window on Feb 14 to run the detector on unseen data and submit results.

## Data Format

- **`dataset.posts&users.{id}.json`** — JSON with top-level keys: `id`, `lang`, `metadata`, `users[]`, `posts[]`. Each user has `id`, `name`, `description`, `location`, `tweet_count`, `z_score`. Each post has `author_id`, `text`, `created_at`.
- **`dataset.bots.{id}.txt`** — One bot user UUID per line (ground truth labels).
- Datasets 30/32 are English (~24% bots); datasets 31/33 are French (~16% bots).
- Submission format: one user ID per line, matching the `.bots.txt` format.

## Architecture

The plan (in `BOT_DETECTION_PLAN.md`) calls for:

1. **Language-specific models** — separate EN and FR classifiers (different bot profiles per language)
2. **Feature engineering pipeline** — profile features, post-aggregated text features, temporal features, optional sentence-transformer embeddings
3. **LightGBM primary model** with threshold tuning to maximize the custom score
4. **Two-stage ensemble** (LightGBM + Logistic Regression) for robustness

Training splits: EN uses datasets 30+32, FR uses datasets 31+33.

## Key Files

- `eda_analysis.py` — EDA script comparing bot vs human feature distributions across all datasets. Run with `python eda_analysis.py` (requires only stdlib).
- `BOT_DETECTION_PLAN.md` — Full implementation plan with feature specs, model choices, and checklist.
- `bot_or_not_challenge.pdf` — Original challenge specification.

## Commands

```bash
# Run EDA analysis (no dependencies beyond stdlib)
python eda_analysis.py

# Final detector (to be built per plan)
python detect_bots.py <input.json> <output.txt>
```

## Dependencies

Core: `lightgbm`, `scikit-learn`, `pandas`, `numpy`. Optional: `sentence-transformers` (for text embeddings). See `BOT_DETECTION_PLAN.md` Appendix C for versions.

## Important Patterns

- **Engagement bait regex** — key bot signal: DM prompts, "check my bio", "follow", "opt in" (see Appendix A in plan)
- **Profile structure** — pipe `|` separators, hashtags in bio, formulaic descriptions indicate bots
- **Strong features** (consistent across all datasets): `tweet_count`, `z_score`, `hashtag_count_mean`, `char_len_mean`, `has_dm_prompt`
- **Counter-intuitive:** bots use *fewer* URLs than humans
