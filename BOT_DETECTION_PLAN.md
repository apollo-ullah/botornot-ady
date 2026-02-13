# Bot or Not Challenge: Final Model Plan

**Author:** Senior Research Scientist Analysis  
**Date:** February 12, 2026  
**Stakes:** High—false positives damage trust; false negatives enable manipulation. **Precision prioritized over recall.**

---

## Executive Summary

This document synthesizes: (1) contest rules and scoring, (2) exploratory data analysis across all four practice datasets, (3) state-of-the-art bot detection research, and (4) a concrete implementation plan for the final model. The scoring function (+4 TP, -1 FN, -2 FP) makes **precision 2.5× more valuable than recall**—every false positive costs half the value of a true positive.

---

## 1. Contest & Scoring Analysis

### 1.1 Scoring Function

| Outcome | Points |
|---------|--------|
| True Positive (correctly detect bot) | +4 |
| False Negative (miss a bot) | -1 |
| False Positive (flag human as bot) | -2 |

**Implication:** Score = `4×TP - 1×FN - 2×FP` = `5×TP - 2×FP - B` (where B = total bots, constant).

- **Break-even precision:** A detection is profitable only if P(bot|flagged) > 2/7 ≈ **28.6%**
- **Optimal strategy:** Use a **high-precision, moderate-recall** classifier. Only flag when confident.
- **Calibration:** Tune decision threshold to maximize expected score, not accuracy or F1.

### 1.2 Submission Constraints

- **1-hour window** from eval release (12:00 PM) to deadline (1:00 PM EST)
- Output format: one user ID per line, identical to `dataset.bots.txt`
- Separate models for English vs. French recommended (organizers note detectors differ by language)
- Final eval will include **unseen bot algorithms** of similar sophistication

---

## 2. Data Analysis Summary

### 2.1 Dataset Overview

| Dataset | Language | Users | Bots | Bot % | Posts |
|---------|----------|-------|------|-------|-------|
| 30 | EN | 275 | 66 | 24.0% | 7,528 |
| 31 | FR | 171 | 27 | 15.8% | 4,643 |
| 32 | EN | 271 | 63 | 23.2% | 8,237 |
| 33 | FR | 172 | 28 | 16.3% | 4,361 |

**Observation:** English datasets have ~24% bots; French ~16%. Train language-specific models.

### 2.2 Discriminative Features (EDA Results)

| Feature | Bot vs Human | Consistency |
|---------|--------------|-------------|
| **tweet_count** | Bots post 35–40 vs humans 22–28 | ✓ All datasets |
| **z_score** | Bots +0.35 to +0.70; humans negative | ✓ All datasets |
| **hashtag_count_mean** | Bots 0.6–0.9; humans 0.2 | ✓ All datasets |
| **char_len_mean** | Bots 124–145; humans 101–111 | ✓ All datasets |
| **has_dm_prompt** (engagement bait) | Bots 0.7–1.0; humans 0.2–0.5 | ✓ Strong signal |
| **url_count_mean** | Bots *fewer* URLs than humans | ✓ Counter-intuitive |
| **profile_has_pipe** (e.g., "|" in bio) | Bots higher in EN | EN only |
| **inter_tweet_std** | Bots more uniform (EN) or mixed (FR) | Language-dependent |
| **profile_has_emoji** | Mixed; FR bots *less* emoji | Language-dependent |

### 2.3 Bot Tweet Patterns (Qualitative)

- **Engagement bait:** "check my bio", "DM", "follow", "opt in", "CLICK"
- **Generic hashtags:** #LoveInTheCrust, #AnimeVsPop, #ManifestingMagic, #TokyoVibes
- **Templated / polished:** Longer, more structured text; sometimes AI-like phrasing
- **Profile:** Pipe separators ("|"), hashtags in bio, formulaic descriptions ("Culinary artist | Recipe creator | #Foodie")
- **Volume:** Consistently high tweet counts (often near 80–100 in 2-day window)

### 2.4 Human Tweet Patterns

- Shorter, casual, typo-rich ("wit", "rn", "tbh")
- More URLs (sharing links)
- Fewer hashtags
- Idiosyncratic bios and names
- Variable posting frequency

---

## 3. State-of-the-Art Methods (Literature)

### 3.1 Key Approaches

1. **BotArtist (2023–2025):** Semi-automatic ML pipeline on 9 public datasets; profile + text features; F1 83% (specific) / 68.5% (general). Emphasizes **user profile features**.

2. **Multimodal (2024–2025):** Profile + text + graph; ~5% improvement over text-only. Our data has **no graph**—focus on profile + text.

3. **LMBot (2024):** Distills graph knowledge into language models for graph-less deployment. Suggests **transformer-based text encoders** can capture behavioral patterns from text alone.

4. **Language-agnostic (ACL 2019):** Account + tweet + behavioral features; 0.988 accuracy, 0.995 AUC; portable across languages. Supports **language-specific models** with shared feature design.

5. **Linguistic embeddings + neural nets:** Text-only models competitive with handcrafted features. **Sentence transformers** (e.g., paraphrase-multilingual) viable for EN+FR.

### 3.2 Feature Categories (Synthesis)

| Category | Examples | Available in Our Data |
|----------|----------|----------------------|
| **Profile** | tweet_count, z_score, description length, location, name patterns | ✓ |
| **Text (aggregate)** | hashtag/URL/mention/emoji rates, length, engagement bait | ✓ |
| **Temporal** | inter-tweet intervals, posts/hour, burstiness | ✓ |
| **Linguistic** | embeddings, formality, repetitiveness | ✓ (from text) |
| **Graph** | followers, retweets, mentions network | ✗ |

---

## 4. Implementation Plan

### 4.1 Architecture: Two-Stage, Language-Specific

```
[Raw Data] → [Feature Extraction] → [Classifier] → [Threshold Tuning] → [Output]
                    ↓
            EN model / FR model (separate)
```

### 4.2 Feature Engineering

#### A. User-Level (from `users` + aggregated `posts`)

| Feature | Description |
|---------|-------------|
| `tweet_count` | Raw count (strong signal) |
| `z_score` | Outlier score for volume |
| `has_description` | 0/1 |
| `has_location` | 0/1 |
| `desc_len`, `name_len` | Character counts |
| `profile_has_pipe` | "|" in name/description |
| `profile_has_hashtag` | "#" in profile |
| `profile_has_emoji` | Emoji in name/description |
| `username_len`, `username_digit_ratio` | Username patterns |
| `desc_word_count` | Word count in description |

#### B. Post-Aggregated (per user)

| Feature | Aggregation |
|---------|-------------|
| `hashtag_count` | mean, max, sum |
| `url_count` | mean, max |
| `mention_count` | mean |
| `char_len`, `word_count` | mean, std |
| `caps_ratio` | mean |
| `has_dm_prompt` | sum, mean (engagement bait) |
| `has_generic_hashtags` | sum |
| `excl_count`, `quest_count` | mean |
| `unique_hashtag_ratio` | diversity |
| `tweet_similarity` | mean pairwise similarity (repetition) |

#### C. Temporal

| Feature | Description |
|---------|-------------|
| `inter_tweet_mean`, `inter_tweet_std` | Posting regularity |
| `posts_per_hour` | Activity rate |
| `temporal_burstiness` | Variance in inter-tweet times |
| `hour_entropy` | Distribution of posting hours |

#### D. Text Embeddings (Optional but High-Value)

- Concatenate user's tweets → encode with `paraphrase-multilingual-MiniLM-L12-v2` or `sentence-transformers/all-MiniLM-L6-v2` (EN) / `dangvantuan/sentence-camembert-large` (FR)
- Use mean pooling → 384/768-dim vector per user
- Feed to classifier or use as additional features

### 4.3 Model Choices

| Option | Pros | Cons |
|--------|------|------|
| **Logistic Regression** | Interpretable, fast, robust to overfitting | May miss interactions |
| **XGBoost/LightGBM** | Handles interactions, good with tabular data | Needs tuning |
| **Random Forest** | Robust, no scaling needed | Can overfit on small data |
| **Neural Net (MLP)** | Can learn complex patterns | Needs more data |
| **Ensemble** | Best of multiple models | More complexity |

**Recommendation:** Start with **LightGBM** (handles mixed types, missing values, fast). Add **calibrated probability** output for threshold tuning. Consider **ensemble** of LightGBM + Logistic Regression for stability.

### 4.4 Threshold Optimization

**Critical:** Do not use default 0.5. Optimize threshold to maximize:

```
score = 4 * TP - 1 * FN - 2 * FP
```

For each candidate threshold τ:
1. Predict bot if P(bot) > τ
2. Compute TP, FN, FP on validation set
3. Compute score
4. Choose τ that maximizes score

**Expected:** Optimal τ likely in 0.6–0.85 range (favor precision).

### 4.5 Training Strategy

1. **Split:** Use datasets 30+32 for EN (train/val split 80/20 or cross-validation); 31+33 for FR.
2. **Cross-validation:** 5-fold stratified CV; tune threshold on validation fold.
3. **Calibration:** Platt scaling or isotonic regression if probabilities are miscalibrated.
4. **Final eval:** Retrain on full EN (30+32) and full FR (31+33) with best hyperparameters.

### 4.6 Robustness to Unseen Bots

- **Regularization:** Avoid overfitting to practice bot types (L1/L2, max_depth limits).
- **Diverse features:** Rely on behavioral (temporal, volume) and linguistic (text) signals that generalize.
- **Conservative threshold:** Slightly higher threshold to reduce FP on novel bot types.
- **Ablation:** Train without strongest single feature to test robustness.

---

## 5. Implementation Checklist

### Phase 1: Data Pipeline (Day 1)
- [ ] Load and parse all 4 datasets
- [ ] Implement feature extraction (profile, post-aggregate, temporal)
- [ ] Build train/val splits (EN: 30+32, FR: 31+33)
- [ ] Validate feature distributions match EDA

### Phase 2: Baseline Model (Day 2)
- [ ] Train LightGBM on EN with default params
- [ ] Implement custom scoring metric
- [ ] Grid search threshold (0.3–0.95, step 0.05)
- [ ] Repeat for FR

### Phase 3: Feature & Model Tuning (Day 3)
- [ ] Add text embeddings (sentence-transformers)
- [ ] Hyperparameter tuning (max_depth, n_estimators, learning_rate)
- [ ] Feature importance analysis
- [ ] Ablation studies

### Phase 4: Ensemble & Calibration (Day 4)
- [ ] Train secondary model (e.g., Logistic Regression)
- [ ] Ensemble: average probabilities or voting
- [ ] Calibrate probabilities
- [ ] Final threshold optimization

### Phase 5: Production Script (Day 5)
- [ ] Single script: `python detect_bots.py <input.json> <output.txt>`
- [ ] Auto-detect language from metadata
- [ ] README with run instructions
- [ ] Dry run on practice data, verify output format

### Phase 6: Eval Day (Feb 14)
- [ ] Download eval datasets at 12:00 PM
- [ ] Run detector, generate `[team].detections.en.txt` and/or `.fr.txt`
- [ ] Submit before 1:00 PM EST

---

## 6. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Unseen bot algorithms | Conservative threshold; diverse feature set |
| French model underperforms | Dedicated FR features; consider CamemBERT embeddings |
| Overfitting to practice data | Strong regularization; cross-dataset validation |
| 1-hour eval window | Fully automated pipeline; test run time |
| API/package failures | Pin dependencies; minimal external calls |

---

## 7. Key Takeaways

1. **Precision > Recall:** Tune for the custom score, not F1.
2. **Language-specific models:** EN and FR have different bot profiles.
3. **Strong signals:** tweet_count, z_score, hashtag_rate, engagement bait, profile structure.
4. **Text matters:** Embeddings can capture subtle linguistic patterns.
5. **Threshold is critical:** Optimize τ on validation score.
6. **Keep it simple:** LightGBM + good features likely sufficient; avoid over-engineering.

---

## Appendix A: Regex Patterns for Engagement Bait

```python
ENGAGEMENT_BAIT_PATTERNS = [
    r'\bDM\b', r'\bfollow\b', r'check my bio', r'\bclick\b', r'opt in',
    r'opt-in', r'link in bio', r'🔒', r'subscribe', r'join (us|me)',
]
```

## Appendix B: Generic Hashtag Patterns (Bot-Like)

```python
GENERIC_HASHTAGS = [
    r'#LoveInTheCrust', r'#AnimeVsPop', r'#TokyoVibes', r'#ManifestingMagic',
    r'#UnexpectedJourneys', r'#relatablefail', r'#PositiveVibes', r'#Gratitude',
]
```

## Appendix C: Suggested requirements.txt

```
lightgbm>=4.0.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
sentence-transformers>=2.2.0  # optional, for embeddings
```

---

*Document generated from contest materials, EDA, and literature review. Revise as new insights emerge.*
