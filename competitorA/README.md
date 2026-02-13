# Bot or Not Challenge — Detection System

## How It Works

Our detector uses a **hybrid rule-based + machine learning ensemble** architecture that combines high-confidence heuristic rules with a trained Random Forest / Gradient Boosting classifier.

### Architecture

```
Input Dataset → Feature Extraction (45+ features) → Rule-Based Hard Rules → ML Ensemble → Threshold → Detections
                                                          ↓                      ↓
                                                   High-confidence           Probabilistic
                                                   (garbled names,           (trained on
                                                    LLM leakage)            practice data)
```

### Phase 1: Feature Extraction (45+ features, 7 categories)

1. **Volume & Metadata**: tweet count, z-score, description length
2. **Profile Anomalies**: garbled names (non-printable chars), weird locations, generated usernames
3. **Temporal Patterns**: inter-post gap statistics (mean, median, CV), burst detection, posting hour entropy, off-hour ratio
4. **Text Content**: avg length, hashtag/exclamation/question density, emoji rate, lexical diversity, capitalization
5. **LLM Leakage**: regex patterns for leaked prompt framing ("Here are some of my recent tweets"), URL typos
6. **Content Duplication**: Jaccard similarity between all post pairs, consecutive similarity
7. **Brand/Media Indicators**: description keywords, URL-heavy posting patterns (reduces false positives on media accounts)

### Phase 2: Rule-Based Hard Classifiers

High-confidence rules with near-zero false positive rate:
- **Garbled names**: Non-printable `\x1f` characters in display names (broken Unicode in bot pipelines)
- **LLM output leakage**: Posts containing "Here are some of my recent tweets" or "revised versions"
- **URL typos**: Systematic `htts://` instead of `https://`
- **Content duplication + signals**: High duplicate pairs with corroborating behavioral signals

### Phase 3: ML Ensemble

- **Random Forest** (300 trees, depth 12, balanced weights) + **Gradient Boosting** (200 trees, depth 6)
- Soft-voting ensemble with threshold optimized for competition scoring (+4 TP, -1 FN, -2 FP)
- Trained on ALL 4 practice datasets combined (~889 users, 184 bots)
- Conservative threshold with margin for generalization

### Phase 4: Borderline Multi-Signal Rescue

Users with moderate ML probability but 4+ strong behavioral signals are flagged (with negative adjustments for brand/media accounts).

## Performance

### Training Mode (train on all 4, test on each)

| Dataset | Lang | Users | TP | FP | FN | F1    | Score     |
|---------|------|-------|----|----|----|-------|-----------|
| ds_30   | EN   | 275   | 66 | 1  | 0  | 0.992 | 262/264   |
| ds_31   | FR   | 171   | 27 | 0  | 0  | 1.000 | 108/108 ★ |
| ds_32   | EN   | 271   | 63 | 2  | 0  | 0.984 | 248/252   |
| ds_33   | FR   | 172   | 28 | 0  | 0  | 1.000 | 112/112 ★ |
| **Total** | **Both** | **889** | **184** | **3** | **0** | **0.992** | **730/736** |

### Cross-Dataset Validation (leave-one-out)

Total Score: 657/736 (89.3%) — more conservative estimate of generalization.

## Usage

### Quick Run (competition day)

```bash
# Install dependency
pip install scikit-learn

# Detect bots in evaluation dataset
python bot_detector_v2.py run eval_dataset.json \
    --train-data "ds30.json:bots30.txt,ds31.json:bots31.txt,ds32.json:bots32.txt,ds33.json:bots33.txt" \
    --output team.detections.en.txt
```

### Or use the shell script

```bash
chmod +x run_competition.sh
./run_competition.sh eval_en.json eval_fr.json myteam
```

## Dependencies

- Python 3.8+
- scikit-learn

## Key Findings

1. **LLM generation artifacts** are the strongest bot signal — prompt framing leaking into outputs
2. **Unicode encoding corruption** in display names reveals bot pipeline issues
3. **Temporal uniformity** (high hour entropy) separates bots from humans who have natural diurnal cycles
4. **Hashtag/exclamation density** is ~3-5x higher in bots (engagement-bait behavior)
5. **Brand/media accounts** are the main source of false positives — they share many bot-like traits
