#!/usr/bin/env python3
"""
Greedy forward feature selection using cross-dataset validation.
Start from the minimal proven core and add features only if they improve generalization.
"""

import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

from detect_bots import (
    load_dataset, load_bots, prepare_dataset, get_feature_columns,
    compute_score, optimize_threshold
)
import lightgbm as lgb

warnings.filterwarnings('ignore')

base_dir = Path(__file__).parent
lang_map = {30: 'en', 31: 'fr', 32: 'en', 33: 'fr'}

# Load datasets
datasets = {}
for ds_id in [30, 31, 32, 33]:
    datasets[ds_id] = {
        'data': load_dataset(base_dir / f'dataset.posts&users.{ds_id}.json'),
        'bots': load_bots(base_dir / f'dataset.bots.{ds_id}.txt'),
    }

dfs = {}
for ds_id, ds in datasets.items():
    df, _ = prepare_dataset(ds['data'], ds['bots'], lang_map[ds_id])
    dfs[ds_id] = df


def eval_cross_dataset_en(feature_cols):
    """Evaluate EN cross-dataset mean accuracy."""
    results = []
    for train_id, val_id in [(30, 32), (32, 30)]:
        X_tr = np.nan_to_num(dfs[train_id][feature_cols].values)
        y_tr = dfs[train_id]['label'].values
        X_val = np.nan_to_num(dfs[val_id][feature_cols].values)
        y_val = dfs[val_id]['label'].values

        model = lgb.LGBMClassifier(
            n_estimators=800, max_depth=4, learning_rate=0.03,
            num_leaves=15, min_child_samples=10, subsample=0.7,
            colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=5.0,
            min_split_gain=0.01, random_state=42, verbose=-1,
            is_unbalance=True,
        )
        model.fit(X_tr, y_tr)
        probs = model.predict_proba(X_val)[:, 1]
        threshold, score, _ = optimize_threshold(y_val, probs)
        y_pred = (probs >= threshold).astype(int)
        s, tp, fn, fp, tn = compute_score(y_val, y_pred)
        acc = (tp + tn) / len(y_val)
        results.append((s, acc))

    mean_score = np.mean([r[0] for r in results])
    mean_acc = np.mean([r[1] for r in results])
    # Also return the min accuracy (worst case matters more)
    min_acc = min(r[1] for r in results)
    return mean_score, mean_acc, min_acc, results


def eval_cross_dataset_fr(feature_cols):
    """Evaluate FR cross-dataset mean accuracy."""
    results = []
    for train_id, val_id in [(31, 33), (33, 31)]:
        X_tr = np.nan_to_num(dfs[train_id][feature_cols].values)
        y_tr = dfs[train_id]['label'].values
        X_val = np.nan_to_num(dfs[val_id][feature_cols].values)
        y_val = dfs[val_id]['label'].values

        model = lgb.LGBMClassifier(
            n_estimators=800, max_depth=4, learning_rate=0.03,
            num_leaves=15, min_child_samples=10, subsample=0.7,
            colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=5.0,
            min_split_gain=0.01, random_state=42, verbose=-1,
            is_unbalance=True,
        )
        model.fit(X_tr, y_tr)
        probs = model.predict_proba(X_val)[:, 1]
        threshold, score, _ = optimize_threshold(y_val, probs)
        y_pred = (probs >= threshold).astype(int)
        s, tp, fn, fp, tn = compute_score(y_val, y_pred)
        acc = (tp + tn) / len(y_val)
        results.append((s, acc))

    mean_score = np.mean([r[0] for r in results])
    mean_acc = np.mean([r[1] for r in results])
    min_acc = min(r[1] for r in results)
    return mean_score, mean_acc, min_acc, results


all_features = get_feature_columns(dfs[30])

# ============================================================================
# GREEDY FORWARD SELECTION FOR EN
# ============================================================================
print("=" * 80)
print("GREEDY FORWARD FEATURE SELECTION (EN, cross-dataset metric)")
print("=" * 80)

# Start with top 5 most important individual features
seed_features = [
    'hour_entropy',          # #1 in LGB importance, temporal behavioral
    'inter_tweet_cv',        # #2, temporal behavioral
    'hashtag_diversity',     # strong discriminator
    'hapax_ratio',           # text diversity
    'night_ratio',           # temporal behavioral
]

# Verify seed
ms, ma, mi, _ = eval_cross_dataset_en(seed_features)
print(f"Seed ({len(seed_features)} features): mean_acc={ma:.3f}, min_acc={mi:.3f}, mean_score={ms:.0f}")

selected = list(seed_features)
remaining = [f for f in all_features if f not in selected]

# Greedy forward selection
improvement_threshold = 0.002  # Only add if >0.2% improvement
rounds_without_improvement = 0
max_features = 50  # Hard cap

for round_num in range(len(remaining)):
    if rounds_without_improvement >= 10 or len(selected) >= max_features:
        break

    best_feat = None
    best_acc = -1
    best_min = -1
    current_ms, current_ma, current_mi, _ = eval_cross_dataset_en(selected)

    for feat in remaining:
        trial = selected + [feat]
        ms, ma, mi, _ = eval_cross_dataset_en(trial)
        # Optimize for min accuracy (worst case) rather than mean
        if mi > best_min or (mi == best_min and ma > best_acc):
            best_min = mi
            best_acc = ma
            best_feat = feat
            best_score = ms

    delta_min = best_min - current_mi
    delta_mean = best_acc - current_ma

    if delta_min >= improvement_threshold or (delta_min >= 0 and delta_mean >= improvement_threshold):
        selected.append(best_feat)
        remaining.remove(best_feat)
        rounds_without_improvement = 0
        print(f"  Round {round_num+1}: +{best_feat:40s} → min={best_min:.3f}(Δ={delta_min:+.3f}), mean={best_acc:.3f}(Δ={delta_mean:+.3f}), score={best_score:.0f} [{len(selected)} feats]")
    else:
        rounds_without_improvement += 1
        if rounds_without_improvement == 1:
            print(f"  Round {round_num+1}: best candidate {best_feat} only gives Δ_min={delta_min:+.3f}, Δ_mean={delta_mean:+.3f} — skipping")

ms, ma, mi, details = eval_cross_dataset_en(selected)
print(f"\nFinal EN feature set ({len(selected)} features):")
print(f"  Mean acc: {ma:.3f}, Min acc: {mi:.3f}, Mean score: {ms:.0f}")
print(f"  ds30→ds32: score={details[0][0]}, acc={details[0][1]:.3f}")
print(f"  ds32→ds30: score={details[1][0]}, acc={details[1][1]:.3f}")
print(f"  Features: {selected}")

# ============================================================================
# VERIFY ON FR TOO
# ============================================================================
print("\n" + "=" * 80)
print("VERIFY SELECTED FEATURES ON FR")
print("=" * 80)
ms, ma, mi, details = eval_cross_dataset_fr(selected)
print(f"FR with EN-selected features ({len(selected)} features):")
print(f"  Mean acc: {ma:.3f}, Min acc: {mi:.3f}, Mean score: {ms:.0f}")
print(f"  ds31→ds33: score={details[0][0]}, acc={details[0][1]:.3f}")
print(f"  ds33→ds31: score={details[1][0]}, acc={details[1][1]:.3f}")

# Also do FR-specific forward selection
print("\n" + "=" * 80)
print("GREEDY FORWARD FEATURE SELECTION (FR, cross-dataset metric)")
print("=" * 80)

fr_seed = ['night_ratio', 'hapax_ratio', 'hour_entropy', 'hashtag_diversity', 'char_len_cv']
ms, ma, mi, _ = eval_cross_dataset_fr(fr_seed)
print(f"Seed ({len(fr_seed)} features): mean_acc={ma:.3f}, min_acc={mi:.3f}")

fr_selected = list(fr_seed)
fr_remaining = [f for f in all_features if f not in fr_selected]
rounds_without_improvement = 0

for round_num in range(len(fr_remaining)):
    if rounds_without_improvement >= 10 or len(fr_selected) >= max_features:
        break

    best_feat = None
    best_acc = -1
    best_min = -1
    current_ms, current_ma, current_mi, _ = eval_cross_dataset_fr(fr_selected)

    for feat in fr_remaining:
        trial = fr_selected + [feat]
        ms, ma, mi, _ = eval_cross_dataset_fr(trial)
        if mi > best_min or (mi == best_min and ma > best_acc):
            best_min = mi
            best_acc = ma
            best_feat = feat
            best_score = ms

    delta_min = best_min - current_mi
    delta_mean = best_acc - current_ma

    if delta_min >= improvement_threshold or (delta_min >= 0 and delta_mean >= improvement_threshold):
        fr_selected.append(best_feat)
        fr_remaining.remove(best_feat)
        rounds_without_improvement = 0
        print(f"  Round {round_num+1}: +{best_feat:40s} → min={best_min:.3f}(Δ={delta_min:+.3f}), mean={best_acc:.3f}(Δ={delta_mean:+.3f}) [{len(fr_selected)} feats]")
    else:
        rounds_without_improvement += 1
        if rounds_without_improvement == 1:
            print(f"  Round {round_num+1}: best candidate {best_feat} only gives Δ_min={delta_min:+.3f}, Δ_mean={delta_mean:+.3f} — skipping")

ms, ma, mi, details = eval_cross_dataset_fr(fr_selected)
print(f"\nFinal FR feature set ({len(fr_selected)} features):")
print(f"  Mean acc: {ma:.3f}, Min acc: {mi:.3f}, Mean score: {ms:.0f}")
print(f"  Features: {fr_selected}")

# ============================================================================
# UNION SET FOR SHARED FEATURE ENGINEERING
# ============================================================================
print("\n" + "=" * 80)
print("COMBINED FEATURE SET")
print("=" * 80)

combined = sorted(set(selected) | set(fr_selected))
print(f"Union of EN + FR features: {len(combined)} features")
print(f"Features: {combined}")

# Verify combined works for both
ms, ma, mi, _ = eval_cross_dataset_en(combined)
print(f"\nEN with combined set: mean_acc={ma:.3f}, min_acc={mi:.3f}, score={ms:.0f}")
ms, ma, mi, _ = eval_cross_dataset_fr(combined)
print(f"FR with combined set: mean_acc={ma:.3f}, min_acc={mi:.3f}, score={ms:.0f}")
