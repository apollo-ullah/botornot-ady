#!/usr/bin/env python3
"""
Fast feature selection using cross-dataset validation.
Uses lighter models and batch evaluation.
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from detect_bots import (
    load_dataset, load_bots, prepare_dataset, get_feature_columns,
    compute_score, optimize_threshold
)
import lightgbm as lgb

warnings.filterwarnings('ignore')

base_dir = Path(__file__).parent
lang_map = {30: 'en', 31: 'fr', 32: 'en', 33: 'fr'}

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

all_features = get_feature_columns(dfs[30])
print(f"Total features available: {len(all_features)}")

# Fast LGB config
lgb_fast = dict(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    num_leaves=15, min_child_samples=10, subsample=0.7,
    colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=5.0,
    random_state=42, verbose=-1, is_unbalance=True,
)


def cross_dataset_score_en(feature_cols):
    """Quick EN cross-dataset evaluation."""
    total_score = 0
    total_acc = 0
    worst_acc = 1.0
    for train_id, val_id in [(30, 32), (32, 30)]:
        X_tr = np.nan_to_num(dfs[train_id][feature_cols].values)
        y_tr = dfs[train_id]['label'].values
        X_val = np.nan_to_num(dfs[val_id][feature_cols].values)
        y_val = dfs[val_id]['label'].values
        model = lgb.LGBMClassifier(**lgb_fast)
        model.fit(X_tr, y_tr)
        probs = model.predict_proba(X_val)[:, 1]
        t, s, _ = optimize_threshold(y_val, probs)
        y_pred = (probs >= t).astype(int)
        sc, tp, fn, fp, tn = compute_score(y_val, y_pred)
        acc = (tp + tn) / len(y_val)
        total_score += sc
        total_acc += acc
        worst_acc = min(worst_acc, acc)
    return total_score, total_acc / 2, worst_acc


def cross_dataset_score_fr(feature_cols):
    """Quick FR cross-dataset evaluation."""
    total_score = 0
    total_acc = 0
    worst_acc = 1.0
    for train_id, val_id in [(31, 33), (33, 31)]:
        X_tr = np.nan_to_num(dfs[train_id][feature_cols].values)
        y_tr = dfs[train_id]['label'].values
        X_val = np.nan_to_num(dfs[val_id][feature_cols].values)
        y_val = dfs[val_id]['label'].values
        model = lgb.LGBMClassifier(**lgb_fast)
        model.fit(X_tr, y_tr)
        probs = model.predict_proba(X_val)[:, 1]
        t, s, _ = optimize_threshold(y_val, probs)
        y_pred = (probs >= t).astype(int)
        sc, tp, fn, fp, tn = compute_score(y_val, y_pred)
        acc = (tp + tn) / len(y_val)
        total_score += sc
        total_acc += acc
        worst_acc = min(worst_acc, acc)
    return total_score, total_acc / 2, worst_acc


# ============================================================================
# Step 1: Rank all features by individual cross-dataset performance
# ============================================================================
print("\n=== INDIVIDUAL FEATURE RANKING (EN) ===")
feature_scores = []
base_feat = ['hour_entropy']  # minimal anchor
_, base_acc, base_worst = cross_dataset_score_en(base_feat)

for feat in all_features:
    if feat == 'hour_entropy':
        continue
    trial = ['hour_entropy', feat]
    s, a, w = cross_dataset_score_en(trial)
    feature_scores.append((feat, s, a, w))

feature_scores.sort(key=lambda x: x[2], reverse=True)
print("Top 30 features (paired with hour_entropy):")
for feat, s, a, w in feature_scores[:30]:
    print(f"  {feat:40s}: score={s:4.0f}, mean_acc={a:.3f}, worst_acc={w:.3f}")

print("\nBottom 10 features:")
for feat, s, a, w in feature_scores[-10:]:
    print(f"  {feat:40s}: score={s:4.0f}, mean_acc={a:.3f}, worst_acc={w:.3f}")


# ============================================================================
# Step 2: Greedy forward selection from seed
# ============================================================================
print("\n=== GREEDY FORWARD SELECTION (EN) ===")

# Seed: top 3 individual features
seed = ['hour_entropy', feature_scores[0][0], feature_scores[1][0]]
_, seed_acc, seed_worst = cross_dataset_score_en(seed)
print(f"Seed: {seed} → acc={seed_acc:.3f}, worst={seed_worst:.3f}")

selected = list(seed)
# Candidates: ordered by individual ranking
candidates = [f for f, _, _, _ in feature_scores if f not in selected]

for round_num in range(60):
    _, current_acc, current_worst = cross_dataset_score_en(selected)

    best_feat = None
    best_acc = current_acc
    best_worst = current_worst

    # Test each candidate
    for feat in candidates[:50]:  # Only try top 50 remaining candidates
        trial = selected + [feat]
        s, a, w = cross_dataset_score_en(trial)
        # Prefer improving worst case
        if w > best_worst + 0.001 or (w >= best_worst and a > best_acc + 0.001):
            best_worst = w
            best_acc = a
            best_feat = feat

    if best_feat is None:
        print(f"  No improvement found at round {round_num+1}. Stopping.")
        break

    delta_acc = best_acc - current_acc
    delta_worst = best_worst - current_worst
    selected.append(best_feat)
    candidates.remove(best_feat)
    print(f"  +{best_feat:40s} → acc={best_acc:.3f}(Δ={delta_acc:+.3f}), worst={best_worst:.3f}(Δ={delta_worst:+.3f}) [{len(selected)} feats]")

    if len(selected) >= 45:
        print("  Hit 45 feature cap.")
        break

# Final EN results
s, a, w = cross_dataset_score_en(selected)
print(f"\nFinal EN: {len(selected)} features, acc={a:.3f}, worst={w:.3f}, score={s:.0f}")
print(f"Features: {selected}")

# ============================================================================
# Step 3: Verify on FR
# ============================================================================
print("\n=== VERIFY ON FR ===")
s, a, w = cross_dataset_score_fr(selected)
print(f"EN features on FR: acc={a:.3f}, worst={w:.3f}, score={s:.0f}")

# Also try FR-specific selection
print("\n=== GREEDY FORWARD SELECTION (FR) ===")
fr_feature_scores = []
for feat in all_features:
    if feat == 'night_ratio':
        continue
    trial = ['night_ratio', feat]
    s, a, w = cross_dataset_score_fr(trial)
    fr_feature_scores.append((feat, s, a, w))
fr_feature_scores.sort(key=lambda x: x[2], reverse=True)

fr_seed = ['night_ratio', fr_feature_scores[0][0], fr_feature_scores[1][0]]
_, fr_seed_acc, fr_seed_worst = cross_dataset_score_fr(fr_seed)
print(f"FR Seed: {fr_seed} → acc={fr_seed_acc:.3f}, worst={fr_seed_worst:.3f}")

fr_selected = list(fr_seed)
fr_candidates = [f for f, _, _, _ in fr_feature_scores if f not in fr_selected]

for round_num in range(60):
    _, current_acc, current_worst = cross_dataset_score_fr(fr_selected)
    best_feat = None
    best_acc = current_acc
    best_worst = current_worst
    for feat in fr_candidates[:50]:
        trial = fr_selected + [feat]
        s, a, w = cross_dataset_score_fr(trial)
        if w > best_worst + 0.001 or (w >= best_worst and a > best_acc + 0.001):
            best_worst = w
            best_acc = a
            best_feat = feat
    if best_feat is None:
        print(f"  No improvement found at round {round_num+1}. Stopping.")
        break
    delta_acc = best_acc - current_acc
    delta_worst = best_worst - current_worst
    fr_selected.append(best_feat)
    fr_candidates.remove(best_feat)
    print(f"  +{best_feat:40s} → acc={best_acc:.3f}(Δ={delta_acc:+.3f}), worst={best_worst:.3f}(Δ={delta_worst:+.3f}) [{len(fr_selected)} feats]")
    if len(fr_selected) >= 45:
        break

s, a, w = cross_dataset_score_fr(fr_selected)
print(f"\nFinal FR: {len(fr_selected)} features, acc={a:.3f}, worst={w:.3f}, score={s:.0f}")
print(f"Features: {fr_selected}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
combined = sorted(set(selected) | set(fr_selected))
print(f"Union features: {len(combined)}")

# Compare with all features
s_all, a_all, w_all = cross_dataset_score_en(all_features)
s_sel, a_sel, w_sel = cross_dataset_score_en(selected)
print(f"\nEN: All {len(all_features)} features → acc={a_all:.3f}, worst={w_all:.3f}")
print(f"EN: Selected {len(selected)} features → acc={a_sel:.3f}, worst={w_sel:.3f}")

s_all, a_all, w_all = cross_dataset_score_fr(all_features)
s_sel, a_sel, w_sel = cross_dataset_score_fr(fr_selected)
print(f"FR: All {len(all_features)} features → acc={a_all:.3f}, worst={w_all:.3f}")
print(f"FR: Selected {len(fr_selected)} features → acc={a_sel:.3f}, worst={w_sel:.3f}")
