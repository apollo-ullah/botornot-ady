#!/usr/bin/env python3
"""
Feature ablation study for bot detection model.
Uses cross-dataset validation (the honest metric) to evaluate feature subsets.
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
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

warnings.filterwarnings('ignore')

base_dir = Path(__file__).parent
lang_map = {30: 'en', 31: 'fr', 32: 'en', 33: 'fr'}

# Load all datasets
datasets = {}
for ds_id in [30, 31, 32, 33]:
    data_path = base_dir / f'dataset.posts&users.{ds_id}.json'
    bots_path = base_dir / f'dataset.bots.{ds_id}.txt'
    datasets[ds_id] = {
        'data': load_dataset(data_path),
        'bots': load_bots(bots_path),
    }

# Prepare per-dataset feature matrices
dfs = {}
for ds_id, ds in datasets.items():
    lang = lang_map[ds_id]
    df, _ = prepare_dataset(ds['data'], ds['bots'], lang)
    dfs[ds_id] = df

en_dfs = [dfs[30], dfs[32]]
fr_dfs = [dfs[31], dfs[33]]


def eval_cross_dataset(train_df, val_df, feature_cols):
    """Train on one dataset, evaluate on another. Returns competition score and accuracy."""
    X_train = train_df[feature_cols].values
    y_train = train_df['label'].values
    X_val = val_df[feature_cols].values
    y_val = val_df['label'].values

    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)

    model = lgb.LGBMClassifier(
        n_estimators=800, max_depth=4, learning_rate=0.03,
        num_leaves=15, min_child_samples=10, subsample=0.7,
        colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=5.0,
        min_split_gain=0.01, random_state=42, verbose=-1,
        is_unbalance=True,
    )
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_val)[:, 1]
    threshold, score, _ = optimize_threshold(y_val, probs)
    y_pred = (probs >= threshold).astype(int)
    s, tp, fn, fp, tn = compute_score(y_val, y_pred)
    acc = (tp + tn) / len(y_val)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return s, acc, prec, rec, threshold


def eval_cv(combined_df, feature_cols):
    """5-fold CV on combined data. Returns score, accuracy."""
    X = combined_df[feature_cols].values
    y = combined_df['label'].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_probs = np.zeros(len(y))

    for train_idx, val_idx in skf.split(X, y):
        model = lgb.LGBMClassifier(
            n_estimators=800, max_depth=4, learning_rate=0.03,
            num_leaves=15, min_child_samples=10, subsample=0.7,
            colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=5.0,
            min_split_gain=0.01, random_state=42, verbose=-1,
            is_unbalance=True,
        )
        model.fit(X[train_idx], y[train_idx])
        all_probs[val_idx] = model.predict_proba(X[val_idx])[:, 1]

    threshold, score, _ = optimize_threshold(y, all_probs)
    y_pred = (all_probs >= threshold).astype(int)
    s, tp, fn, fp, tn = compute_score(y, y_pred)
    acc = (tp + tn) / len(y)
    return s, acc, threshold


# ============================================================================
# 1. BASELINE: All 108 features
# ============================================================================
print("=" * 80)
print("FEATURE ABLATION STUDY")
print("=" * 80)

all_features = get_feature_columns(dfs[30])
print(f"\nTotal features: {len(all_features)}")

# EN cross-dataset baseline
print("\n--- EN BASELINE (all features) ---")
s1, a1, p1, r1, t1 = eval_cross_dataset(dfs[30], dfs[32], all_features)
s2, a2, p2, r2, t2 = eval_cross_dataset(dfs[32], dfs[30], all_features)
print(f"  ds30→ds32: Score={s1}, Acc={a1:.3f}, P={p1:.3f}, R={r1:.3f}, τ={t1:.2f}")
print(f"  ds32→ds30: Score={s2}, Acc={a2:.3f}, P={p2:.3f}, R={r2:.3f}, τ={t2:.2f}")
print(f"  Mean acc: {(a1+a2)/2:.3f}")
en_baseline = (a1 + a2) / 2

# FR cross-dataset baseline
print("\n--- FR BASELINE (all features) ---")
s1, a1, p1, r1, t1 = eval_cross_dataset(dfs[31], dfs[33], all_features)
s2, a2, p2, r2, t2 = eval_cross_dataset(dfs[33], dfs[31], all_features)
print(f"  ds31→ds33: Score={s1}, Acc={a1:.3f}, P={p1:.3f}, R={r1:.3f}, τ={t1:.2f}")
print(f"  ds33→ds31: Score={s2}, Acc={a2:.3f}, P={p2:.3f}, R={r2:.3f}, τ={t2:.2f}")
print(f"  Mean acc: {(a1+a2)/2:.3f}")
fr_baseline = (a1 + a2) / 2


# ============================================================================
# 2. FEATURE CATEGORY ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("FEATURE CATEGORY CONTRIBUTION (leave-one-category-out)")
print("=" * 80)

# Categorize features
categories = {
    'temporal': ['inter_tweet_mean', 'inter_tweet_std', 'inter_tweet_cv',
                 'inter_tweet_min', 'inter_tweet_max', 'inter_tweet_median',
                 'posts_per_hour', 'hour_entropy', 'night_ratio',
                 'rapid_fire_ratio', 'regular_interval_ratio',
                 'temporal_burstiness', 'active_hours', 'max_gap_ratio',
                 'posting_span_hours'],
    'profile': ['tweet_count', 'z_score', 'has_description', 'has_location',
                'desc_len', 'desc_word_count', 'name_len', 'username_len',
                'username_digit_ratio', 'username_has_underscore',
                'profile_has_pipe', 'profile_has_hashtag', 'profile_has_emoji',
                'profile_has_url', 'name_has_special_chars', 'desc_has_newline',
                'desc_pipe_count', 'desc_formulaic', 'username_bot_pattern',
                'name_generic_pattern'],
    'diversity_stylometry': ['ttr', 'hapax_ratio', 'vocab_richness',
                             'avg_word_len', 'long_word_ratio',
                             'content_jaccard_mean', 'content_jaccard_std',
                             'unique_first_words', 'repeated_phrase_ratio',
                             'punctuation_diversity', 'char_len_cv',
                             'tweet_len_uniformity', 'url_consistency',
                             'avg_sentence_completeness'],
    'content_specific': ['topic_diversity', 'formality_score', 'slang_ratio',
                         'just_starter_ratio', 'ai_phrase_ratio',
                         'artificial_caps_ratio', 'generic_life_ratio',
                         'short_generic_ratio'],
    'hashtag': ['total_hashtags', 'unique_hashtags', 'hashtag_diversity',
                'hashtag_per_post', 'max_hashtags_per_tweet', 'hashtag_post_ratio'],
    'cross_user': ['cross_user_sim_mean', 'cross_user_sim_max'],
    'interactions': ['hashtag_volume_interaction', 'entropy_volume_interaction',
                     'url_hashtag_gap'],
}

# Check which features are actually in the dataset
for cat, feats in categories.items():
    present = [f for f in feats if f in all_features]
    missing = [f for f in feats if f not in all_features]
    # Find uncategorized features

uncategorized = [f for f in all_features if not any(f in feats for feats in categories.values())]
print(f"\nUncategorized features ({len(uncategorized)}): {uncategorized[:20]}")

# Text aggregate features (the _mean, _sum, _std, _max suffixed ones)
text_agg_features = [f for f in uncategorized if any(
    f.endswith(s) for s in ('_mean', '_sum', '_std', '_max')
)]
categories['text_aggregates'] = text_agg_features

remaining_uncategorized = [f for f in uncategorized if f not in text_agg_features]
if remaining_uncategorized:
    print(f"Still uncategorized: {remaining_uncategorized}")

print("\nCategory sizes:")
for cat, feats in sorted(categories.items()):
    present = [f for f in feats if f in all_features]
    print(f"  {cat:25s}: {len(present)} features")

# Leave-one-category-out analysis
print("\nLeave-one-category-out (EN cross-dataset):")
for cat, feats in sorted(categories.items()):
    present = [f for f in feats if f in all_features]
    reduced = [f for f in all_features if f not in present]
    if not reduced:
        continue
    s1, a1, _, _, _ = eval_cross_dataset(dfs[30], dfs[32], reduced)
    s2, a2, _, _, _ = eval_cross_dataset(dfs[32], dfs[30], reduced)
    mean_acc = (a1 + a2) / 2
    delta = mean_acc - en_baseline
    print(f"  Without {cat:25s}: acc={mean_acc:.3f} (Δ={delta:+.3f}) {'← HARMFUL (removing helps)' if delta > 0.005 else '← ESSENTIAL' if delta < -0.005 else ''}")

# Single-category analysis
print("\nSingle-category-only (EN cross-dataset):")
for cat, feats in sorted(categories.items()):
    present = [f for f in feats if f in all_features]
    if len(present) < 2:
        continue
    s1, a1, _, _, _ = eval_cross_dataset(dfs[30], dfs[32], present)
    s2, a2, _, _, _ = eval_cross_dataset(dfs[32], dfs[30], present)
    mean_acc = (a1 + a2) / 2
    print(f"  Only {cat:25s}: acc={mean_acc:.3f} ({len(present)} features)")


# ============================================================================
# 3. INDIVIDUAL FEATURE DROP ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("INDIVIDUAL FEATURE DROP ANALYSIS (EN cross-dataset)")
print("=" * 80)

# For content_specific features (the likely overfit ones)
suspect_features = categories['content_specific'] + categories['interactions'] + categories['cross_user']
print(f"\nDropping suspect features one at a time:")
for feat in suspect_features:
    if feat not in all_features:
        continue
    reduced = [f for f in all_features if f != feat]
    s1, a1, _, _, _ = eval_cross_dataset(dfs[30], dfs[32], reduced)
    s2, a2, _, _, _ = eval_cross_dataset(dfs[32], dfs[30], reduced)
    mean_acc = (a1 + a2) / 2
    delta = mean_acc - en_baseline
    print(f"  Without {feat:35s}: acc={mean_acc:.3f} (Δ={delta:+.3f}) {'← drop it' if delta > 0.002 else ''}")


# ============================================================================
# 4. BUILD ROBUST SUBSET
# ============================================================================
print("\n" + "=" * 80)
print("ROBUST FEATURE SUBSET ANALYSIS")
print("=" * 80)

# Core behavioral features (should generalize to any bot type)
core_behavioral = [
    # Temporal (bot behavior = regular posting)
    'hour_entropy', 'inter_tweet_cv', 'temporal_burstiness', 'night_ratio',
    'posting_span_hours', 'inter_tweet_median', 'active_hours', 'max_gap_ratio',
    'rapid_fire_ratio',
    # Text diversity (bot behavior = uniform text)
    'ttr', 'hapax_ratio', 'vocab_richness', 'char_len_cv',
    'punctuation_diversity', 'avg_sentence_completeness',
    'unique_first_words', 'tweet_len_uniformity',
    'avg_word_len', 'long_word_ratio',
    # Profile structure (bot behavior = structured profiles)
    'z_score', 'tweet_count', 'profile_has_pipe', 'desc_formulaic',
    'desc_len', 'name_len', 'username_digit_ratio',
    # Hashtag patterns (bot behavior = hashtag spam)
    'hashtag_diversity', 'hashtag_per_post', 'unique_hashtags',
    # Basic text stats
    'caps_ratio_mean', 'char_len_mean', 'char_len_std',
    'excl_count_mean', 'quest_count_mean',
    'has_url_mean', 'has_hashtag_sum',
]

# Filter to only features that exist
core_behavioral = [f for f in core_behavioral if f in all_features]
print(f"\nCore behavioral features: {len(core_behavioral)}")

s1, a1, p1, r1, t1 = eval_cross_dataset(dfs[30], dfs[32], core_behavioral)
s2, a2, p2, r2, t2 = eval_cross_dataset(dfs[32], dfs[30], core_behavioral)
print(f"  EN ds30→ds32: Score={s1}, Acc={a1:.3f}, P={p1:.3f}, R={r1:.3f}")
print(f"  EN ds32→ds30: Score={s2}, Acc={a2:.3f}, P={p2:.3f}, R={r2:.3f}")
print(f"  Mean acc: {(a1+a2)/2:.3f} (vs baseline {en_baseline:.3f})")

s1, a1, p1, r1, t1 = eval_cross_dataset(dfs[31], dfs[33], core_behavioral)
s2, a2, p2, r2, t2 = eval_cross_dataset(dfs[33], dfs[31], core_behavioral)
print(f"  FR ds31→ds33: Score={s1}, Acc={a1:.3f}, P={p1:.3f}, R={r1:.3f}")
print(f"  FR ds33→ds31: Score={s2}, Acc={a2:.3f}, P={p2:.3f}, R={r2:.3f}")
print(f"  FR Mean acc: {(a1+a2)/2:.3f} (vs baseline {fr_baseline:.3f})")


# Now try adding back features one at a time to see which ones help
print("\n--- Forward selection from core: adding features back ---")
current_best = list(core_behavioral)
candidates = [f for f in all_features if f not in core_behavioral]

# Sort by LightGBM importance first
combined_en = pd.concat(en_dfs, ignore_index=True)
X_all = combined_en[all_features].values
y_all = combined_en['label'].values
X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
temp_lgb = lgb.LGBMClassifier(n_estimators=200, max_depth=4, verbose=-1, random_state=42)
temp_lgb.fit(X_all, y_all)
imp_dict = dict(zip(all_features, temp_lgb.feature_importances_))

candidates_sorted = sorted(candidates, key=lambda f: imp_dict.get(f, 0), reverse=True)

s1_base, a1_base, _, _, _ = eval_cross_dataset(dfs[30], dfs[32], current_best)
s2_base, a2_base, _, _, _ = eval_cross_dataset(dfs[32], dfs[30], current_best)
base_mean = (a1_base + a2_base) / 2

print(f"Starting from core ({len(current_best)} features, acc={base_mean:.3f})")
for feat in candidates_sorted[:30]:  # top 30 candidates
    trial = current_best + [feat]
    s1, a1, _, _, _ = eval_cross_dataset(dfs[30], dfs[32], trial)
    s2, a2, _, _, _ = eval_cross_dataset(dfs[32], dfs[30], trial)
    mean_acc = (a1 + a2) / 2
    delta = mean_acc - base_mean
    if delta > 0.002:
        print(f"  + {feat:35s}: acc={mean_acc:.3f} (Δ={delta:+.3f}) ← ADD")
    elif delta < -0.005:
        print(f"  + {feat:35s}: acc={mean_acc:.3f} (Δ={delta:+.3f}) ← HURTS")

# ============================================================================
# 5. ENSEMBLE SIZE ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("MODEL COUNT ANALYSIS")
print("=" * 80)

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb_lib

def eval_ensemble(train_df, val_df, feature_cols, model_config):
    """Evaluate a specific ensemble configuration."""
    X_train = train_df[feature_cols].values
    y_train = train_df['label'].values
    X_val = val_df[feature_cols].values
    y_val = val_df['label'].values

    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)

    probs = np.zeros(len(y_val))
    total_weight = 0

    for name, model_cls, kwargs, weight, use_scaled in model_config:
        model = model_cls(**kwargs)
        if use_scaled:
            model.fit(X_train_sc, y_train)
            p = model.predict_proba(X_val_sc)[:, 1]
        else:
            model.fit(X_train, y_train)
            p = model.predict_proba(X_val)[:, 1]
        probs += p * weight
        total_weight += weight

    probs /= total_weight
    threshold, score, _ = optimize_threshold(y_val, probs)
    y_pred = (probs >= threshold).astype(int)
    s, tp, fn, fp, tn = compute_score(y_val, y_pred)
    acc = (tp + tn) / len(y_val)
    return s, acc

# Define ensemble configs
lgb_params = dict(n_estimators=800, max_depth=4, learning_rate=0.03, num_leaves=15,
                  min_child_samples=10, subsample=0.7, colsample_bytree=0.7,
                  reg_alpha=1.0, reg_lambda=5.0, min_split_gain=0.01,
                  random_state=42, verbose=-1, is_unbalance=True)
xgb_params = dict(n_estimators=800, max_depth=4, learning_rate=0.03, subsample=0.7,
                  colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=5.0, gamma=0.1,
                  min_child_weight=5, random_state=42, verbosity=0, eval_metric='logloss')
gb_params = dict(n_estimators=500, max_depth=3, learning_rate=0.03, subsample=0.7,
                 min_samples_leaf=10, random_state=42)
rf_params = dict(n_estimators=500, max_depth=6, min_samples_leaf=5,
                 class_weight='balanced', random_state=42, n_jobs=-1)
lr_params = dict(C=0.5, penalty='l2', max_iter=2000, random_state=42,
                 class_weight='balanced', solver='lbfgs')

configs = {
    'LGB only': [('lgb', lgb.LGBMClassifier, lgb_params, 1.0, False)],
    'LGB+XGB': [
        ('lgb', lgb.LGBMClassifier, lgb_params, 0.5, False),
        ('xgb', xgb_lib.XGBClassifier, xgb_params, 0.5, False),
    ],
    'LGB+XGB+GB': [
        ('lgb', lgb.LGBMClassifier, lgb_params, 0.35, False),
        ('xgb', xgb_lib.XGBClassifier, xgb_params, 0.35, False),
        ('gb', GradientBoostingClassifier, gb_params, 0.30, False),
    ],
    'Full 5-model': [
        ('lgb', lgb.LGBMClassifier, lgb_params, 0.25, False),
        ('xgb', xgb_lib.XGBClassifier, xgb_params, 0.25, False),
        ('gb', GradientBoostingClassifier, gb_params, 0.20, False),
        ('rf', RandomForestClassifier, rf_params, 0.20, False),
        ('lr', LogisticRegression, lr_params, 0.10, True),
    ],
}

for name, config in configs.items():
    s1, a1 = eval_ensemble(dfs[30], dfs[32], all_features, config)
    s2, a2 = eval_ensemble(dfs[32], dfs[30], all_features, config)
    mean = (a1 + a2) / 2
    print(f"  {name:20s}: EN cross-dataset acc={mean:.3f} (ds30→32={a1:.3f}, ds32→30={a2:.3f})")

print("\nDone!")
