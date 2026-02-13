#!/usr/bin/env python3
"""
Bot or Not Challenge - Production Competition Detector
======================================================
Trains on ALL practice datasets, then applies to unseen evaluation data.

Usage:
    # Training mode (builds model from practice data):
    python bot_detector_v2.py train <ds1.json> <bots1.txt> [<ds2.json> <bots2.txt> ...]

    # Competition mode (applies trained model to new data):
    python bot_detector_v2.py detect <eval_dataset.json> --output team.detections.en.txt

    # All-in-one (train on practice data, detect on eval data):
    python bot_detector_v2.py run <eval_dataset.json> --output team.detections.en.txt \
        --train-data ds30.json:bots30.txt,ds31.json:bots31.txt,...
"""

import json
import sys
import math
import re
import os
import pickle
from collections import defaultdict, Counter
from datetime import datetime
from typing import Dict, List, Set, Tuple, Any, Optional
import statistics

# ============================================================================
# FEATURE EXTRACTION (same as v1 but with additional features)
# ============================================================================

def parse_datetime(dt_str: str) -> datetime:
    return datetime.fromisoformat(dt_str.replace('Z', '+00:00').replace('+00:00', ''))

def extract_features(user: dict, posts: List[dict], dataset_meta: dict) -> dict:
    """Extract 40+ features for a single user."""
    feats = {}
    if not posts:
        return feats
    
    texts = [p['text'] for p in posts]
    posts_sorted = sorted(posts, key=lambda x: x['created_at'])
    times = [parse_datetime(p['created_at']) for p in posts_sorted]
    
    # === 1. VOLUME & METADATA ===
    feats['tweet_count'] = user.get('tweet_count', len(posts))
    feats['z_score'] = user.get('z_score', 0)
    feats['desc_length'] = len(str(user.get('description', '') or ''))
    feats['has_location'] = 1 if user.get('location') and str(user.get('location', '')).lower() not in ('none', '', 'null') else 0
    feats['has_description'] = 1 if feats['desc_length'] > 0 else 0
    
    # === 2. PROFILE ANOMALIES ===
    name = str(user.get('name', '') or '')
    username = str(user.get('username', '') or '')
    location = str(user.get('location', '') or '')
    
    feats['garbled_name'] = 1 if any(ord(c) < 32 for c in name) else 0
    feats['weird_location'] = 1 if any(m in location for m in [':null:', 'O:', '.:']) else 0
    feats['username_trailing_nums'] = 1 if re.search(r'\d{3,}$', username) else 0
    feats['username_underscore_num'] = 1 if re.search(r'_\d+$', username) else 0
    feats['generic_name'] = 1 if name.lower().strip() in ['john doe', 'jane doe'] else 0
    
    # Name-username similarity (bots often have mismatched names)
    name_words = set(name.lower().split())
    user_words = set(re.sub(r'[_\d]', ' ', username.lower()).split())
    name_user_overlap = len(name_words & user_words) / max(len(name_words | user_words), 1)
    feats['name_username_sim'] = name_user_overlap
    
    # === 3. TEMPORAL ===
    if len(times) >= 2:
        gaps = [(times[i+1] - times[i]).total_seconds() for i in range(len(times)-1)]
        
        feats['avg_gap'] = statistics.mean(gaps)
        feats['median_gap'] = statistics.median(gaps)
        feats['min_gap'] = min(gaps)
        feats['max_gap'] = max(gaps)
        feats['std_gap'] = statistics.stdev(gaps) if len(gaps) > 1 else 0
        feats['cv_gap'] = feats['std_gap'] / feats['avg_gap'] if feats['avg_gap'] > 0 else 0
        
        burst_count = sum(1 for g in gaps if g <= 2)
        feats['burst_ratio'] = burst_count / len(gaps)
        feats['burst_count'] = burst_count
        
        same_second = sum(1 for g in gaps if g <= 1)
        feats['same_second_ratio'] = same_second / len(gaps)
        
        round_times = sum(1 for t in times if t.second == 0)
        feats['round_time_ratio'] = round_times / len(times)
        
        # Very short gaps (< 60s) ratio
        short_gaps = sum(1 for g in gaps if g < 60)
        feats['short_gap_ratio'] = short_gaps / len(gaps)
    else:
        for k in ['avg_gap','median_gap','min_gap','max_gap','std_gap','cv_gap',
                   'burst_ratio','burst_count','same_second_ratio','round_time_ratio','short_gap_ratio']:
            feats[k] = 0
    
    hours = [t.hour for t in times]
    hour_counts = Counter(hours)
    total = len(hours)
    feats['hour_entropy'] = -sum((c/total) * math.log2(c/total) for c in hour_counts.values() if c > 0)
    feats['unique_hours'] = len(hour_counts)
    
    off_hour_posts = sum(1 for h in hours if 5 <= h <= 11)
    feats['off_hour_ratio'] = off_hour_posts / len(hours)
    
    # Day distribution
    days = [t.day for t in times]
    day_counts = Counter(days)
    feats['unique_days'] = len(day_counts)
    feats['posts_per_day'] = len(times) / max(len(day_counts), 1)
    
    # === 4. TEXT CONTENT ===
    feats['avg_text_length'] = statistics.mean(len(t) for t in texts)
    feats['std_text_length'] = statistics.stdev(len(t) for t in texts) if len(texts) > 1 else 0
    feats['max_text_length'] = max(len(t) for t in texts)
    feats['min_text_length'] = min(len(t) for t in texts)
    
    feats['url_ratio'] = sum(1 for t in texts if 'https://t.co' in t or 'http' in t) / len(texts)
    
    all_hashtags = sum(t.count('#') for t in texts)
    feats['hashtag_per_tweet'] = all_hashtags / len(texts)
    feats['hashtag_ratio'] = sum(1 for t in texts if '#' in t) / len(texts)
    
    feats['mention_ratio'] = sum(1 for t in texts if '@mention' in t or '@' in t) / len(texts)
    
    feats['excl_per_tweet'] = sum(t.count('!') for t in texts) / len(texts)
    feats['quest_per_tweet'] = sum(t.count('?') for t in texts) / len(texts)
    
    emoji_count = sum(1 for t in texts for c in t if ord(c) > 0x1F600)
    feats['emoji_per_tweet'] = emoji_count / len(texts)
    
    all_words = ' '.join(texts).lower().split()
    feats['lexical_diversity'] = len(set(all_words)) / max(len(all_words), 1)
    feats['total_words'] = len(all_words)
    feats['unique_words'] = len(set(all_words))
    feats['avg_words_per_tweet'] = len(all_words) / len(texts)
    
    # Capitalization features
    all_text = ' '.join(texts)
    alpha_chars = [c for c in all_text if c.isalpha()]
    feats['caps_ratio'] = sum(1 for c in alpha_chars if c.isupper()) / max(len(alpha_chars), 1)
    
    # === 5. LLM LEAKAGE ===
    llm_leak_patterns = [
        r'here are some of my recent tweets',
        r'here are the re-?written',
        r'here are some (?:recent )?rewrites',
        r'here are some (?:modified|revised|alternative) versions',
        r'rewritten tweet',
        r'here are some changes i made',
        r'as an ai\b',
        r'as a language model',
        r'voici (?:quelques|mes) (?:récents? )?tweets',  # French
        r'voici (?:les|des) versions? (?:révisée|modifiée)',  # French
    ]
    leak_count = 0
    for t in texts:
        for pat in llm_leak_patterns:
            if re.search(pat, t.lower()):
                leak_count += 1
                break
    feats['llm_leak_ratio'] = leak_count / len(texts)
    feats['has_llm_leak'] = 1 if leak_count > 0 else 0
    
    typo_urls = sum(1 for t in texts if 'htts://' in t or 'htt://' in t)
    feats['url_typo_count'] = typo_urls
    feats['has_url_typo'] = 1 if typo_urls > 0 else 0
    
    llm_phrases = [
        'mind-blown', 'can\'t stop thinking', 'who else is excited',
        'what a twist', 'so many layers', 'hot take', 'unpopular opinion',
        'let\'s talk about', 'thoughts on this', 'here\'s the thing',
        'can we talk about', 'just a reminder', 'on a real note',
        'seriously though', 'i\'m not gonna lie',
        # French equivalents
        'je n\'arrive pas à', 'qui d\'autre', 'quelle surprise',
        'parlons de', 'sérieusement',
    ]
    llm_phrase_hits = sum(1 for t in texts for ph in llm_phrases if ph.lower() in t.lower())
    feats['llm_phrase_rate'] = llm_phrase_hits / len(texts)
    
    # === 6. CONTENT DUPLICATION ===
    def jaccard(a, b):
        sa, sb = set(a.lower().split()), set(b.lower().split())
        if not sa or not sb: return 0
        return len(sa & sb) / len(sa | sb)
    
    if len(texts) >= 2:
        consec_sims = [jaccard(texts[i], texts[i+1]) for i in range(len(texts)-1)]
        feats['avg_consecutive_sim'] = statistics.mean(consec_sims)
        feats['max_consecutive_sim'] = max(consec_sims)
        
        max_sim = 0
        high_sim_count = 0
        for i in range(min(len(texts), 50)):
            for j in range(i+1, min(len(texts), 50)):
                s = jaccard(texts[i], texts[j])
                max_sim = max(max_sim, s)
                if s > 0.7:
                    high_sim_count += 1
        feats['max_pair_similarity'] = max_sim
        feats['high_sim_pair_count'] = high_sim_count
    else:
        feats['avg_consecutive_sim'] = 0
        feats['max_consecutive_sim'] = 0
        feats['max_pair_similarity'] = 0
        feats['high_sim_pair_count'] = 0
    
    topics = dataset_meta.get('topics', [])
    if not topics:
        # Try to extract from metadata
        meta = dataset_meta.get('metadata', {})
        topics = meta.get('topics', [])
    
    topic_hits = 0
    for topic_info in topics:
        keywords = [kw.lower().replace('#', '') for kw in topic_info.get('keywords', [])]
        topic_name = topic_info.get('topic', '').lower()
        all_kw = keywords + [topic_name]
        for t in texts:
            t_lower = t.lower()
            if any(kw in t_lower for kw in all_kw):
                topic_hits += 1
                break
    feats['topic_coverage'] = topic_hits / max(len(topics), 1)
    
    quote_starts = sum(1 for t in texts if t.startswith("'") or t.startswith('"') or t.startswith('['))
    feats['quote_start_ratio'] = quote_starts / len(texts)
    
    # RT / retweet ratio
    rt_count = sum(1 for t in texts if t.lower().startswith('rt ') or t.lower().startswith('rt:'))
    feats['rt_ratio'] = rt_count / len(texts)
    
    # Media/brand account indicators (these look bot-like but are human-run)
    # - Consistent formatting with emoji bullets
    # - URL in most posts (sharing links)
    # - Description contains brand/business keywords
    desc_lower = str(user.get('description', '') or '').lower()
    brand_keywords = ['media', 'sport', 'pari', 'bet', 'boost', 'news', 'live',
                      'official', 'officiel', 'actu', 'info', 'direct',
                      'podcast', 'émission', 'magazine', 'journal']
    feats['brand_desc'] = 1 if any(kw in desc_lower for kw in brand_keywords) else 0
    
    # High URL + low exclamation combo (media shares links, bots use exclamation)
    feats['url_heavy_low_excl'] = 1 if feats['url_ratio'] > 0.6 and feats['excl_per_tweet'] < 0.3 else 0
    
    return feats


# ============================================================================
# RULE-BASED HARD CLASSIFIERS
# ============================================================================

def apply_hard_rules(user: dict, posts: List[dict], feats: dict) -> Tuple[bool, str]:
    if feats.get('has_llm_leak', 0) == 1:
        return True, "LLM output leakage"
    
    if feats.get('garbled_name', 0) == 1:
        return True, "Garbled name"
    
    if feats.get('weird_location', 0) == 1:
        return True, "Suspicious location"
    
    if feats.get('has_url_typo', 0) == 1:
        return True, "URL typos"
    
    # Content duplication with strong corroboration
    if feats.get('high_sim_pair_count', 0) >= 5:
        signals = 0
        if feats.get('burst_ratio', 0) > 0.15: signals += 1
        if feats.get('hashtag_per_tweet', 0) > 1.5 and feats.get('excl_per_tweet', 0) > 0.5: signals += 1
        if feats.get('hour_entropy', 0) > 3.8: signals += 1
        if feats.get('excl_per_tweet', 0) > 1.0: signals += 1
        if feats.get('llm_phrase_rate', 0) > 0.1: signals += 1
        if signals >= 2 or feats.get('high_sim_pair_count', 0) >= 200:
            return True, "Content duplication + signals"
    
    return False, ""


# ============================================================================
# TRAINING: Build model from multiple practice datasets
# ============================================================================

def load_dataset(ds_path, bots_path):
    with open(ds_path) as f:
        data = json.load(f)
    with open(bots_path) as f:
        bot_ids = set(line.strip() for line in f if line.strip())
    return data, bot_ids

def build_training_data(dataset_pairs: List[Tuple[str, str]]):
    """Build combined feature matrix from multiple datasets."""
    all_features = {}
    all_labels = {}
    
    for ds_path, bots_path in dataset_pairs:
        print(f"  Loading {ds_path}...")
        data, bot_ids = load_dataset(ds_path, bots_path)
        
        posts_by_author = defaultdict(list)
        for p in data['posts']:
            posts_by_author[p['author_id']].append(p)
        
        meta = {k: v for k, v in data.items() if k not in ('posts', 'users')}
        
        for user in data['users']:
            uid = user['id']
            user_posts = posts_by_author.get(uid, [])
            if len(user_posts) >= 1:
                # Use dataset-prefixed uid to avoid collisions
                key = f"{ds_path}::{uid}"
                all_features[key] = extract_features(user, user_posts, meta)
                all_labels[key] = 1 if uid in bot_ids else 0
    
    return all_features, all_labels


def train_model(all_features: dict, all_labels: dict):
    """Train ensemble model on combined data."""
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    import numpy as np
    
    keys = list(all_features.keys())
    feat_names = sorted(all_features[keys[0]].keys())
    
    X = np.array([[all_features[k].get(f, 0) for f in feat_names] for k in keys])
    y = np.array([all_labels[k] for k in keys])
    
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    rf = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=3,
                                 random_state=42, class_weight='balanced')
    gbm = GradientBoostingClassifier(n_estimators=200, max_depth=6, learning_rate=0.08,
                                      min_samples_leaf=3, random_state=42)
    
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('gbm', gbm)],
        voting='soft'
    )
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(ensemble, X_scaled, y, cv=cv, scoring='f1')
    print(f"  CV F1: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    ensemble.fit(X_scaled, y)
    
    # Find optimal threshold using training data (with penalty for overfitting)
    probas = ensemble.predict_proba(X_scaled)[:, 1]
    best_score = -999
    best_thresh = 0.5
    
    for thresh in [i/100 for i in range(15, 85)]:
        preds = (probas >= thresh).astype(int)
        tp = sum(1 for i in range(len(keys)) if preds[i] == 1 and y[i] == 1)
        fp = sum(1 for i in range(len(keys)) if preds[i] == 1 and y[i] == 0)
        fn = sum(1 for i in range(len(keys)) if preds[i] == 0 and y[i] == 1)
        score = 4 * tp - 1 * fn - 4 * fp  # Very conservative on FP for generalization
        if score > best_score:
            best_score = score
            best_thresh = thresh
    
    # Add significant margin for generalization
    best_thresh = min(best_thresh + 0.08, 0.85)
    print(f"  Optimal threshold: {best_thresh:.2f}")
    
    # Feature importance
    rf_solo = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42, class_weight='balanced')
    rf_solo.fit(X_scaled, y)
    importances = sorted(zip(feat_names, rf_solo.feature_importances_), key=lambda x: -x[1])
    print("  Top 10 features:")
    for fname, imp in importances[:10]:
        print(f"    {fname:30s}: {imp:.4f}")
    
    return ensemble, scaler, feat_names, best_thresh


# ============================================================================
# DETECTION: Apply trained model to new dataset
# ============================================================================

def detect_bots(dataset_path: str, model, scaler, feat_names: list, threshold: float,
                output_path: str, bots_path: str = None) -> Set[str]:
    """Apply trained model to detect bots in a new dataset."""
    
    print(f"\nDetecting bots in {dataset_path}...")
    with open(dataset_path) as f:
        data = json.load(f)
    
    posts = data.get('posts', [])
    users = data.get('users', [])
    meta = {k: v for k, v in data.items() if k not in ('posts', 'users')}
    
    print(f"  Users: {len(users)}, Posts: {len(posts)}, Lang: {data.get('lang', '?')}")
    
    posts_by_author = defaultdict(list)
    for p in posts:
        posts_by_author[p['author_id']].append(p)
    
    users_by_id = {u['id']: u for u in users}
    
    # Ground truth (if available)
    bot_ids = set()
    if bots_path and os.path.exists(bots_path):
        with open(bots_path) as f:
            bot_ids = set(line.strip() for line in f if line.strip())
    
    # Extract features
    features_dict = {}
    for user in users:
        uid = user['id']
        user_posts = posts_by_author.get(uid, [])
        if len(user_posts) >= 1:
            features_dict[uid] = extract_features(user, user_posts, meta)
    
    # Phase 1: Rule-based detection
    detected = set()
    for uid in features_dict:
        user = users_by_id.get(uid, {})
        is_bot, reason = apply_hard_rules(user, posts_by_author.get(uid, []), features_dict[uid])
        if is_bot:
            detected.add(uid)
    
    print(f"  Rule-based detections: {len(detected)}")
    
    # Phase 2: ML detection
    import numpy as np
    
    uids = list(features_dict.keys())
    X = np.array([[features_dict[uid].get(f, 0) for f in feat_names] for uid in uids])
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    X_scaled = scaler.transform(X)
    
    probas = model.predict_proba(X_scaled)[:, 1]
    
    for i, uid in enumerate(uids):
        if probas[i] >= threshold:
            detected.add(uid)
    
    # Phase 3: Borderline cases with very strong multi-signal
    for i, uid in enumerate(uids):
        if uid in detected:
            continue
        feats = features_dict[uid]
        if probas[i] >= threshold * 0.7:  # Must be fairly suspicious already
            signals = 0
            if feats.get('burst_ratio', 0) > 0.2: signals += 1
            if feats.get('hashtag_per_tweet', 0) > 2.0: signals += 1
            if feats.get('excl_per_tweet', 0) > 1.0: signals += 1
            if feats.get('hour_entropy', 0) > 4.0: signals += 1
            if feats.get('llm_phrase_rate', 0) > 0.15: signals += 1
            if feats.get('high_sim_pair_count', 0) >= 5: signals += 1
            if feats.get('off_hour_ratio', 0) > 0.4: signals += 1
            if feats.get('quest_per_tweet', 0) > 0.6: signals += 1
            # Negative signal: brand/media accounts often look bot-like
            if feats.get('brand_desc', 0): signals -= 1
            if feats.get('url_heavy_low_excl', 0): signals -= 1
            if signals >= 4:  # Require very strong evidence
                detected.add(uid)
    
    print(f"  Total detections: {len(detected)}")
    
    # Evaluation
    if bot_ids:
        tp = len(detected & bot_ids)
        fp = len(detected - bot_ids)
        fn = len(bot_ids - detected)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        score = 4 * tp - 1 * fn - 2 * fp
        print(f"  TP={tp} FP={fp} FN={fn} P={precision:.3f} R={recall:.3f} F1={f1:.3f} Score={score}")
        
        if fn > 0:
            print(f"  Missed bots:")
            for uid in (bot_ids - detected):
                u = users_by_id.get(uid, {})
                print(f"    {u.get('username','?'):25s} prob={probas[uids.index(uid)]:.3f}")
        if fp > 0:
            print(f"  False positives:")
            for uid in (detected - bot_ids):
                u = users_by_id.get(uid, {})
                prob = probas[uids.index(uid)] if uid in uids else -1
                print(f"    {u.get('username','?'):25s} prob={prob:.3f}")
    
    # Write output
    if output_path:
        with open(output_path, 'w') as f:
            for uid in sorted(detected):
                f.write(uid + '\n')
        print(f"  Written to {output_path}")
    
    return detected


# ============================================================================
# CROSS-DATASET VALIDATION
# ============================================================================

def cross_dataset_validation(dataset_pairs):
    """Leave-one-dataset-out validation to estimate generalization."""
    print("\n=== CROSS-DATASET VALIDATION ===")
    
    total_tp, total_fp, total_fn = 0, 0, 0
    
    for i in range(len(dataset_pairs)):
        test_ds, test_bots = dataset_pairs[i]
        train_pairs = [p for j, p in enumerate(dataset_pairs) if j != i]
        
        print(f"\n  Held out: {test_ds}")
        
        # Train on other datasets
        all_features, all_labels = build_training_data(train_pairs)
        model, scaler, feat_names, threshold = train_model(all_features, all_labels)
        
        # Test on held-out
        detected = detect_bots(test_ds, model, scaler, feat_names, threshold,
                               output_path=None, bots_path=test_bots)
        
        with open(test_bots) as f:
            bot_ids = set(line.strip() for line in f if line.strip())
        
        tp = len(detected & bot_ids)
        fp = len(detected - bot_ids)
        fn = len(bot_ids - detected)
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
    
    total_score = 4 * total_tp - 1 * total_fn - 2 * total_fp
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    
    print(f"\n=== CROSS-DATASET TOTALS ===")
    print(f"  TP={total_tp} FP={total_fp} FN={total_fn}")
    print(f"  Precision={precision:.3f} Recall={recall:.3f}")
    print(f"  Competition Score={total_score}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Bot or Not Challenge Detector v2')
    parser.add_argument('mode', choices=['train', 'detect', 'run', 'validate'],
                        help='Mode: train, detect, run (train+detect), validate (cross-dataset)')
    parser.add_argument('dataset', nargs='?', help='Evaluation dataset path')
    parser.add_argument('--output', '-o', default='detections.txt', help='Output file')
    parser.add_argument('--train-data', '-t', 
                        help='Comma-separated pairs of dataset:bots files for training')
    parser.add_argument('--bots', '-b', help='Ground truth bots file for evaluation')
    parser.add_argument('--model', '-m', default='bot_model.pkl', help='Model save/load path')
    
    args = parser.parse_args()
    
    # Parse training data pairs
    train_pairs = []
    if args.train_data:
        for pair in args.train_data.split(','):
            ds, bots = pair.split(':')
            train_pairs.append((ds, bots))
    
    if args.mode == 'validate':
        if not train_pairs:
            print("Error: --train-data required for validation")
            sys.exit(1)
        cross_dataset_validation(train_pairs)
    
    elif args.mode == 'train':
        if not train_pairs:
            print("Error: --train-data required")
            sys.exit(1)
        print("Building training data...")
        all_features, all_labels = build_training_data(train_pairs)
        print(f"Training on {len(all_features)} users ({sum(all_labels.values())} bots)...")
        model, scaler, feat_names, threshold = train_model(all_features, all_labels)
        
        with open(args.model, 'wb') as f:
            pickle.dump({'model': model, 'scaler': scaler, 'feat_names': feat_names, 
                         'threshold': threshold}, f)
        print(f"Model saved to {args.model}")
    
    elif args.mode == 'detect':
        if not args.dataset:
            print("Error: dataset path required")
            sys.exit(1)
        
        with open(args.model, 'rb') as f:
            saved = pickle.load(f)
        
        detect_bots(args.dataset, saved['model'], saved['scaler'], 
                    saved['feat_names'], saved['threshold'],
                    args.output, args.bots)
    
    elif args.mode == 'run':
        if not args.dataset or not train_pairs:
            print("Error: dataset and --train-data required")
            sys.exit(1)
        
        print("Building training data...")
        all_features, all_labels = build_training_data(train_pairs)
        print(f"Training on {len(all_features)} users ({sum(all_labels.values())} bots)...")
        model, scaler, feat_names, threshold = train_model(all_features, all_labels)
        
        detect_bots(args.dataset, model, scaler, feat_names, threshold,
                    args.output, args.bots)
