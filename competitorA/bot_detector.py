#!/usr/bin/env python3
"""
Bot or Not Challenge - Bot Detection System
=============================================
Senior Research Scientist approach: Multi-signal ensemble detector.

Architecture:
1. Feature Extraction (30+ features across 6 categories)
2. Rule-based hard classifiers (near-certain bot indicators)
3. Statistical ensemble classifier (Random Forest + Gradient Boosting)
4. Threshold optimization for asymmetric scoring (+4 TP, -1 FN, -2 FP)

Usage:
    python bot_detector.py <dataset_posts_users.json> [--output detections.txt]
"""

import json
import sys
import math
import re
import os
from collections import defaultdict, Counter
from datetime import datetime
from typing import Dict, List, Set, Tuple, Any
import statistics

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def parse_datetime(dt_str: str) -> datetime:
    return datetime.fromisoformat(dt_str.replace('Z', '+00:00').replace('+00:00', ''))

def extract_features(user: dict, posts: List[dict], dataset_meta: dict) -> dict:
    """Extract 30+ features for a single user."""
    feats = {}
    
    if not posts:
        return feats
    
    texts = [p['text'] for p in posts]
    posts_sorted = sorted(posts, key=lambda x: x['created_at'])
    times = [parse_datetime(p['created_at']) for p in posts_sorted]
    
    # === 1. VOLUME & METADATA FEATURES ===
    feats['tweet_count'] = user.get('tweet_count', len(posts))
    feats['z_score'] = user.get('z_score', 0)
    feats['desc_length'] = len(str(user.get('description', '') or ''))
    feats['has_location'] = 1 if user.get('location') and str(user.get('location', '')).lower() not in ('none', '', 'null') else 0
    feats['has_description'] = 1 if feats['desc_length'] > 0 else 0
    
    # === 2. NAME/PROFILE ANOMALY FEATURES ===
    name = str(user.get('name', '') or '')
    username = str(user.get('username', '') or '')
    location = str(user.get('location', '') or '')
    description = str(user.get('description', '') or '')
    
    # Garbled name (non-printable characters - strong bot signal)
    feats['garbled_name'] = 1 if any(ord(c) < 32 for c in name) else 0
    
    # Weird location markers (:null:, O:location:O, .:HOME:.)
    feats['weird_location'] = 1 if any(marker in location for marker in [':null:', 'O:', '.:']) else 0
    
    # Username has trailing numbers pattern (common in generated usernames)
    feats['username_trailing_nums'] = 1 if re.search(r'\d{3,}$', username) else 0
    
    # Username has underscore + number pattern
    feats['username_underscore_num'] = 1 if re.search(r'_\d+$', username) else 0
    
    # Name looks generic/generated
    generic_names = ['john doe', 'jane doe', 'user', 'bot']
    feats['generic_name'] = 1 if name.lower().strip() in generic_names else 0
    
    # === 3. TEMPORAL FEATURES ===
    if len(times) >= 2:
        gaps = [(times[i+1] - times[i]).total_seconds() for i in range(len(times)-1)]
        
        feats['avg_gap'] = statistics.mean(gaps)
        feats['median_gap'] = statistics.median(gaps)
        feats['min_gap'] = min(gaps)
        feats['max_gap'] = max(gaps)
        feats['std_gap'] = statistics.stdev(gaps) if len(gaps) > 1 else 0
        feats['cv_gap'] = feats['std_gap'] / feats['avg_gap'] if feats['avg_gap'] > 0 else 0
        
        # Same-second bursts (multiple posts within 2 seconds)
        burst_count = 0
        for i in range(len(gaps)):
            if gaps[i] <= 2:
                burst_count += 1
        feats['burst_ratio'] = burst_count / len(gaps) if gaps else 0
        feats['burst_count'] = burst_count
        
        # Exact same-second posts
        same_second = sum(1 for g in gaps if g <= 1)
        feats['same_second_ratio'] = same_second / len(gaps)
        
        # Posts at exact :00 seconds
        round_times = sum(1 for t in times if t.second == 0)
        feats['round_time_ratio'] = round_times / len(times)
    else:
        for k in ['avg_gap','median_gap','min_gap','max_gap','std_gap','cv_gap',
                   'burst_ratio','burst_count','same_second_ratio','round_time_ratio']:
            feats[k] = 0
    
    # Hour distribution entropy (bots tend to be more uniform)
    hours = [t.hour for t in times]
    hour_counts = Counter(hours)
    total = len(hours)
    feats['hour_entropy'] = -sum((c/total) * math.log2(c/total) for c in hour_counts.values() if c > 0)
    feats['unique_hours'] = len(hour_counts)
    
    # Ratio of posts in off-hours (5am-11am UTC - low for US humans)
    off_hour_posts = sum(1 for h in hours if 5 <= h <= 11)
    feats['off_hour_ratio'] = off_hour_posts / len(hours)
    
    # === 4. TEXT CONTENT FEATURES ===
    feats['avg_text_length'] = statistics.mean(len(t) for t in texts)
    feats['std_text_length'] = statistics.stdev(len(t) for t in texts) if len(texts) > 1 else 0
    
    # URL patterns
    feats['url_ratio'] = sum(1 for t in texts if 'https://t.co' in t or 'http' in t) / len(texts)
    
    # Hashtag density
    all_hashtags = sum(t.count('#') for t in texts)
    feats['hashtag_per_tweet'] = all_hashtags / len(texts)
    feats['hashtag_ratio'] = sum(1 for t in texts if '#' in t) / len(texts)
    
    # Mention patterns
    feats['mention_ratio'] = sum(1 for t in texts if '@mention' in t or '@' in t) / len(texts)
    
    # Exclamation/question mark density
    feats['excl_per_tweet'] = sum(t.count('!') for t in texts) / len(texts)
    feats['quest_per_tweet'] = sum(t.count('?') for t in texts) / len(texts)
    
    # Emoji density  
    emoji_count = sum(1 for t in texts for c in t if ord(c) > 0x1F600)
    feats['emoji_per_tweet'] = emoji_count / len(texts)
    
    # Lexical diversity (type-token ratio)
    all_words = ' '.join(texts).lower().split()
    feats['lexical_diversity'] = len(set(all_words)) / max(len(all_words), 1)
    feats['total_words'] = len(all_words)
    feats['unique_words'] = len(set(all_words))
    
    # === 5. LLM LEAKAGE / BOT SIGNATURE FEATURES ===
    
    # Direct LLM output leakage
    llm_leak_patterns = [
        r'here are some of my recent tweets',
        r'here are the re-?written',
        r'here are some (?:recent )?rewrites',
        r'here are some (?:modified|revised|alternative) versions',
        r'rewritten tweet',
        r'here are some changes i made',
        r'as an ai',
        r'as a language model',
    ]
    leak_count = 0
    for t in texts:
        for pat in llm_leak_patterns:
            if re.search(pat, t.lower()):
                leak_count += 1
                break
    feats['llm_leak_ratio'] = leak_count / len(texts)
    feats['has_llm_leak'] = 1 if leak_count > 0 else 0
    
    # Suspicious URL typos (htts:// instead of https://)
    typo_urls = sum(1 for t in texts if 'htts://' in t or 'htt://' in t)
    feats['url_typo_count'] = typo_urls
    
    # Stereotypical LLM conversational phrases
    llm_phrases = [
        'mind-blown', 'can\'t stop thinking', 'who else is excited',
        'what a twist', 'so many layers', 'hot take', 'unpopular opinion',
        'ngl', 'let\'s talk about', 'thoughts on this',
        'here\'s the thing', 'i\'m not gonna lie', 'can we talk about',
        'just a reminder', 'on a real note', 'seriously though',
    ]
    llm_phrase_hits = sum(1 for t in texts for ph in llm_phrases if ph.lower() in t.lower())
    feats['llm_phrase_rate'] = llm_phrase_hits / len(texts)
    
    # === 6. CONTENT DUPLICATION & COHERENCE ===
    
    # Near-duplicate detection (Jaccard similarity between consecutive posts)
    def jaccard(a, b):
        sa, sb = set(a.lower().split()), set(b.lower().split())
        if not sa or not sb:
            return 0
        return len(sa & sb) / len(sa | sb)
    
    if len(texts) >= 2:
        sims = [jaccard(texts[i], texts[i+1]) for i in range(len(texts)-1)]
        feats['avg_consecutive_sim'] = statistics.mean(sims)
        feats['max_consecutive_sim'] = max(sims)
        
        # All-pairs max similarity (check for repeated content)
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
    
    # Topic breadth (how many different dataset topics the user engages with)
    topics = dataset_meta.get('topics', [])
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
    
    # Posts starting with certain patterns
    quote_starts = sum(1 for t in texts if t.startswith("'") or t.startswith('"') or t.startswith('['))
    feats['quote_start_ratio'] = quote_starts / len(texts)
    
    return feats


# ============================================================================
# RULE-BASED HARD CLASSIFIERS
# ============================================================================

def apply_hard_rules(user: dict, posts: List[dict], feats: dict) -> Tuple[bool, str]:
    """
    Apply high-confidence rules that almost certainly indicate bots.
    Returns (is_bot, reason).
    """
    # Rule 1: LLM output leakage
    if feats.get('has_llm_leak', 0) == 1:
        return True, "LLM output leakage detected"
    
    # Rule 2: Garbled name (non-printable characters)
    if feats.get('garbled_name', 0) == 1:
        return True, "Garbled name with non-printable characters"
    
    # Rule 3: Weird location markers
    if feats.get('weird_location', 0) == 1:
        return True, "Suspicious location format"
    
    # Rule 4: URL typos (htts://)
    if feats.get('url_typo_count', 0) > 0:
        return True, "Suspicious URL typos"
    
    # Rule 5: Very high content duplication + other bot signals
    # (Humans can also be repetitive - spammers, live-tweeters, template posters)
    # So require duplication + at least TWO other signals, or extreme duplication
    if feats.get('high_sim_pair_count', 0) >= 5:
        other_signals = 0
        if feats.get('burst_ratio', 0) > 0.15: other_signals += 1
        if feats.get('hashtag_per_tweet', 0) > 1.5 and feats.get('excl_per_tweet', 0) > 0.5: other_signals += 1
        if feats.get('hour_entropy', 0) > 3.8: other_signals += 1
        if feats.get('excl_per_tweet', 0) > 1.0: other_signals += 1
        if feats.get('garbled_name', 0): other_signals += 1
        if feats.get('weird_location', 0): other_signals += 1
        if feats.get('llm_phrase_rate', 0) > 0.1: other_signals += 1
        if other_signals >= 2 or feats.get('high_sim_pair_count', 0) >= 200:
            return True, "High content duplication + corroborating signals"
    
    return False, ""


# ============================================================================
# STATISTICAL CLASSIFIER
# ============================================================================

def train_classifier(features_dict: Dict[str, dict], bot_ids: Set[str]):
    """Train an ensemble classifier. Returns trained model and feature names."""
    try:
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import cross_val_score
        import numpy as np
        
        # Prepare feature matrix
        uids = list(features_dict.keys())
        feat_names = sorted(features_dict[uids[0]].keys())
        
        X = np.array([[features_dict[uid].get(f, 0) for f in feat_names] for uid in uids])
        y = np.array([1 if uid in bot_ids else 0 for uid in uids])
        
        # Handle NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Ensemble of RF and GBM
        rf = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=3, 
                                     random_state=42, class_weight='balanced')
        gbm = GradientBoostingClassifier(n_estimators=150, max_depth=5, learning_rate=0.1,
                                          random_state=42)
        
        ensemble = VotingClassifier(
            estimators=[('rf', rf), ('gbm', gbm)],
            voting='soft'
        )
        
        ensemble.fit(X_scaled, y)
        
        # Cross-validation
        cv_scores = cross_val_score(ensemble, X_scaled, y, cv=5, scoring='f1')
        print(f"  Cross-validation F1: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        # Feature importance (from RF)
        rf_solo = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight='balanced')
        rf_solo.fit(X_scaled, y)
        importances = sorted(zip(feat_names, rf_solo.feature_importances_), key=lambda x: -x[1])
        print("  Top 15 features:")
        for fname, imp in importances[:15]:
            print(f"    {fname:30s}: {imp:.4f}")
        
        return ensemble, scaler, feat_names, uids, X_scaled, y
        
    except ImportError:
        print("  sklearn not available, falling back to rule-based only")
        return None, None, None, None, None, None


def optimize_threshold(model, scaler, X, y, uids, bot_ids):
    """
    Find optimal probability threshold to maximize the competition score:
    +4 TP, -1 FN, -2 FP
    
    We use leave-one-out style: since we're training on the same data,
    we need to be conservative to generalize. Use a slightly higher threshold.
    """
    import numpy as np
    
    probas = model.predict_proba(X)[:, 1]
    
    best_score = -999
    best_thresh = 0.5
    
    for thresh in [i/100 for i in range(10, 90)]:
        preds = (probas >= thresh).astype(int)
        tp = sum(1 for i, uid in enumerate(uids) if preds[i] == 1 and uid in bot_ids)
        fp = sum(1 for i, uid in enumerate(uids) if preds[i] == 1 and uid not in bot_ids)
        fn = sum(1 for i, uid in enumerate(uids) if preds[i] == 0 and uid in bot_ids)
        
        # Use slightly penalized FP to be more conservative for generalization
        score = 4 * tp - 1 * fn - 2.5 * fp
        if score > best_score:
            best_score = score
            best_thresh = thresh
    
    # Add a small margin for generalization
    best_thresh = min(best_thresh + 0.05, 0.85)
    
    # Recompute with actual scoring
    preds = (probas >= best_thresh).astype(int)
    tp = sum(1 for i, uid in enumerate(uids) if preds[i] == 1 and uid in bot_ids)
    fp = sum(1 for i, uid in enumerate(uids) if preds[i] == 1 and uid not in bot_ids)
    fn = sum(1 for i, uid in enumerate(uids) if preds[i] == 0 and uid in bot_ids)
    actual_score = 4 * tp - 1 * fn - 2 * fp
    
    print(f"  Optimal threshold: {best_thresh:.2f} (score: {actual_score})")
    return best_thresh


# ============================================================================
# MAIN DETECTION PIPELINE
# ============================================================================

def detect_bots(dataset_path: str, bots_path: str = None, output_path: str = None) -> Set[str]:
    """
    Main detection pipeline.
    
    Args:
        dataset_path: Path to dataset_posts_users.json
        bots_path: Optional path to ground truth bots file (for evaluation)
        output_path: Path to write detected bot IDs
    
    Returns:
        Set of detected bot user IDs
    """
    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    posts = data.get('posts', [])
    users = data.get('users', [])
    metadata = data.get('metadata', {})
    
    # Extract metadata from top level if not nested
    if not metadata:
        metadata = {k: v for k, v in data.items() if k not in ('posts', 'users')}
    
    print(f"  Users: {len(users)}, Posts: {len(posts)}")
    print(f"  Language: {data.get('lang', 'unknown')}")
    
    # Group posts by author
    posts_by_author = defaultdict(list)
    for p in posts:
        posts_by_author[p['author_id']].append(p)
    
    users_by_id = {u['id']: u for u in users}
    
    # Load ground truth if available
    bot_ids = set()
    if bots_path and os.path.exists(bots_path):
        with open(bots_path, 'r') as f:
            bot_ids = set(line.strip() for line in f if line.strip())
        print(f"  Ground truth bots: {len(bot_ids)}")
    
    # === PHASE 1: Feature Extraction ===
    print("\nPhase 1: Extracting features...")
    features_dict = {}
    for user in users:
        uid = user['id']
        user_posts = posts_by_author.get(uid, [])
        if len(user_posts) >= 1:
            features_dict[uid] = extract_features(user, user_posts, metadata)
    
    print(f"  Extracted features for {len(features_dict)} users")
    
    # === PHASE 2: Rule-based detection ===
    print("\nPhase 2: Applying hard rules...")
    rule_detected = set()
    for uid in features_dict:
        user = users_by_id.get(uid, {})
        user_posts = posts_by_author.get(uid, [])
        is_bot, reason = apply_hard_rules(user, user_posts, features_dict[uid])
        if is_bot:
            rule_detected.add(uid)
            if bot_ids:
                status = "✓" if uid in bot_ids else "✗"
                print(f"  {status} Rule detected: {user.get('username', '?'):25s} - {reason}")
    
    print(f"  Rule-based detections: {len(rule_detected)}")
    
    # === PHASE 3: Statistical classifier ===
    print("\nPhase 3: Training statistical classifier...")
    detected = set(rule_detected)
    
    if bot_ids and len(features_dict) > 20:
        result = train_classifier(features_dict, bot_ids)
        model, scaler, feat_names, uids, X, y = result
        
        if model is not None:
            import numpy as np
            
            # Optimize threshold
            thresh = optimize_threshold(model, scaler, X, y, uids, bot_ids)
            
            # Get predictions
            probas = model.predict_proba(X)[:, 1]
            
            for i, uid in enumerate(uids):
                if probas[i] >= thresh:
                    detected.add(uid)
            
            # Also add borderline cases that have strong individual signals
            for i, uid in enumerate(uids):
                feats = features_dict[uid]
                if probas[i] >= 0.3:  # Somewhat suspicious
                    signals = 0
                    if feats.get('burst_ratio', 0) > 0.2: signals += 1
                    if feats.get('hashtag_per_tweet', 0) > 1.5: signals += 1
                    if feats.get('excl_per_tweet', 0) > 1.0: signals += 1
                    if feats.get('hour_entropy', 0) > 3.8: signals += 1
                    if feats.get('llm_phrase_rate', 0) > 0.1: signals += 1
                    if feats.get('high_sim_pair_count', 0) >= 3: signals += 1
                    if signals >= 3:
                        detected.add(uid)
    else:
        # No ground truth - use unsupervised approach with aggressive rule-based
        print("  No ground truth available, using threshold-based detection...")
        for uid, feats in features_dict.items():
            score = 0
            # Weight signals
            if feats.get('garbled_name', 0): score += 5
            if feats.get('weird_location', 0): score += 5
            if feats.get('has_llm_leak', 0): score += 5
            if feats.get('url_typo_count', 0) > 0: score += 4
            if feats.get('burst_ratio', 0) > 0.15: score += 3
            if feats.get('same_second_ratio', 0) > 0.1: score += 3
            if feats.get('hashtag_per_tweet', 0) > 1.5: score += 2
            if feats.get('excl_per_tweet', 0) > 0.8: score += 2
            if feats.get('quest_per_tweet', 0) > 0.6: score += 1
            if feats.get('hour_entropy', 0) > 3.5: score += 2
            if feats.get('off_hour_ratio', 0) > 0.35: score += 2
            if feats.get('high_sim_pair_count', 0) >= 3: score += 3
            if feats.get('llm_phrase_rate', 0) > 0.08: score += 2
            if feats.get('avg_text_length', 0) > 150: score += 1
            if feats.get('z_score', 0) > 1.0: score += 1
            if feats.get('tweet_count', 0) >= 40: score += 1
            if feats.get('topic_coverage', 0) > 0.5: score += 1
            
            if score >= 5:
                detected.add(uid)
    
    # === PHASE 4: Evaluation ===
    print(f"\n{'='*60}")
    print(f"RESULTS: Detected {len(detected)} bots out of {len(users)} users")
    
    if bot_ids:
        tp = len(detected & bot_ids)
        fp = len(detected - bot_ids)
        fn = len(bot_ids - detected)
        tn = len(set(u['id'] for u in users) - bot_ids - detected)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        competition_score = 4 * tp - 1 * fn - 2 * fp
        
        print(f"  True Positives:  {tp:3d} (× +4 = {4*tp:+4d})")
        print(f"  False Positives: {fp:3d} (× -2 = {-2*fp:+4d})")
        print(f"  False Negatives: {fn:3d} (× -1 = {-1*fn:+4d})")
        print(f"  True Negatives:  {tn:3d}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall:    {recall:.3f}")
        print(f"  F1 Score:  {f1:.3f}")
        print(f"  COMPETITION SCORE: {competition_score}")
        
        # Show false positives and false negatives for debugging
        if fp > 0:
            print(f"\n  FALSE POSITIVES (humans flagged as bots):")
            for uid in (detected - bot_ids):
                u = users_by_id.get(uid, {})
                print(f"    {u.get('username','?'):25s} tweets={u.get('tweet_count',0)}")
        
        if fn > 0:
            print(f"\n  FALSE NEGATIVES (bots missed):")
            for uid in (bot_ids - detected):
                u = users_by_id.get(uid, {})
                f = features_dict.get(uid, {})
                print(f"    {u.get('username','?'):25s} tweets={u.get('tweet_count',0)} "
                      f"burst={f.get('burst_ratio',0):.2f} htag={f.get('hashtag_per_tweet',0):.2f} "
                      f"excl={f.get('excl_per_tweet',0):.2f} h_ent={f.get('hour_entropy',0):.2f}")
    
    # === PHASE 5: Output ===
    if output_path:
        with open(output_path, 'w') as f:
            for uid in sorted(detected):
                f.write(uid + '\n')
        print(f"\nDetections written to {output_path}")
    
    return detected


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python bot_detector.py <dataset.json> [bots.txt] [--output out.txt]")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    bots_path = None
    output_path = None
    
    args = sys.argv[2:]
    i = 0
    while i < len(args):
        if args[i] == '--output' and i + 1 < len(args):
            output_path = args[i + 1]
            i += 2
        else:
            bots_path = args[i]
            i += 1
    
    if output_path is None:
        output_path = 'detections.txt'
    
    detect_bots(dataset_path, bots_path, output_path)
