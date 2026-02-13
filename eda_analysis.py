#!/usr/bin/env python3
"""
Exploratory Data Analysis for Bot or Not Challenge
Analyzes bot vs human characteristics across all datasets
"""

import json
import re
from collections import defaultdict
from pathlib import Path

def load_dataset(dataset_id):
    """Load dataset posts and users."""
    path = Path(f"dataset.posts&users.{dataset_id}.json")
    with open(path) as f:
        data = json.load(f)
    return data

def load_bots(dataset_id):
    """Load bot user IDs."""
    path = Path(f"dataset.bots.{dataset_id}.txt")
    with open(path) as f:
        return set(line.strip() for line in f if line.strip())

def analyze_tweet_text(text):
    """Extract features from tweet text."""
    if not text:
        return {}
    # URL pattern
    url_count = len(re.findall(r'https?://\S+|t\.co/\S+', text))
    # Hashtag count
    hashtag_count = len(re.findall(r'#\w+', text))
    # Mention count
    mention_count = len(re.findall(r'@\w+|@mention', text))
    # Emoji count (simple)
    emoji_count = len(re.findall(r'[\U0001F300-\U0001F9FF]|[\u2600-\u26FF]|[\u2700-\u27BF]', text))
    # Length
    char_len = len(text)
    word_count = len(text.split())
    # All caps ratio
    caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    # Exclamation/question marks
    excl_count = text.count('!')
    quest_count = text.count('?')
    # Generic/templated patterns
    has_dm_prompt = bool(re.search(r'DM|follow|check my bio|click|opt in', text, re.I))
    has_generic_hashtags = bool(re.search(r'#(LoveInTheCrust|AnimeVsPop|TokyoVibes|ManifestingMagic|UnexpectedJourneys|relatablefail)', text))
    return {
        'url_count': url_count,
        'hashtag_count': hashtag_count,
        'mention_count': mention_count,
        'emoji_count': emoji_count,
        'char_len': char_len,
        'word_count': word_count,
        'caps_ratio': caps_ratio,
        'excl_count': excl_count,
        'quest_count': quest_count,
        'has_dm_prompt': has_dm_prompt,
        'has_generic_hashtags': has_generic_hashtags,
    }

def analyze_user(user, posts_by_user):
    """Aggregate user-level features from profile and posts."""
    post_texts = [p['text'] for p in posts_by_user]
    all_features = [analyze_tweet_text(t) for t in post_texts]
    
    # Aggregate post features
    agg = {}
    for key in ['url_count', 'hashtag_count', 'mention_count', 'emoji_count', 'char_len', 'word_count', 'caps_ratio', 'excl_count', 'quest_count', 'has_dm_prompt', 'has_generic_hashtags']:
        vals = [f[key] for f in all_features if key in f]
        if vals:
            agg[f'{key}_mean'] = sum(vals) / len(vals)
            agg[f'{key}_max'] = max(vals)
            agg[f'{key}_sum'] = sum(vals)
    
    # User profile features
    agg['tweet_count'] = user.get('tweet_count', len(posts_by_user))
    agg['z_score'] = user.get('z_score', 0)
    agg['has_description'] = 1 if (user.get('description') or '').strip() else 0
    agg['has_location'] = 1 if (user.get('location') or '').strip() else 0
    agg['name_len'] = len(user.get('name', '') or '')
    agg['desc_len'] = len(user.get('description', '') or '')
    
    # Profile patterns (bot-like)
    desc = (user.get('description') or '').lower()
    name = (user.get('name') or '').lower()
    agg['profile_has_emoji'] = 1 if any(c in (desc + name) for c in '🍳👩‍🍳🎮✨☕️🌟') else 0
    agg['profile_has_pipe'] = 1 if '|' in (desc + name) else 0
    agg['profile_has_hashtag'] = 1 if '#' in (desc + name) else 0
    
    return agg

def compute_temporal_features(posts):
    """Compute posting temporal patterns."""
    from datetime import datetime
    if len(posts) < 2:
        return {'inter_tweet_std': 0, 'inter_tweet_mean': 0, 'posts_per_hour': 0}
    times = sorted([datetime.fromisoformat(p['created_at'].replace('Z', '+00:00')) for p in posts])
    deltas = [(times[i+1] - times[i]).total_seconds() for i in range(len(times)-1)]
    return {
        'inter_tweet_std': (sum((d - sum(deltas)/len(deltas))**2 for d in deltas) / len(deltas))**0.5 if deltas else 0,
        'inter_tweet_mean': sum(deltas) / len(deltas) if deltas else 0,
        'posts_per_hour': len(posts) / max((times[-1] - times[0]).total_seconds() / 3600, 0.01),
    }

def main():
    datasets = [30, 31, 32, 33]
    lang_map = {30: 'en', 31: 'fr', 32: 'en', 33: 'fr'}
    
    print("=" * 80)
    print("BOT OR NOT CHALLENGE - EXPLORATORY DATA ANALYSIS")
    print("=" * 80)
    
    for ds_id in datasets:
        print(f"\n{'='*60}")
        print(f"DATASET {ds_id} ({lang_map[ds_id].upper()})")
        print("="*60)
        
        data = load_dataset(ds_id)
        bots = load_bots(ds_id)
        
        users = {u['id']: u for u in data['users']}
        posts_by_author = defaultdict(list)
        for p in data['posts']:
            posts_by_author[p['author_id']].append(p)
        
        bot_users = [u for u in users if u in bots]
        human_users = [u for u in users if u not in bots]
        
        print(f"Total users: {len(users)}")
        print(f"Bots: {len(bot_users)} ({100*len(bot_users)/len(users):.1f}%)")
        print(f"Humans: {len(human_users)}")
        
        # Aggregate features by class
        bot_feats = []
        human_feats = []
        
        for uid in bot_users:
            if uid in users and uid in posts_by_author:
                u = users[uid]
                posts = posts_by_author[uid]
                feats = analyze_user(u, posts)
                feats.update(compute_temporal_features(posts))
                bot_feats.append(feats)
        
        for uid in human_users:
            if uid in users and uid in posts_by_author:
                u = users[uid]
                posts = posts_by_author[uid]
                feats = analyze_user(u, posts)
                feats.update(compute_temporal_features(posts))
                human_feats.append(feats)
        
        # Compare key features
        key_features = ['tweet_count', 'z_score', 'has_dm_prompt_sum', 'has_generic_hashtags_sum', 
                        'url_count_mean', 'hashtag_count_mean', 'char_len_mean', 'posts_per_hour',
                        'profile_has_emoji', 'profile_has_pipe', 'inter_tweet_std']
        
        print("\nFeature comparison (Bot vs Human mean):")
        for feat in key_features:
            b_vals = [f[feat] for f in bot_feats if feat in f and f[feat] is not None]
            h_vals = [f[feat] for f in human_feats if feat in f and f[feat] is not None]
            if b_vals and h_vals:
                b_mean = sum(b_vals)/len(b_vals)
                h_mean = sum(h_vals)/len(h_vals)
                diff = b_mean - h_mean
                sig = "***" if abs(diff) > 0.1 else ""
                print(f"  {feat:30s}: Bot={b_mean:8.3f}  Human={h_mean:8.3f}  Diff={diff:+.3f} {sig}")
        
        # Sample bot vs human tweets
        print("\nSample BOT tweets (first 3):")
        for uid in list(bot_users)[:3]:
            if uid in posts_by_author:
                for p in posts_by_author[uid][:1]:
                    text = p['text'][:80] + "..." if len(p['text']) > 80 else p['text']
                    print(f"  - {text}")
        
        print("\nSample HUMAN tweets (first 3):")
        for uid in list(human_users)[:3]:
            if uid in posts_by_author:
                for p in posts_by_author[uid][:1]:
                    text = p['text'][:80] + "..." if len(p['text']) > 80 else p['text']
                    print(f"  - {text}")
    
    print("\n" + "="*80)
    print("SCORING REMINDER: +4 TP, -1 FN, -2 FP → Precision critical (FP costs 2x TP value)")
    print("="*80)

if __name__ == "__main__":
    main()
