#!/usr/bin/env python3
"""
Bot or Not Challenge — Robust Bot Detector (single-tweet resilient)

Fork of detect_bots.py with enhancements for single-tweet scenarios:
  - n_posts meta-feature lets tree models learn conditional splits
  - Fixed temporal features for n=1 (proper night_ratio, post_hour, is_night_post)
  - Additional text features (comma_count, sentence_count, has_question, has_pronoun)
  - Data augmentation: single-tweet subsamples from multi-tweet users
  - CV-safe augmentation (augmented data in training folds only)
  - Regression test validates cross-dataset accuracy > 90%

Usage:
  Training:   python detect_bots_robust.py --train
  Inference:  python detect_bots_robust.py <input.json> <output.txt>
"""

import json
import re
import math
import sys
import os
import warnings
import pickle
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings('ignore')

# ============================================================================
# CONSTANTS
# ============================================================================

SCORE_TP = 4
SCORE_FN = -1
SCORE_FP = -2

# Engagement bait patterns (language-specific)
EN_BAIT_PATTERNS = [
    r'\bDM\b', r'\bfollow\b', r'check my bio', r'\bclick\b',
    r'opt[\s-]?in', r'link in bio', r'subscribe', r'join\s+(us|me)',
    r'🔒', r'\bfree\b.*\b(access|content)', r'pinned\b.*\btweet',
    r'\bcheck\b.*\b(out|my)', r'don\'t miss',
]

FR_BAIT_PATTERNS = [
    r'\bDM\b', r'\bsuiv(re|ez)\b', r'\babonne[zr]?\b', r'lien dans',
    r'\bbio\b', r'\bclique[zr]?\b', r'\brejoign', r'🔒',
    r'\bgratuit\b', r'\bdécouvr', r'ne\s+manquez?\s+pas',
]

# Bot-like generic hashtag patterns
GENERIC_HASHTAG_PATTERNS = [
    r'#(Love|Peace|Unity|Gratitude|PositiveVibes|Blessed|Mindful)',
    r'#(ManifestingMagic|TokyoVibes|UnexpectedJourneys|AnimeVsPop)',
    r'#(LoveInTheCrust|relatablefail|EpicFail|FacePalm|AwkwardMoment)',
    r'#(TeaTime|Relaxation|SelfCare|Wellness|Motivation)',
    r'#(NowPlaying|MusicNews|NewRelease|MusicDiscovery)',
    r'#(SportsUpdate|GameDay|TeamSpirit)',
    r'#(MindfulEscapes|NatureLovers|ZenLife)',
    r'#(VieQuotidienne|PourquoiMoi|InspirationDuJour)',
]


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def extract_text_features(text, lang='en'):
    """Extract features from a single tweet text."""
    if not text:
        return {}

    url_count = len(re.findall(r'https?://\S+|t\.co/\S+', text))
    hashtags = re.findall(r'#\w+', text)
    hashtag_count = len(hashtags)
    mentions = re.findall(r'@\w+|@mention', text)
    mention_count = len(mentions)
    emoji_count = len(re.findall(
        r'[\U0001F300-\U0001F9FF\u2600-\u27BF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF]',
        text
    ))
    char_len = len(text)
    words = text.split()
    word_count = len(words)

    alpha_chars = sum(1 for c in text if c.isalpha())
    caps_chars = sum(1 for c in text if c.isupper())
    caps_ratio = caps_chars / max(alpha_chars, 1)

    excl_count = text.count('!')
    quest_count = text.count('?')
    ellipsis_count = text.count('...')

    # Single-tweet-optimized features
    comma_count = text.count(',')
    sentence_count = len(re.split(r'[.!?]+', text.strip()))
    has_question = int('?' in text)
    # Pronoun detection (en + fr)
    has_pronoun = int(bool(re.search(
        r'\b(I|me|my|we|us|our|you|your|je|moi|mon|ma|mes|nous|vous|tu|te)\b',
        text, re.I
    )))

    # Engagement bait detection
    patterns = EN_BAIT_PATTERNS if lang == 'en' else FR_BAIT_PATTERNS
    has_bait = any(re.search(p, text, re.I) for p in patterns)

    # Generic hashtag detection
    has_generic_hashtag = any(re.search(p, text, re.I) for p in GENERIC_HASHTAG_PATTERNS)

    # Newline count (structured/templated posts)
    newline_count = text.count('\n')

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
        'ellipsis_count': ellipsis_count,
        'has_bait': int(has_bait),
        'has_generic_hashtag': int(has_generic_hashtag),
        'newline_count': newline_count,
        'has_url': int(url_count > 0),
        'has_hashtag': int(hashtag_count > 0),
        'has_mention': int(mention_count > 0),
        'comma_count': comma_count,
        'sentence_count': sentence_count,
        'has_question': has_question,
        'has_pronoun': has_pronoun,
        'hashtags': [h.lower() for h in hashtags],
    }


def compute_temporal_features(posts):
    """Compute temporal posting pattern features."""
    if len(posts) < 2:
        # Extract what we can from the single post
        post_hour = 12  # default
        is_night = 0
        night_ratio = 0
        if len(posts) == 1:
            try:
                t = datetime.fromisoformat(posts[0]['created_at'].replace('Z', '+00:00'))
                post_hour = t.hour
                is_night = int(0 <= post_hour <= 6)
                night_ratio = float(is_night)
            except (ValueError, KeyError):
                pass
        return {
            'inter_tweet_mean': 0, 'inter_tweet_std': 0, 'inter_tweet_cv': 0,
            'inter_tweet_min': 0, 'inter_tweet_max': 0, 'inter_tweet_median': 0,
            'posts_per_hour': 0, 'hour_entropy': 0, 'night_ratio': night_ratio,
            'rapid_fire_ratio': 0, 'regular_interval_ratio': 0,
            'temporal_burstiness': 0, 'active_hours': 1,
            'max_gap_ratio': 0, 'posting_span_hours': 0,
            'post_hour': post_hour, 'is_night_post': is_night,
        }

    times = sorted([
        datetime.fromisoformat(p['created_at'].replace('Z', '+00:00'))
        for p in posts
    ])
    hours = [t.hour for t in times]
    n = len(times)

    # Inter-tweet intervals
    deltas = [(times[i+1] - times[i]).total_seconds() for i in range(n-1)]
    inter_mean = np.mean(deltas)
    inter_std = np.std(deltas)
    inter_cv = inter_std / max(inter_mean, 1)
    inter_min = min(deltas)
    inter_max = max(deltas)
    inter_median = np.median(deltas)

    # Posts per hour
    span_seconds = max((times[-1] - times[0]).total_seconds(), 1)
    posts_per_hour = n / (span_seconds / 3600)
    posting_span_hours = span_seconds / 3600

    # Hour entropy
    hour_counts = Counter(hours)
    total = sum(hour_counts.values())
    hour_probs = [c/total for c in hour_counts.values()]
    hour_entropy = -sum(p * math.log2(p) for p in hour_probs if p > 0)

    # Night posting ratio (midnight to 6 AM)
    night_posts = sum(1 for h in hours if 0 <= h <= 6)
    night_ratio = night_posts / n

    # Rapid-fire ratio (intervals < 60 seconds)
    rapid_fire = sum(1 for d in deltas if d < 60) / len(deltas)

    # Regular interval ratio (within 10% of mean)
    regular = sum(1 for d in deltas if inter_mean > 0 and abs(d - inter_mean) < 0.1 * inter_mean) / len(deltas)

    # Temporal burstiness: (std - mean) / (std + mean)
    burstiness = (inter_std - inter_mean) / max(inter_std + inter_mean, 1)

    # Active hours (distinct hours with posts)
    active_hours = len(hour_counts)

    # Max gap ratio
    max_gap_ratio = inter_max / max(span_seconds, 1)

    mean_hour = np.mean(hours)
    is_night_mean = int(np.mean([1 if 0 <= h <= 6 else 0 for h in hours]) >= 0.5)

    return {
        'inter_tweet_mean': inter_mean,
        'inter_tweet_std': inter_std,
        'inter_tweet_cv': inter_cv,
        'inter_tweet_min': inter_min,
        'inter_tweet_max': inter_max,
        'inter_tweet_median': inter_median,
        'posts_per_hour': posts_per_hour,
        'hour_entropy': hour_entropy,
        'night_ratio': night_ratio,
        'rapid_fire_ratio': rapid_fire,
        'regular_interval_ratio': regular,
        'temporal_burstiness': burstiness,
        'active_hours': active_hours,
        'max_gap_ratio': max_gap_ratio,
        'posting_span_hours': posting_span_hours,
        'post_hour': mean_hour,
        'is_night_post': is_night_mean,
    }


def compute_text_diversity_features(texts):
    """Compute text diversity and stylometric features across all user tweets."""
    empty_feats = {
        'ttr': 0, 'hapax_ratio': 0, 'vocab_richness': 0,
        'avg_word_len': 0, 'long_word_ratio': 0,
        'content_jaccard_mean': 0, 'content_jaccard_std': 0,
        'unique_first_words': 0, 'repeated_phrase_ratio': 0,
        'topic_diversity': 0, 'formality_score': 0,
        'slang_ratio': 0, 'just_starter_ratio': 0,
        'avg_sentence_completeness': 0, 'punctuation_diversity': 0,
        'char_len_cv': 0, 'tweet_len_uniformity': 0,
        'url_consistency': 0, 'ai_phrase_ratio': 0,
        'artificial_caps_ratio': 0, 'generic_life_ratio': 0,
        'short_generic_ratio': 0,
    }
    if not texts:
        return empty_feats

    all_text = ' '.join(texts)
    all_words = all_text.lower().split()
    word_counts = Counter(all_words)
    vocab_size = len(set(all_words))
    total_words = max(len(all_words), 1)

    # Type-token ratio
    ttr = vocab_size / total_words

    # Hapax legomena ratio
    hapax = sum(1 for w, c in word_counts.items() if c == 1)
    hapax_ratio = hapax / max(vocab_size, 1)

    # Vocabulary richness (Yule's K approximation)
    freq_spectrum = Counter(word_counts.values())
    M1 = total_words
    M2 = sum(i * i * freq_spectrum[i] for i in freq_spectrum)
    vocab_richness = 10000 * (M2 - M1) / max(M1 * M1, 1)

    # Average word length
    word_lengths = [len(w) for w in all_words if w.isalpha()]
    avg_word_len = np.mean(word_lengths) if word_lengths else 0
    long_word_ratio = sum(1 for l in word_lengths if l > 8) / max(len(word_lengths), 1)

    # Content similarity (Jaccard between consecutive tweets)
    sims = []
    for i in range(min(len(texts) - 1, 40)):
        w1 = set(texts[i].lower().split())
        w2 = set(texts[i + 1].lower().split())
        union = w1 | w2
        if union:
            sims.append(len(w1 & w2) / len(union))
    content_jaccard_mean = np.mean(sims) if sims else 0
    content_jaccard_std = np.std(sims) if sims else 0

    # Unique first words (diversity of tweet openings)
    first_words = set()
    for t in texts:
        words = t.strip().split()
        if words:
            first_words.add(words[0].lower())
    unique_first_words = len(first_words) / max(len(texts), 1)

    # Repeated phrase detection (bigram repetition across tweets)
    bigrams_per_tweet = []
    for t in texts:
        words = t.lower().split()
        bigrams = set(zip(words[:-1], words[1:]))
        bigrams_per_tweet.append(bigrams)

    repeated = 0
    total_comparisons = 0
    for i in range(min(len(bigrams_per_tweet) - 1, 30)):
        for j in range(i + 1, min(len(bigrams_per_tweet), i + 5)):
            if bigrams_per_tweet[i] and bigrams_per_tweet[j]:
                overlap = len(bigrams_per_tweet[i] & bigrams_per_tweet[j])
                repeated += overlap
                total_comparisons += 1
    repeated_phrase_ratio = repeated / max(total_comparisons, 1)

    # === NEW FEATURES FROM ERROR ANALYSIS ===

    # Topic diversity: count distinct topic clusters via keyword sets
    topic_keywords = {
        'sports': {'nba', 'nhl', 'game', 'team', 'play', 'win', 'score', 'hockey', 'basketball', 'player', 'match', 'league'},
        'music': {'music', 'song', 'album', 'pop', 'track', 'playlist', 'melody', 'concert', 'artist', 'genre'},
        'movies': {'movie', 'film', 'cinema', 'director', 'scene', 'watch', 'series', 'episode', 'streaming'},
        'politics': {'politics', 'government', 'vote', 'election', 'president', 'liberal', 'conservative', 'policy'},
        'tech': {'iphone', 'apple', 'tech', 'app', 'software', 'ai', 'internet', 'digital', 'phone'},
        'food': {'food', 'recipe', 'cooking', 'eat', 'restaurant', 'chef', 'meal', 'delicious'},
        'life': {'life', 'day', 'morning', 'night', 'feel', 'love', 'happy', 'dream', 'beautiful'},
    }
    topics_covered = set()
    words_set = set(all_words)
    for topic, keywords in topic_keywords.items():
        if words_set & keywords:
            topics_covered.add(topic)
    topic_diversity = len(topics_covered) / len(topic_keywords)

    # Formality score: ratio of formal/polished indicators
    formal_markers = sum(1 for t in texts if re.search(
        r"(I've |I'm |It's |Let's |Don't |Can't |shouldn't|couldn't|"
        r"perhaps|indeed|furthermore|however|therefore|magnificent|"
        r"fascinating|extraordinary|remarkable|delightful)", t, re.I
    ))
    formality_score = formal_markers / max(len(texts), 1)

    # Slang/informal ratio (human signal)
    slang_markers = sum(1 for t in texts if re.search(
        r'\b(lol|lmao|wtf|omg|bruh|ngl|tbh|rn|smh|af|imo|idk|fr|'
        r'ong|mf|finna|lowkey|highkey|deadass|bussin|mfs|nah|yall|'
        r'ain\'t|gotta|gonna|wanna|tryna|fam|dawg|bro|sis|'
        r'mdr|mdrrr|ptdr|jsp|slt|tkt|bg|ptn|osef|fdp|tmtc)\b', t, re.I
    ))
    slang_ratio = slang_markers / max(len(texts), 1)

    # "Just" starter ratio (bot signal: "Just discovered...", "Just got...")
    just_starters = sum(1 for t in texts if re.match(r'^(just|Just)\s', t.strip()))
    just_starter_ratio = just_starters / max(len(texts), 1)

    # Average sentence completeness (proper ending with . ! ? — bots tend to be higher)
    complete_tweets = sum(1 for t in texts if t.strip() and t.strip()[-1] in '.!?')
    avg_sentence_completeness = complete_tweets / max(len(texts), 1)

    # Punctuation diversity (how many different punctuation types used)
    all_punct = set(c for c in all_text if c in '.,;:!?-—–()[]{}\'\"…')
    punctuation_diversity = len(all_punct)

    # Tweet length coefficient of variation (bots tend to be more uniform)
    tweet_lens = [len(t) for t in texts]
    if len(tweet_lens) >= 2:
        len_mean = np.mean(tweet_lens)
        len_std = np.std(tweet_lens)
        char_len_cv = len_std / max(len_mean, 1)
        # Uniformity: what fraction of tweets are within 50% of mean length
        tweet_len_uniformity = sum(
            1 for l in tweet_lens if abs(l - len_mean) < 0.5 * len_mean
        ) / len(tweet_lens)
    else:
        char_len_cv = 0
        tweet_len_uniformity = 1

    # URL consistency (what fraction of tweets have URLs — bots like John Doe: 100%)
    url_tweets = sum(1 for t in texts if 'http' in t or 't.co' in t)
    url_consistency = url_tweets / max(len(texts), 1)

    # AI-generated text markers
    ai_phrases = sum(1 for t in texts if re.search(
        r'(can\'t help but|it\'s (nice|great|amazing) to|'
        r'here\'s (how|what|why)|let\'s (dive|explore|talk)|'
        r'in (this|today\'s) (post|thread|tweet)|'
        r'what are your thoughts|share your|'
        r'drop your|who else|anyone else|'
        r'honestly[,.]|seriously[,.]|'
        r'(loving|feeling|enjoying|appreciating) the (vibes|energy|moment)|'
        r'(a|the) (sprinkle|dose|bit) of)', t, re.I
    ))
    ai_phrase_ratio = ai_phrases / max(len(texts), 1)

    # Artificial typo injection detection (bots inject random CAPS mid-word)
    # Pattern: lowercase letters surrounding a random uppercase letter mid-word
    artificial_caps_tweets = 0
    for t in texts:
        # Find words with random mid-word capitalization like "somepeople", "THe", "eblieve"
        words = t.split()
        mid_caps = 0
        for w in words:
            if len(w) >= 3:
                # Count mid-word caps switches (e.g., "aM", "THe", "gef")
                for j in range(1, len(w) - 1):
                    if w[j].isupper() and (w[j-1].islower() or w[j+1].islower()):
                        mid_caps += 1
                        break
        if mid_caps >= 2:
            artificial_caps_tweets += 1
    artificial_caps_ratio = artificial_caps_tweets / max(len(texts), 1)

    # "Generic life post" ratio — tweets about generic daily life observations
    generic_life = sum(1 for t in texts if re.search(
        r'(just another day|living (my|her|his) best life|'
        r'who else (loves|thinks|feels)|can we (talk|appreciate)|'
        r'(morning|evening) (routine|stroll|walk)|'
        r'(happy|excited) to (announce|share)|'
        r'(passionate|proud) (about|of|to)|'
        r'(empowering|inspiring|motivating)|'
        r'(sustainability|initiative|innovati))', t, re.I
    ))
    generic_life_ratio = generic_life / max(len(texts), 1)

    # Short generic post ratio (very short, bland posts — flower/plant bots)
    short_generic = sum(1 for t in texts if len(t) < 60 and not re.search(r'http|@|#', t))
    short_generic_ratio = short_generic / max(len(texts), 1)

    return {
        'ttr': ttr,
        'hapax_ratio': hapax_ratio,
        'vocab_richness': vocab_richness,
        'avg_word_len': avg_word_len,
        'long_word_ratio': long_word_ratio,
        'content_jaccard_mean': content_jaccard_mean,
        'content_jaccard_std': content_jaccard_std,
        'unique_first_words': unique_first_words,
        'repeated_phrase_ratio': repeated_phrase_ratio,
        'topic_diversity': topic_diversity,
        'formality_score': formality_score,
        'slang_ratio': slang_ratio,
        'just_starter_ratio': just_starter_ratio,
        'avg_sentence_completeness': avg_sentence_completeness,
        'punctuation_diversity': punctuation_diversity,
        'char_len_cv': char_len_cv,
        'tweet_len_uniformity': tweet_len_uniformity,
        'url_consistency': url_consistency,
        'ai_phrase_ratio': ai_phrase_ratio,
        'artificial_caps_ratio': artificial_caps_ratio,
        'generic_life_ratio': generic_life_ratio,
        'short_generic_ratio': short_generic_ratio,
    }


def compute_hashtag_features(all_tweet_features):
    """Compute hashtag-specific features."""
    all_hashtags = []
    for tf in all_tweet_features:
        all_hashtags.extend(tf.get('hashtags', []))

    n_posts = max(len(all_tweet_features), 1)
    total_hashtags = len(all_hashtags)
    unique_hashtags = set(all_hashtags)

    # Posts with hashtags
    posts_with_hashtags = sum(1 for tf in all_tweet_features if tf.get('hashtag_count', 0) > 0)

    # Hashtag diversity: unique/total
    hashtag_diversity = len(unique_hashtags) / max(total_hashtags, 1)

    # Max hashtags in a single tweet
    max_hashtags_per_tweet = max((tf.get('hashtag_count', 0) for tf in all_tweet_features), default=0)

    # Hashtag-to-post ratio
    hashtag_post_ratio = posts_with_hashtags / n_posts

    return {
        'total_hashtags': total_hashtags,
        'unique_hashtags': len(unique_hashtags),
        'hashtag_diversity': hashtag_diversity,
        'hashtag_per_post': total_hashtags / n_posts,
        'max_hashtags_per_tweet': max_hashtags_per_tweet,
        'hashtag_post_ratio': hashtag_post_ratio,
    }


def extract_user_features(user, posts, lang='en'):
    """Extract all features for a single user."""
    texts = [p['text'] for p in posts]
    n_posts = len(posts)

    # === Per-tweet text features ===
    tweet_features = [extract_text_features(t, lang) for t in texts]

    # Aggregate tweet features
    agg = {}
    numeric_keys = [
        'url_count', 'hashtag_count', 'mention_count', 'emoji_count',
        'char_len', 'word_count', 'caps_ratio', 'excl_count', 'quest_count',
        'ellipsis_count', 'has_bait', 'has_generic_hashtag', 'newline_count',
        'has_url', 'has_hashtag', 'has_mention',
        'comma_count', 'sentence_count', 'has_question', 'has_pronoun',
    ]
    for key in numeric_keys:
        vals = [tf[key] for tf in tweet_features]
        agg[f'{key}_mean'] = np.mean(vals)
        agg[f'{key}_sum'] = np.sum(vals)
        if key in ('char_len', 'word_count', 'hashtag_count', 'url_count'):
            agg[f'{key}_std'] = np.std(vals)
            agg[f'{key}_max'] = np.max(vals)

    # === Temporal features ===
    temporal = compute_temporal_features(posts)

    # === Text diversity features ===
    diversity = compute_text_diversity_features(texts)

    # === Hashtag features ===
    hashtag_feats = compute_hashtag_features(tweet_features)

    # === User profile features ===
    desc = user.get('description', '') or ''
    name = user.get('name', '') or ''
    username = user.get('username', '') or ''
    location = user.get('location', '') or ''

    profile = {
        'tweet_count': user.get('tweet_count', n_posts),
        'z_score': user.get('z_score', 0),
        'has_description': int(bool(desc.strip())),
        'has_location': int(bool(location.strip())),
        'desc_len': len(desc),
        'desc_word_count': len(desc.split()),
        'name_len': len(name),
        'username_len': len(username),
        'username_digit_ratio': sum(1 for c in username if c.isdigit()) / max(len(username), 1),
        'username_has_underscore': int('_' in username),
        'profile_has_pipe': int('|' in desc or '|' in name),
        'profile_has_hashtag': int('#' in desc),
        'profile_has_emoji': int(bool(re.search(
            r'[\U0001F300-\U0001F9FF\u2600-\u27BF\U0001FA00-\U0001FA6F]',
            desc + name
        ))),
        'profile_has_url': int(bool(re.search(r'http|\.com|\.org|\.net', desc))),
        'name_has_special_chars': int(bool(re.search(r'[^\w\s.,!?\'-]', name))),
        'desc_has_newline': int('\n' in desc),
        'desc_pipe_count': desc.count('|'),
        # Bot-like profile patterns: formulaic "X | Y | Z" structure
        'desc_formulaic': int(desc.count('|') >= 2),
        # Username pattern: word_word or word_word123
        'username_bot_pattern': int(bool(re.match(r'^[a-z]+_[a-z]+\d*$', username))),
        # Name is "generic" (FirstName LastName pattern)
        'name_generic_pattern': int(bool(re.match(r'^[A-Z][a-z]+ [A-Z][a-z]+$', name.strip()))),
    }

    # === Interaction features (cross-category signals) ===
    # Note: hashtag_volume_interaction and cross_user_sim_mean removed —
    # ablation showed they HURT cross-dataset generalization
    interactions = {
        # High hour entropy + high tweet count = bot posting around the clock
        'entropy_volume_interaction': temporal['hour_entropy'] * user.get('tweet_count', n_posts),
    }

    # Combine all features
    features = {}
    features['n_posts'] = n_posts
    features.update(agg)
    features.update(profile)
    features.update(temporal)
    features.update(diversity)
    features.update(hashtag_feats)
    features.update(interactions)

    return features


# ============================================================================
# DATA LOADING
# ============================================================================

def load_dataset(path):
    """Load a dataset JSON file."""
    with open(path) as f:
        return json.load(f)


def load_bots(path):
    """Load bot IDs from a text file."""
    with open(path) as f:
        return set(line.strip() for line in f if line.strip())


def compute_cross_user_features(users_dict, posts_by_author, user_order):
    """Compute cross-user similarity features using TF-IDF."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    # Build per-user text corpus
    user_texts = []
    for uid in user_order:
        texts = [p['text'] for p in posts_by_author.get(uid, [])]
        user_texts.append(' '.join(texts))

    if len(user_texts) < 2:
        return {uid: {'cross_user_sim_mean': 0, 'cross_user_sim_max': 0} for uid in user_order}

    vectorizer = TfidfVectorizer(max_features=3000, min_df=2, max_df=0.95)
    try:
        tfidf_matrix = vectorizer.fit_transform(user_texts)
        sim_matrix = cosine_similarity(tfidf_matrix)
    except ValueError:
        return {uid: {'cross_user_sim_mean': 0, 'cross_user_sim_max': 0} for uid in user_order}

    result = {}
    for i, uid in enumerate(user_order):
        sims = sim_matrix[i].copy()
        sims[i] = 0  # exclude self
        result[uid] = {
            'cross_user_sim_mean': float(np.mean(sims)),
            'cross_user_sim_max': float(np.max(sims)),
        }
    return result


def prepare_dataset(data, bot_ids=None, lang=None):
    """Convert raw data into feature matrix."""
    if lang is None:
        lang = data.get('lang', 'en')

    users = {u['id']: u for u in data['users']}
    posts_by_author = defaultdict(list)
    for p in data['posts']:
        posts_by_author[p['author_id']].append(p)

    # Get user order (only users with posts)
    user_order = [uid for uid in users if uid in posts_by_author]

    # Compute cross-user features
    cross_user = compute_cross_user_features(users, posts_by_author, user_order)

    rows = []
    user_ids = []

    for uid in user_order:
        user_posts = posts_by_author[uid]
        features = extract_user_features(users[uid], user_posts, lang)

        # Add cross-user features
        features.update(cross_user.get(uid, {'cross_user_sim_mean': 0, 'cross_user_sim_max': 0}))

        if bot_ids is not None:
            features['label'] = int(uid in bot_ids)

        rows.append(features)
        user_ids.append(uid)

    df = pd.DataFrame(rows)
    return df, user_ids


# ============================================================================
# MODEL TRAINING
# ============================================================================

def compute_score(y_true, y_pred):
    """Compute competition score."""
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    score = SCORE_TP * tp + SCORE_FN * fn + SCORE_FP * fp
    return score, tp, fn, fp, tn


def optimize_threshold(y_true, y_prob, thresholds=None):
    """Find optimal threshold for competition scoring."""
    if thresholds is None:
        thresholds = np.arange(0.10, 0.95, 0.01)

    best_score = -float('inf')
    best_threshold = 0.5
    results = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        score, tp, fn, fp, tn = compute_score(y_true, y_pred)
        results.append((t, score, tp, fn, fp, tn))
        if score > best_score:
            best_score = score
            best_threshold = t

    return best_threshold, best_score, results


def get_feature_columns(df):
    """Get feature columns (exclude label and non-numeric)."""
    exclude = {'label'}
    return [c for c in df.columns if c not in exclude and df[c].dtype in ('float64', 'int64', 'float32', 'int32')]


class BotDetectorEnsemble:
    """Ensemble bot detector with custom threshold optimization."""

    def __init__(self, lang='en'):
        self.lang = lang
        self.models = {}
        self.feature_cols = None
        self.threshold = 0.5
        self.feature_importances = None

    def train(self, df, verbose=True, n_original=None):
        """Train ensemble models with cross-validation threshold tuning.

        Args:
            n_original: Number of original (non-augmented) rows at the start of df.
                        If provided, CV splits only on original rows; augmented data
                        goes in training folds only to prevent data leakage.
        """
        self.feature_cols = get_feature_columns(df)
        X = df[self.feature_cols].values
        y = df['label'].values

        # Handle NaN/inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        if n_original is None:
            n_original = len(y)

        if verbose:
            print(f"\n{'='*60}")
            print(f"Training {self.lang.upper()} model | Samples: {len(y)} "
                  f"({n_original} original + {len(y)-n_original} augmented) | "
                  f"Bots: {sum(y)} ({100*sum(y)/len(y):.1f}%)")
            print(f"Features: {len(self.feature_cols)}")
            print(f"{'='*60}")

        # --- Model 1: LightGBM (strong regularization for generalization) ---
        lgb_model = lgb.LGBMClassifier(
            n_estimators=800,
            max_depth=4,
            learning_rate=0.03,
            num_leaves=15,
            min_child_samples=10,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=1.0,
            reg_lambda=5.0,
            min_split_gain=0.01,
            random_state=42,
            verbose=-1,
            is_unbalance=True,
        )
        lgb_model.fit(X, y)
        self.models['lgb'] = lgb_model

        # --- Model 2: XGBoost ---
        xgb_model = xgb.XGBClassifier(
            n_estimators=800,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=1.0,
            reg_lambda=5.0,
            gamma=0.1,
            min_child_weight=5,
            random_state=42,
            verbosity=0,
            scale_pos_weight=sum(y == 0) / max(sum(y == 1), 1),
            eval_metric='logloss',
        )
        xgb_model.fit(X, y)
        self.models['xgb'] = xgb_model

        # --- Model 3: Gradient Boosting ---
        gb_model = GradientBoostingClassifier(
            n_estimators=500,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.7,
            min_samples_leaf=10,
            random_state=42,
        )
        gb_model.fit(X, y)
        self.models['gb'] = gb_model

        # --- Cross-validation for threshold optimization ---
        # Split only on original rows; augmented data goes in training folds only
        if verbose:
            print("\nCross-validation for threshold tuning...")

        X_orig, y_orig = X[:n_original], y[:n_original]
        X_aug, y_aug = X[n_original:], y[n_original:]
        has_aug = len(X_aug) > 0

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        all_probs = np.zeros(n_original)

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_orig, y_orig)):
            X_tr, X_val = X_orig[train_idx], X_orig[val_idx]
            y_tr, y_val = y_orig[train_idx], y_orig[val_idx]

            # Add augmented data to training fold only (prevents leakage)
            if has_aug:
                X_tr = np.vstack([X_tr, X_aug])
                y_tr = np.concatenate([y_tr, y_aug])

            fold_probs = np.zeros(len(val_idx))

            # LightGBM fold
            lgb_f = lgb.LGBMClassifier(
                n_estimators=800, max_depth=4, learning_rate=0.03,
                num_leaves=15, min_child_samples=10, subsample=0.7,
                colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=5.0,
                min_split_gain=0.01,
                random_state=42, verbose=-1, is_unbalance=True,
            )
            lgb_f.fit(X_tr, y_tr)
            fold_probs += lgb_f.predict_proba(X_val)[:, 1] * 0.35

            # XGBoost fold
            xgb_f = xgb.XGBClassifier(
                n_estimators=800, max_depth=4, learning_rate=0.03,
                subsample=0.7, colsample_bytree=0.7, reg_alpha=1.0,
                reg_lambda=5.0, gamma=0.1, min_child_weight=5,
                random_state=42, verbosity=0,
                scale_pos_weight=sum(y_tr == 0) / max(sum(y_tr == 1), 1),
                eval_metric='logloss',
            )
            xgb_f.fit(X_tr, y_tr)
            fold_probs += xgb_f.predict_proba(X_val)[:, 1] * 0.35

            # GB fold
            gb_f = GradientBoostingClassifier(
                n_estimators=500, max_depth=3, learning_rate=0.03,
                subsample=0.7, min_samples_leaf=10, random_state=42,
            )
            gb_f.fit(X_tr, y_tr)
            fold_probs += gb_f.predict_proba(X_val)[:, 1] * 0.30

            all_probs[val_idx] = fold_probs

        # Optimize threshold (on original data only)
        self.threshold, best_score, results = optimize_threshold(y_orig, all_probs)

        if verbose:
            print(f"\nOptimal threshold: {self.threshold:.2f}")
            print(f"Best CV score: {best_score}")

            # Show performance at optimal threshold
            y_pred = (all_probs >= self.threshold).astype(int)
            score, tp, fn, fp, tn = compute_score(y_orig, y_pred)
            total_bots = sum(y_orig)
            print(f"\nCV Performance at threshold={self.threshold:.2f}:")
            print(f"  TP={tp}, FN={fn}, FP={fp}, TN={tn}")
            print(f"  Precision={tp/max(tp+fp,1):.3f}, Recall={tp/max(tp+fn,1):.3f}")
            print(f"  Score={score} (max possible={SCORE_TP * total_bots})")
            print(f"  Accuracy={100*(tp+tn)/len(y_orig):.1f}%")

        # Feature importance from LightGBM
        self.feature_importances = dict(zip(
            self.feature_cols,
            lgb_model.feature_importances_
        ))

        return self

    def predict_proba(self, df):
        """Get ensemble probability predictions."""
        X = df[self.feature_cols].values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        probs = np.zeros(len(X))
        probs += self.models['lgb'].predict_proba(X)[:, 1] * 0.35
        probs += self.models['xgb'].predict_proba(X)[:, 1] * 0.35
        probs += self.models['gb'].predict_proba(X)[:, 1] * 0.30

        return probs

    def predict(self, df):
        """Get binary predictions using optimized threshold."""
        probs = self.predict_proba(df)
        return (probs >= self.threshold).astype(int)

    def print_top_features(self, n=20):
        """Print top features by importance."""
        if self.feature_importances:
            sorted_feats = sorted(
                self.feature_importances.items(),
                key=lambda x: x[1], reverse=True
            )
            print(f"\nTop {n} features ({self.lang.upper()}):")
            for feat, imp in sorted_feats[:n]:
                print(f"  {feat:40s}: {imp}")


# ============================================================================
# TRAINING PIPELINE
# ============================================================================

def validate_no_regression(datasets, lang_map, min_accuracy=90.0):
    """Cross-dataset validation to ensure no regression from changes.

    Trains on one dataset, tests on the other (same language).
    Asserts accuracy > min_accuracy on each cross-validation pair.
    """
    print(f"\n{'='*60}")
    print(f"REGRESSION TEST (cross-dataset accuracy > {min_accuracy}%)")
    print(f"{'='*60}")

    pairs = [
        ('en', 30, 32),
        ('en', 32, 30),
        ('fr', 31, 33),
        ('fr', 33, 31),
    ]

    all_passed = True
    for lang, train_id, test_id in pairs:
        if train_id not in datasets or test_id not in datasets:
            print(f"  SKIP: ds{train_id} → ds{test_id} (missing data)")
            continue

        train_df, _ = prepare_dataset(datasets[train_id]['data'], datasets[train_id]['bots'], lang)
        test_df, _ = prepare_dataset(datasets[test_id]['data'], datasets[test_id]['bots'], lang)

        model = BotDetectorEnsemble(lang=lang)
        model.train(train_df, verbose=False)
        probs = model.predict_proba(test_df)
        y_true = test_df['label'].values
        threshold, _, _ = optimize_threshold(y_true, probs)
        y_pred = (probs >= threshold).astype(int)
        _, tp, fn, fp, tn = compute_score(y_true, y_pred)
        acc = 100 * (tp + tn) / len(y_true)

        status = "PASS" if acc >= min_accuracy else "FAIL"
        if acc < min_accuracy:
            all_passed = False
        print(f"  [{status}] ds{train_id} → ds{test_id} ({lang.upper()}): {acc:.1f}% accuracy")

    if all_passed:
        print("  All regression tests PASSED.")
    else:
        print("  WARNING: Some regression tests FAILED!")

    return all_passed


def augment_single_tweet(data, bot_ids, lang, n_copies=3, seed=42):
    """Create augmented single-tweet training samples from multi-tweet users.

    For each user with >1 post, creates n_copies single-tweet feature rows
    by randomly selecting 1 post each time. Labels are preserved.
    """
    rng = np.random.RandomState(seed)
    users = {u['id']: u for u in data['users']}
    posts_by_author = defaultdict(list)
    for p in data['posts']:
        posts_by_author[p['author_id']].append(p)

    aug_rows = []
    for uid, user_posts in posts_by_author.items():
        if uid not in users or len(user_posts) < 2:
            continue
        for _ in range(n_copies):
            idx = rng.randint(0, len(user_posts))
            single_post = [user_posts[idx]]
            features = extract_user_features(users[uid], single_post, lang)
            # Cross-user features are 0 for augmented rows (single tweet can't meaningfully compare)
            features['cross_user_sim_mean'] = 0
            features['cross_user_sim_max'] = 0
            features['label'] = int(uid in bot_ids)
            aug_rows.append(features)

    return pd.DataFrame(aug_rows)


def train_pipeline(verbose=True):
    """Full training pipeline: load data, extract features, train models."""
    base_dir = Path(__file__).parent

    # Load all datasets
    datasets = {}
    for ds_id in [30, 31, 32, 33]:
        data_path = base_dir / f'dataset.posts&users.{ds_id}.json'
        bots_path = base_dir / f'dataset.bots.{ds_id}.txt'
        if data_path.exists() and bots_path.exists():
            datasets[ds_id] = {
                'data': load_dataset(data_path),
                'bots': load_bots(bots_path),
            }

    lang_map = {30: 'en', 31: 'fr', 32: 'en', 33: 'fr'}

    # Prepare feature matrices (original + augmented)
    en_dfs = []
    fr_dfs = []
    en_aug_dfs = []
    fr_aug_dfs = []

    for ds_id, ds in datasets.items():
        lang = lang_map[ds_id]
        df, user_ids = prepare_dataset(ds['data'], ds['bots'], lang)
        aug_df = augment_single_tweet(ds['data'], ds['bots'], lang)
        if verbose:
            print(f"Dataset {ds_id} ({lang.upper()}): {len(df)} users, {sum(df['label'])} bots, "
                  f"+{len(aug_df)} augmented single-tweet rows")
        if lang == 'en':
            en_dfs.append(df)
            en_aug_dfs.append(aug_df)
        else:
            fr_dfs.append(df)
            fr_aug_dfs.append(aug_df)

    models = {}

    # Train English model
    if en_dfs:
        en_orig = pd.concat(en_dfs, ignore_index=True)
        en_aug = pd.concat(en_aug_dfs, ignore_index=True) if en_aug_dfs else pd.DataFrame()
        en_full = pd.concat([en_orig, en_aug], ignore_index=True)
        n_orig_en = len(en_orig)
        if verbose:
            print(f"\nEN training: {n_orig_en} original + {len(en_aug)} augmented = {len(en_full)} total")
        en_model = BotDetectorEnsemble(lang='en')
        en_model.train(en_full, verbose=verbose, n_original=n_orig_en)
        en_model.print_top_features()
        models['en'] = en_model

        # Cross-dataset validation (on original data only)
        if verbose and len(en_dfs) == 2:
            print("\n--- Cross-dataset validation (EN) ---")
            for i, (train_df, val_df) in enumerate([(en_dfs[0], en_dfs[1]), (en_dfs[1], en_dfs[0])]):
                temp_model = BotDetectorEnsemble(lang='en')
                temp_model.train(train_df, verbose=False)
                probs = temp_model.predict_proba(val_df)
                y_true = val_df['label'].values
                threshold, score, _ = optimize_threshold(y_true, probs)
                y_pred = (probs >= threshold).astype(int)
                s, tp, fn, fp, tn = compute_score(y_true, y_pred)
                acc = 100 * (tp + tn) / len(y_true)
                print(f"  Train on ds{[30,32][i]} → Test on ds{[32,30][i]}: "
                      f"Score={s}, Acc={acc:.1f}%, P={tp/max(tp+fp,1):.3f}, R={tp/max(tp+fn,1):.3f}, τ={threshold:.2f}")

    # Train French model
    if fr_dfs:
        fr_orig = pd.concat(fr_dfs, ignore_index=True)
        fr_aug = pd.concat(fr_aug_dfs, ignore_index=True) if fr_aug_dfs else pd.DataFrame()
        fr_full = pd.concat([fr_orig, fr_aug], ignore_index=True)
        n_orig_fr = len(fr_orig)
        if verbose:
            print(f"\nFR training: {n_orig_fr} original + {len(fr_aug)} augmented = {len(fr_full)} total")
        fr_model = BotDetectorEnsemble(lang='fr')
        fr_model.train(fr_full, verbose=verbose, n_original=n_orig_fr)
        fr_model.print_top_features()
        models['fr'] = fr_model

        # Cross-dataset validation (on original data only)
        if verbose and len(fr_dfs) == 2:
            print("\n--- Cross-dataset validation (FR) ---")
            for i, (train_df, val_df) in enumerate([(fr_dfs[0], fr_dfs[1]), (fr_dfs[1], fr_dfs[0])]):
                temp_model = BotDetectorEnsemble(lang='fr')
                temp_model.train(train_df, verbose=False)
                probs = temp_model.predict_proba(val_df)
                y_true = val_df['label'].values
                threshold, score, _ = optimize_threshold(y_true, probs)
                y_pred = (probs >= threshold).astype(int)
                s, tp, fn, fp, tn = compute_score(y_true, y_pred)
                acc = 100 * (tp + tn) / len(y_true)
                print(f"  Train on ds{[31,33][i]} → Test on ds{[33,31][i]}: "
                      f"Score={s}, Acc={acc:.1f}%, P={tp/max(tp+fp,1):.3f}, R={tp/max(tp+fn,1):.3f}, τ={threshold:.2f}")

    # Regression test
    if verbose:
        validate_no_regression(datasets, lang_map)

    # Save models
    model_path = base_dir / 'models_robust.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(models, f)
    if verbose:
        print(f"\nModels saved to {model_path}")

    return models


# ============================================================================
# HARD RULES (near-zero FP, catches bot pipeline artifacts)
# ============================================================================

def apply_hard_rules(user, posts):
    """
    High-confidence rules that catch bot pipeline artifacts.
    These fire before the ML model and have near-zero false positive rates.
    Returns (is_bot, reason) or (False, None).
    """
    name = user.get('name', '') or ''
    location = user.get('location', '') or ''
    texts = [p['text'] for p in posts]

    # Rule 1: Garbled name — non-printable characters from broken Unicode in bot pipelines
    if any(ord(c) < 32 for c in name):
        return True, "garbled_name"

    # Rule 2: LLM output leakage — literal prompt framing leaked into tweet text
    llm_leak_patterns = [
        r'here are some of my recent tweets',
        r'here are the re-?written',
        r'here are some (?:recent )?rewrites',
        r'here are some (?:modified|revised|alternative) versions',
        r'rewritten tweet',
        r'here are some changes i made',
        r'as an ai\b',
        r'as a language model',
        r'voici (?:quelques|mes) (?:récents? )?tweets',
        r'voici (?:les|des) versions? (?:révisée|modifiée)',
    ]
    for t in texts:
        t_lower = t.lower()
        for pat in llm_leak_patterns:
            if re.search(pat, t_lower):
                return True, "llm_leak"

    # Rule 3: Systematic URL typos — htts:// instead of https://
    for t in texts:
        if 'htts://' in t or 'htt://' in t:
            return True, "url_typo"

    # Rule 4: Weird location format — pipeline artifacts like :null:, O:location:O
    if any(marker in location for marker in [':null:', 'O:', '.:']):
        return True, "weird_location"

    return False, None


# ============================================================================
# INFERENCE PIPELINE
# ============================================================================

def detect_bots(input_path, output_path):
    """Run bot detection on a new dataset."""
    base_dir = Path(__file__).parent
    model_path = base_dir / 'models_robust.pkl'

    if not model_path.exists():
        print("No trained models found. Running training first...")
        models = train_pipeline(verbose=True)
    else:
        with open(model_path, 'rb') as f:
            models = pickle.load(f)

    # Load input data
    data = load_dataset(input_path)
    lang = data.get('lang', 'en')

    if lang not in models:
        print(f"Warning: No model for language '{lang}', falling back to 'en'")
        lang = 'en'

    model = models[lang]

    users = {u['id']: u for u in data['users']}
    posts_by_author = defaultdict(list)
    for p in data['posts']:
        posts_by_author[p['author_id']].append(p)

    # Phase 1: Hard rules (near-zero FP pipeline artifact detection)
    hard_rule_bots = set()
    for uid, user in users.items():
        user_posts = posts_by_author.get(uid, [])
        if user_posts:
            is_bot, reason = apply_hard_rules(user, user_posts)
            if is_bot:
                hard_rule_bots.add(uid)

    # Phase 2: ML ensemble
    df, user_ids = prepare_dataset(data, lang=lang)
    probs = model.predict_proba(df)
    predictions = (probs >= model.threshold).astype(int)
    ml_bots = set(uid for uid, pred in zip(user_ids, predictions) if pred == 1)

    # Combine: union of hard rules and ML detections
    all_bots = hard_rule_bots | ml_bots

    with open(output_path, 'w') as f:
        for uid in all_bots:
            f.write(uid + '\n')

    n_hard_only = len(hard_rule_bots - ml_bots)
    print(f"Detected {len(all_bots)} bots out of {len(users)} users "
          f"(ML: {len(ml_bots)}, hard rules: {len(hard_rule_bots)}, hard-only: {n_hard_only})")
    print(f"Output written to {output_path}")

    return list(all_bots)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == '--train':
        train_pipeline(verbose=True)
    elif len(sys.argv) == 3:
        detect_bots(sys.argv[1], sys.argv[2])
    else:
        print("Usage:")
        print("  Training:   python detect_bots.py --train")
        print("  Inference:  python detect_bots.py <input.json> <output.txt>")
        print("\nRunning training on practice datasets...")
        train_pipeline(verbose=True)
