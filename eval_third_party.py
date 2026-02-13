#!/usr/bin/env python3
"""Evaluate our bot detector on the third-party CSV dataset."""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from collections import defaultdict

# Import our feature extraction
from detect_bots import (
    extract_user_features, compute_cross_user_features,
    get_feature_columns, compute_score, optimize_threshold,
    BotDetectorEnsemble
)

def main():
    base_dir = Path(__file__).parent

    # Load CSV
    print("Loading CSV...")
    csv_df = pd.read_csv(base_dir / 'bot_detection_data.csv')
    print(f"  {len(csv_df)} rows, {csv_df['User ID'].nunique()} unique users")
    print(f"  Bot distribution: {csv_df['Bot Label'].value_counts().to_dict()}")

    # Convert CSV rows to the format our model expects
    # Each row = 1 user with 1 tweet
    print("\nConverting to model format and extracting features...")

    users_dict = {}
    posts_by_author = defaultdict(list)

    for _, row in csv_df.iterrows():
        uid = str(row['User ID'])

        # Build user dict (many fields missing from CSV, use defaults)
        if uid not in users_dict:
            users_dict[uid] = {
                'id': uid,
                'name': row['Username'],  # best we have
                'username': row['Username'],
                'description': '',  # not in CSV
                'location': row['Location'] if pd.notna(row['Location']) else '',
                'tweet_count': 1,
                'z_score': 0,
            }

        # Build post dict
        # Prepend hashtags to text if present (since CSV separates them)
        tweet_text = row['Tweet']
        if pd.notna(row['Hashtags']) and row['Hashtags']:
            hashtag_str = ' '.join(f'#{h}' for h in str(row['Hashtags']).split())
            tweet_text = tweet_text + ' ' + hashtag_str

        posts_by_author[uid].append({
            'author_id': uid,
            'text': tweet_text,
            'created_at': row['Created At'].replace(' ', 'T') + '+00:00',
        })

    user_order = list(users_dict.keys())
    labels = {}
    for _, row in csv_df.iterrows():
        labels[str(row['User ID'])] = int(row['Bot Label'])

    # Extract features for all users
    print(f"Extracting features for {len(user_order)} users...")

    # Process in batches to show progress
    batch_size = 5000
    all_rows = []
    all_uids = []

    for i in range(0, len(user_order), batch_size):
        batch_uids = user_order[i:i+batch_size]
        batch_users = {uid: users_dict[uid] for uid in batch_uids}
        batch_posts = {uid: posts_by_author[uid] for uid in batch_uids}

        # Cross-user features for this batch
        cross_user = compute_cross_user_features(batch_users, batch_posts, batch_uids)

        for uid in batch_uids:
            features = extract_user_features(users_dict[uid], posts_by_author[uid], lang='en')
            features.update(cross_user.get(uid, {'cross_user_sim_mean': 0, 'cross_user_sim_max': 0}))
            features['label'] = labels[uid]
            all_rows.append(features)
            all_uids.append(uid)

        print(f"  Processed {min(i+batch_size, len(user_order))}/{len(user_order)} users")

    df = pd.DataFrame(all_rows)

    # Load trained model
    print("\nLoading trained model...")
    model_path = base_dir / 'models.pkl'
    with open(model_path, 'rb') as f:
        models = pickle.load(f)

    model = models['en']
    print(f"  Model threshold: {model.threshold:.2f}")
    print(f"  Model features: {len(model.feature_cols)}")

    # Check feature alignment
    missing_feats = [f for f in model.feature_cols if f not in df.columns]
    extra_feats = [f for f in get_feature_columns(df) if f not in model.feature_cols]
    if missing_feats:
        print(f"  WARNING: Missing features: {missing_feats}")
        for f in missing_feats:
            df[f] = 0
    if extra_feats:
        print(f"  Extra features (ignored): {extra_feats[:5]}...")

    # Run predictions
    y_true = df['label'].values
    probs = model.predict_proba(df)

    # Evaluate at model's trained threshold
    y_pred = (probs >= model.threshold).astype(int)
    score, tp, fn, fp, tn = compute_score(y_true, y_pred)

    print(f"\n{'='*60}")
    print(f"Results at trained threshold ({model.threshold:.2f})")
    print(f"{'='*60}")
    print(f"  TP={tp}, FN={fn}, FP={fp}, TN={tn}")
    print(f"  Accuracy:  {100*(tp+tn)/len(y_true):.2f}%")
    print(f"  Precision: {tp/max(tp+fp,1):.4f}")
    print(f"  Recall:    {tp/max(tp+fn,1):.4f}")
    print(f"  F1:        {2*tp/max(2*tp+fp+fn,1):.4f}")
    print(f"  Custom Score: {score} (max possible: {4*sum(y_true)})")

    # Also find optimal threshold on this dataset (just to see ceiling)
    best_thresh, best_score, _ = optimize_threshold(y_true, probs)
    y_pred_opt = (probs >= best_thresh).astype(int)
    s_opt, tp_o, fn_o, fp_o, tn_o = compute_score(y_true, y_pred_opt)

    print(f"\n{'='*60}")
    print(f"Results at optimal threshold ({best_thresh:.2f}) — ceiling on this data")
    print(f"{'='*60}")
    print(f"  TP={tp_o}, FN={fn_o}, FP={fp_o}, TN={tn_o}")
    print(f"  Accuracy:  {100*(tp_o+tn_o)/len(y_true):.2f}%")
    print(f"  Precision: {tp_o/max(tp_o+fp_o,1):.4f}")
    print(f"  Recall:    {tp_o/max(tp_o+fn_o,1):.4f}")
    print(f"  F1:        {2*tp_o/max(2*tp_o+fp_o+fn_o,1):.4f}")
    print(f"  Custom Score: {s_opt}")

    # Probability distribution analysis
    print(f"\n{'='*60}")
    print(f"Probability distribution")
    print(f"{'='*60}")
    bot_probs = probs[y_true == 1]
    human_probs = probs[y_true == 0]
    print(f"  Actual bots   — mean: {bot_probs.mean():.4f}, median: {np.median(bot_probs):.4f}, "
          f">0.5: {(bot_probs>=0.5).sum()}/{len(bot_probs)}")
    print(f"  Actual humans — mean: {human_probs.mean():.4f}, median: {np.median(human_probs):.4f}, "
          f">0.5: {(human_probs>=0.5).sum()}/{len(human_probs)}")

    # Note about data mismatch
    print(f"\n{'='*60}")
    print("NOTE: Our model was trained on multi-tweet competition data (10-50 tweets/user).")
    print("This dataset has 1 tweet per user, so temporal and diversity features are degenerate.")
    print(f"{'='*60}")

    # === Evaluate robust model ===
    robust_path = base_dir / 'models_robust.pkl'
    if robust_path.exists():
        print(f"\n\n{'#'*60}")
        print("ROBUST MODEL EVALUATION")
        print(f"{'#'*60}")

        with open(robust_path, 'rb') as f:
            robust_models = pickle.load(f)

        # Re-extract features using robust feature extraction
        try:
            from detect_bots_robust import (
                extract_user_features as extract_user_features_robust,
                compute_cross_user_features as compute_cross_user_features_robust,
                get_feature_columns as get_feature_columns_robust,
                compute_score as compute_score_robust,
                optimize_threshold as optimize_threshold_robust,
            )

            print("\nRe-extracting features with robust feature set...")
            robust_rows = []
            for i in range(0, len(user_order), batch_size):
                batch_uids = user_order[i:i+batch_size]
                batch_users = {uid: users_dict[uid] for uid in batch_uids}
                batch_posts = {uid: posts_by_author[uid] for uid in batch_uids}
                cross_user_r = compute_cross_user_features_robust(batch_users, batch_posts, batch_uids)
                for uid in batch_uids:
                    features = extract_user_features_robust(users_dict[uid], posts_by_author[uid], lang='en')
                    features.update(cross_user_r.get(uid, {'cross_user_sim_mean': 0, 'cross_user_sim_max': 0}))
                    features['label'] = labels[uid]
                    robust_rows.append(features)
                print(f"  Processed {min(i+batch_size, len(user_order))}/{len(user_order)} users")

            df_robust = pd.DataFrame(robust_rows)

            robust_model = robust_models['en']
            print(f"  Model threshold: {robust_model.threshold:.2f}")
            print(f"  Model features: {len(robust_model.feature_cols)}")

            # Check feature alignment
            missing_r = [f for f in robust_model.feature_cols if f not in df_robust.columns]
            if missing_r:
                print(f"  WARNING: Missing features: {missing_r}")
                for f in missing_r:
                    df_robust[f] = 0

            y_true_r = df_robust['label'].values
            probs_r = robust_model.predict_proba(df_robust)

            y_pred_r = (probs_r >= robust_model.threshold).astype(int)
            score_r, tp_r, fn_r, fp_r, tn_r = compute_score_robust(y_true_r, y_pred_r)

            print(f"\n{'='*60}")
            print(f"Robust model at trained threshold ({robust_model.threshold:.2f})")
            print(f"{'='*60}")
            print(f"  TP={tp_r}, FN={fn_r}, FP={fp_r}, TN={tn_r}")
            print(f"  Accuracy:  {100*(tp_r+tn_r)/len(y_true_r):.2f}%")
            print(f"  Precision: {tp_r/max(tp_r+fp_r,1):.4f}")
            print(f"  Recall:    {tp_r/max(tp_r+fn_r,1):.4f}")
            print(f"  F1:        {2*tp_r/max(2*tp_r+fp_r+fn_r,1):.4f}")
            print(f"  Custom Score: {score_r}")

            best_thresh_r, best_score_r, _ = optimize_threshold_robust(y_true_r, probs_r)
            y_pred_opt_r = (probs_r >= best_thresh_r).astype(int)
            s_opt_r, tp_or, fn_or, fp_or, tn_or = compute_score_robust(y_true_r, y_pred_opt_r)

            print(f"\n{'='*60}")
            print(f"Robust model at optimal threshold ({best_thresh_r:.2f})")
            print(f"{'='*60}")
            print(f"  TP={tp_or}, FN={fn_or}, FP={fp_or}, TN={tn_or}")
            print(f"  Accuracy:  {100*(tp_or+tn_or)/len(y_true_r):.2f}%")
            print(f"  Precision: {tp_or/max(tp_or+fp_or,1):.4f}")
            print(f"  Recall:    {tp_or/max(tp_or+fn_or,1):.4f}")
            print(f"  Custom Score: {s_opt_r}")

            # Probability distribution
            bot_probs_r = probs_r[y_true_r == 1]
            human_probs_r = probs_r[y_true_r == 0]
            print(f"\n  Actual bots   — mean: {bot_probs_r.mean():.4f}, median: {np.median(bot_probs_r):.4f}, "
                  f">0.5: {(bot_probs_r>=0.5).sum()}/{len(bot_probs_r)}")
            print(f"  Actual humans — mean: {human_probs_r.mean():.4f}, median: {np.median(human_probs_r):.4f}, "
                  f">0.5: {(human_probs_r>=0.5).sum()}/{len(human_probs_r)}")

        except ImportError:
            print("  Could not import detect_bots_robust — skipping robust evaluation")
    else:
        print(f"\nNo robust model found at {robust_path} — run 'python detect_bots_robust.py --train' first")


if __name__ == '__main__':
    main()
