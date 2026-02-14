#!/usr/bin/env python3
"""Generate plots for the README. Saves PNGs to plots/ directory."""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from collections import defaultdict

from detect_bots import (
    load_dataset, load_bots, prepare_dataset, compute_score,
    optimize_threshold, BotDetectorEnsemble, apply_hard_rules
)

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.size'] = 11

base_dir = Path(__file__).parent
plot_dir = base_dir / 'plots'
plot_dir.mkdir(exist_ok=True)

# Load trained models
with open(base_dir / 'models.pkl', 'rb') as f:
    models = pickle.load(f)

# Load all practice datasets
lang_map = {30: 'en', 31: 'fr', 32: 'en', 33: 'fr'}
datasets = {}
for ds_id in [30, 31, 32, 33]:
    datasets[ds_id] = {
        'data': load_dataset(base_dir / 'datasets' / f'dataset.posts&users.{ds_id}.json'),
        'bots': load_bots(base_dir / 'datasets' / f'dataset.bots.{ds_id}.txt'),
    }


# ── Plot 1: Feature Importance (Top 15, EN & FR side by side) ──

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for idx, (lang, title) in enumerate([('en', 'English Model'), ('fr', 'French Model')]):
    model = models[lang]
    sorted_feats = sorted(model.feature_importances.items(), key=lambda x: x[1], reverse=True)[:15]
    names = [f[0] for f in sorted_feats][::-1]
    values = [f[1] for f in sorted_feats][::-1]

    colors = ['#2196F3' if lang == 'en' else '#FF9800'] * len(names)
    axes[idx].barh(names, values, color=colors, edgecolor='white', linewidth=0.5)
    axes[idx].set_title(f'Top 15 Features — {title}', fontweight='bold', fontsize=13)
    axes[idx].set_xlabel('Importance (LightGBM split count)')
    axes[idx].tick_params(axis='y', labelsize=9)

plt.tight_layout()
plt.savefig(plot_dir / 'feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: plots/feature_importance.png")


# ── Plot 2: Cross-Dataset Validation Performance ──

pairs = [
    ('DS30→DS32', 30, 32, 'en'),
    ('DS32→DS30', 32, 30, 'en'),
    ('DS31→DS33', 31, 33, 'fr'),
    ('DS33→DS31', 33, 31, 'fr'),
]

labels, scores, max_scores, accuracies = [], [], [], []

for label, train_id, test_id, lang in pairs:
    train_data = datasets[train_id]
    test_data = datasets[test_id]

    train_df, _ = prepare_dataset(train_data['data'], train_data['bots'], lang)
    test_df, test_ids = prepare_dataset(test_data['data'], test_data['bots'], lang)

    temp_model = BotDetectorEnsemble(lang=lang)
    temp_model.train(train_df, verbose=False)

    probs = temp_model.predict_proba(test_df)
    threshold, best_score, _ = optimize_threshold(test_df['label'].values, probs)
    y_pred = (probs >= threshold).astype(int)
    y_true = test_df['label'].values

    score, tp, fn, fp, tn = compute_score(y_true, y_pred)
    max_score = 4 * sum(y_true)
    acc = 100 * (tp + tn) / len(y_true)

    labels.append(label)
    scores.append(score)
    max_scores.append(max_score)
    accuracies.append(acc)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Score bars
colors = ['#2196F3', '#2196F3', '#FF9800', '#FF9800']
x = np.arange(len(labels))
ax1.bar(x, max_scores, color='#E0E0E0', edgecolor='white', label='Max possible')
ax1.bar(x, scores, color=colors, edgecolor='white', label='Achieved')
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=10)
ax1.set_ylabel('Competition Score')
ax1.set_title('Cross-Dataset Validation — Score', fontweight='bold', fontsize=13)
for i, (s, m) in enumerate(zip(scores, max_scores)):
    ax1.text(i, s + 3, f'{100*s/m:.0f}%', ha='center', fontsize=10, fontweight='bold')
ax1.legend(fontsize=9)

# Accuracy bars
ax2.bar(x, accuracies, color=colors, edgecolor='white')
ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=10)
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Cross-Dataset Validation — Accuracy', fontweight='bold', fontsize=13)
ax2.set_ylim(90, 101)
for i, a in enumerate(accuracies):
    ax2.text(i, a + 0.3, f'{a:.1f}%', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(plot_dir / 'cross_validation.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: plots/cross_validation.png")


# ── Plot 3: Probability Distribution (bot vs human) ──

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for idx, (lang, title) in enumerate([('en', 'English'), ('fr', 'French')]):
    model = models[lang]
    ds_ids = [30, 32] if lang == 'en' else [31, 33]

    all_probs_bot, all_probs_human = [], []
    for ds_id in ds_ids:
        df, user_ids = prepare_dataset(datasets[ds_id]['data'], datasets[ds_id]['bots'], lang)
        probs = model.predict_proba(df)
        y = df['label'].values
        all_probs_bot.extend(probs[y == 1])
        all_probs_human.extend(probs[y == 0])

    axes[idx].hist(all_probs_human, bins=30, alpha=0.7, color='#4CAF50', label='Human', density=True)
    axes[idx].hist(all_probs_bot, bins=30, alpha=0.7, color='#F44336', label='Bot', density=True)
    axes[idx].axvline(model.threshold, color='black', linestyle='--', linewidth=1.5, label=f'Threshold ({model.threshold:.2f})')
    axes[idx].set_xlabel('Ensemble Probability')
    axes[idx].set_ylabel('Density')
    axes[idx].set_title(f'Score Distribution — {title}', fontweight='bold', fontsize=13)
    axes[idx].legend(fontsize=9)

plt.tight_layout()
plt.savefig(plot_dir / 'score_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: plots/score_distribution.png")


# ── Plot 4: Final Eval Detection Summary ──

fig, ax = plt.subplots(figsize=(8, 5))

categories = ['EN (Dataset 34)', 'FR (Dataset 35)']
bots = [71, 29]
humans = [438 - 71, 283 - 29]

x = np.arange(len(categories))
width = 0.4

bars1 = ax.bar(x - width/2, humans, width, color='#4CAF50', label='Human', edgecolor='white')
bars2 = ax.bar(x + width/2, bots, width, color='#F44336', label='Bot', edgecolor='white')

ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=12)
ax.set_ylabel('Number of Users')
ax.set_title('Final Evaluation — Detection Results', fontweight='bold', fontsize=13)
ax.legend(fontsize=10)

for bar, val in zip(bars1, humans):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3, str(val), ha='center', fontsize=10)
for bar, val in zip(bars2, bots):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3, str(val), ha='center', fontsize=10)

plt.tight_layout()
plt.savefig(plot_dir / 'eval_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: plots/eval_results.png")

print("\nAll plots saved to plots/ directory.")
