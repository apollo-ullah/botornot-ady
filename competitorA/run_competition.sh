#!/bin/bash
# Bot or Not Challenge - Competition Run Script
# ==============================================
# This script trains on all practice datasets and detects bots in evaluation data.
#
# Usage:
#   ./run_competition.sh <eval_en.json> <eval_fr.json> <team_name>
#
# Prerequisites:
#   pip install scikit-learn
#
# The script will produce:
#   <team_name>.detections.en.txt
#   <team_name>.detections.fr.txt

set -e

EVAL_EN="${1:?Usage: $0 <eval_en.json> <eval_fr.json> <team_name>}"
EVAL_FR="${2:?Usage: $0 <eval_en.json> <eval_fr.json> <team_name>}"
TEAM="${3:?Usage: $0 <eval_en.json> <eval_fr.json> <team_name>}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DETECTOR="$SCRIPT_DIR/bot_detector_v2.py"

# Practice datasets (adjust paths as needed)
TRAIN_DATA="dataset_posts_users_30.json:dataset_bots_30.txt"
TRAIN_DATA+=",dataset_posts_users_31.json:dataset_bots_31.txt"
TRAIN_DATA+=",dataset_posts_users_32.json:dataset_bots_32.txt"
TRAIN_DATA+=",dataset_posts_users_33.json:dataset_bots_33.txt"

echo "========================================="
echo "Bot or Not Challenge - ${TEAM}"
echo "========================================="
echo ""

echo "--- English Detection ---"
python3 "$DETECTOR" run "$EVAL_EN" \
    --train-data "$TRAIN_DATA" \
    --output "${TEAM}.detections.en.txt"

echo ""
echo "--- French Detection ---"
python3 "$DETECTOR" run "$EVAL_FR" \
    --train-data "$TRAIN_DATA" \
    --output "${TEAM}.detections.fr.txt"

echo ""
echo "========================================="
echo "DONE! Submit these files:"
echo "  - ${TEAM}.detections.en.txt"
echo "  - ${TEAM}.detections.fr.txt"
echo "========================================="
