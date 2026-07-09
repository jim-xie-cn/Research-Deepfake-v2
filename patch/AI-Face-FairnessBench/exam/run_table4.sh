#!/usr/bin/env bash
# ==========================================================================
# run_table4.sh -- train + test every detector to reproduce Table 4.
#
# Each `train_test*.py` run trains for 10 epochs and, after every epoch, runs the
# intersectional fairness evaluation (acc_fairness) whose output is logged to
#     training/checkpoints/<model>/log_training.txt
# Parse those logs afterwards with parse_results_table4.py.
#
# Run from the repo's training/ directory:
#     cd training && bash ../exam/run_table4.sh
#
# Adjust GPU / batch size / data path via the env vars below.
# ==========================================================================
set -euo pipefail

DATAPATH="${DATAPATH:-../dataset/}"          # dir holding train.csv / test.csv
TESTCSV="${TESTCSV:-../dataset/test.csv}"
SAVEPATH="${SAVEPATH:-../results}"
TRAIN_BS="${TRAIN_BS:-128}"
TEST_BS="${TEST_BS:-32}"
LR="${LR:-0.0005}"

echo ">>> Pretrained weight check"
test -f ./pretrained/xception-b5690688.pth || {
  echo "MISSING ./pretrained/xception-b5690688.pth (see README section 3)"; exit 1; }

common="--datapath ${DATAPATH} --test_datapath ${TESTCSV} --savepath ${SAVEPATH} \
        --train_batchsize ${TRAIN_BS} --test_batchsize ${TEST_BS} --lr ${LR}"

# ---- 8 detectors that use the flat train.csv (no_pair) -------------------
for m in xception efficientnet f3net spsl srm core daw_fdd dag_fdd; do
  echo ">>> [no_pair] training/testing ${m}"
  python train_test.py --model "${m}" --dataset_type no_pair ${common}
done

# ---- 2 disentanglement detectors need the paired CSVs (see make_pair_dataset.py)
echo ">>> Generating paired CSVs for ucf / fair_df_detector"
python ../exam/make_pair_dataset.py --train-csv "${DATAPATH}/train.csv" --out-dir "${DATAPATH}"
for m in ucf fair_df_detector; do
  echo ">>> [pair] training/testing ${m}"
  python train_test.py --model "${m}" --dataset_type pair ${common}
done

# ---- ViT-B/16 and UnivFD have their own scripts -------------------------
echo ">>> training/testing vit (ViT-B/16)"
python train_test_vit.py --model vit ${common}
echo ">>> training/testing UnivFD (CLIP)"
python train_test_clip.py --model UnivFD ${common}

echo ">>> All runs done. Now assemble Table 4:"
cat <<'EOF'
  # (optional) individual fairness for the F_IND row, per detector:
  for m in xception efficientnet f3net spsl srm core ucf daw_fdd dag_fdd fair_df_detector; do
    python ../exam/individual_fairness.py --model $m \
      --checkpoint ./checkpoints/$m/${m}9.pth --test-csv ../dataset/test.csv
  done

  # assemble the grid:
  python ../exam/parse_results_table4.py \
    --log xception=./checkpoints/xception/log_training.txt \
    --log efficientnet=./checkpoints/efficientnet/log_training.txt \
    --log f3net=./checkpoints/f3net/log_training.txt \
    --log spsl=./checkpoints/spsl/log_training.txt \
    --log srm=./checkpoints/srm/log_training.txt \
    --log core=./checkpoints/core/log_training.txt \
    --log ucf=./checkpoints/ucf/log_training.txt \
    --log daw_fdd=./checkpoints/daw_fdd/log_training.txt \
    --log dag_fdd=./checkpoints/dag_fdd/log_training.txt \
    --log fair_df_detector=./checkpoints/fair_df_detector/log_training.txt \
    --log vit=./checkpoints/vit/log_training.txt \
    --log UnivFD=./checkpoints/UnivFD/log_training.txt \
    --find-csv ../results/individual_fairness.csv \
    --out-md ../results/table4.md
EOF
