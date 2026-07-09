#!/usr/bin/env bash
# ==========================================================================
# run_test_all.sh — evaluate ALL trained checkpoints (no training) and build
# Table 4.  This is the fast path when you already have checkpoints and only
# want the metrics.
#
# Assumes checkpoints live at  training/checkpoints/<model>/<model>9.pth
# (i.e. the 10th / last epoch, as saved by train_test*.py). Override the epoch
# via EPOCH=, or point CKPT_<model> at a specific file.
#
# Run from the training/ directory:
#     cd training && bash ../exam/run_test_all.sh
# ==========================================================================
set -uo pipefail

TESTCSV="${TESTCSV:-../dataset/test.csv}"
SAVEPATH="${SAVEPATH:-../results}"
TEST_BS="${TEST_BS:-32}"
EPOCH="${EPOCH:-9}"                       # train_test.py saves <model><epoch>.pth
MODELS="${MODELS:-xception efficientnet vit f3net spsl srm ucf UnivFD core daw_fdd dag_fdd fair_df_detector}"

test -f ./pretrained/xception-b5690688.pth || {
  echo "MISSING ./pretrained/xception-b5690688.pth (needed to build backbones)"; exit 1; }

logs=()
for m in ${MODELS}; do
  ckpt_var="CKPT_${m}"
  ckpt="${!ckpt_var:-./checkpoints/${m}/${m}${EPOCH}.pth}"
  if [[ ! -f "${ckpt}" ]]; then
    echo ">>> [skip] ${m}: checkpoint not found at ${ckpt}"
    continue
  fi
  echo ">>> [test] ${m}  <-  ${ckpt}"
  python ../exam/test_only.py \
      --model "${m}" --checkpoint "${ckpt}" \
      --test-csv "${TESTCSV}" --savepath "${SAVEPATH}" --test-batchsize "${TEST_BS}" \
      --log-file "./checkpoints/${m}/log_test.txt"
  logs+=( "--log" "${m}=./checkpoints/${m}/log_test.txt" )
done

echo ">>> Optional: F_IND (Individual row) per detector"
echo "    for m in ${MODELS}; do python ../exam/individual_fairness.py --model \$m \\"
echo "        --checkpoint ./checkpoints/\$m/\$m${EPOCH}.pth --test-csv ${TESTCSV}; done"

if (( ${#logs[@]} > 0 )); then
  echo ">>> Assembling Table 4 from test logs"
  python ../exam/parse_results_table4.py "${logs[@]}" \
      --find-csv "${SAVEPATH}/individual_fairness.csv" \
      --out-md "${SAVEPATH}/table4.md" --out-csv "${SAVEPATH}/table4_long.csv"
else
  echo ">>> No checkpoints were found; nothing to assemble."
fi
