#!/bin/bash
# Table 2：一個測試集的 BTI 列或整份清單列，從打分一路跑到 score_wer.py。
#
#   bti   DPD 對 N=500/1000/5000 打分 → bti.py 取 top-10 + <unbiased> 門檻 → Stage 1 解碼 → 算分
#   full  整份清單（N=500/1000）直接放進 Stage 1 的 prompt → 解碼 → 算分
#         N=5000 約 1.7 萬 token，超過 SmolLM2 的 8192 上下文（論文寫 OOM），只跑前 20 句記錄現象
#
# 用法（在 repo 根目錄）
#   CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 nohup bash src/run_table2.sh test-clean bti > exp/t2_clean_bti.log 2>&1 &
# 換 960 h：S1=exp/stage1_960/best.pt S2=exp/stage2_dpd_960/epoch6.pt TAG=960h ID=960h.dpd.e6 bash src/run_table2.sh ...
#
# 打分可以接續（score_stage2.py 會跳過已完成的句子）；解碼不能，中斷了那一項要整個重跑。

set -u
SPLIT=$1                                   # test-clean / test-other
MODE=$2                                    # bti / full
S1=${S1:-exp/stage1_tc100_s2/best.pt}
S2=${S2:-exp/stage2_dpd/epoch6.pt}
TAG=${TAG:-100h}
ID=${ID:-dpd.e6}                           # 打分檔名；100 h 的 N=5000 已經打完叫 dpd.e6.*，沿用就不會重打
REF=/mnt/disk2/M11515125/fbai-speech/is21_deep_bias/ref/$SPLIT.biasing_1000.tsv   # B-WER 只看第 3 欄，和 N 無關
SHORT=${SPLIT#test-}                       # clean / other
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p exp/score exp/lists exp/bti exp/hyp
run() { echo; echo "[$(date +%m-%d_%H:%M)] $*"; "$@"; }

decode_and_score() {                       # $1 = entity JSON, $2 = 名稱, $3 = 只跑前幾句（可省略）
    local ents=$1 name=$2 limit=${3:-}
    local hyp=exp/hyp/$TAG.$SPLIT.$name.tsv
    run python src/evaluate.py --ckpt "$S1" --stack 2 --manifest data/manifest/$SPLIT.tsv \
        --mode file --entities "$ents" --amp auto --out "$hyp" ${limit:+--limit $limit} &&
    run python src/score_wer.py --hyps "$hyp" --refs "$REF" --name "$TAG.$SHORT.$name.greedy" \
        --tsv exp/wer_all.tsv ${limit:+--lenient}
}

case $MODE in
bti)
    for N in 500 1000 5000; do
        run python src/score_stage2.py --ckpt "$S2" --split $SPLIT --n $N \
            --out exp/score/$ID.$SPLIT.N$N.jsonl --dump-lists exp/lists/$SPLIT.N$N.json &&
        run python src/bti.py exp/score/$ID.$SPLIT.N$N.jsonl --out exp/bti/$ID.$SPLIT.N$N.json
    done
    for N in 500 1000 5000; do
        decode_and_score exp/bti/$ID.$SPLIT.N$N.json bti_dpd.N$N
    done ;;
full)
    for N in 500 1000 5000; do              # --limit 0：不打分，只把清單寫出來
        [ -f exp/lists/$SPLIT.N$N.json ] ||
            run python src/score_stage2.py --ckpt "$S2" --split $SPLIT --n $N --limit 0 \
                --out /dev/null --dump-lists exp/lists/$SPLIT.N$N.json
    done
    decode_and_score exp/lists/$SPLIT.N500.json  full.N500
    decode_and_score exp/lists/$SPLIT.N1000.json full.N1000
    decode_and_score exp/lists/$SPLIT.N5000.json full.N5000.first20 20 ;;
*)
    echo "MODE 要是 bti 或 full"; exit 1 ;;
esac
echo; echo "[$(date +%m-%d_%H:%M)] 全部完成"
