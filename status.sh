show() { [ -f exp/960h/$1.log ] || return; printf '  %-24s %s\n' "$2" "$(tail -c 2000 exp/960h/$1.log | tr '\r' '\n' | grep -av '^\s*$' | tail -n 1)"; }
echo "############ GPU 0"
show stage2_mpd               "960h Stage 2 MPD 訓練"
echo "############ GPU 1"
show stage2_score_bias        "960h Bias 打分"
echo "############ GPU 2"
show table2_bti_test-clean    "960h DPD BTI clean"
echo "############ GPU 3"
show table2_bti_test-other    "960h DPD BTI other"
echo "############ 已完成的 epoch（共 7 個）"
for m in dpd bias mpd; do f=exp/960h/stage2_$m/epoch_summary.tsv; [ -f $f ] && printf '  %-5s %s\n' $m $(( $(wc -l < $f) - 1 )); done
echo "############ 打分進度（clean 2620 / other 2939）"
for f in exp/score/960h.*.N5000.jsonl; do [ -f "$f" ] && printf '  %-45s %s\n' "$(basename $f)" "$(wc -l < $f)"; done
echo "############ GPU 記憶體"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
