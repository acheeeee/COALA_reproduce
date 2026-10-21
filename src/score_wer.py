"""算 WER / U-WER / B-WER，兩種定義並排（論文 Table 2 的格式：B-WER (WER / U-WER)）。

fbai      benchmark 的官方定義（fbai-speech is21_deep_bias/score.py）：
          biasing word = ref 檔第 3 欄（不在 common_words_5k 裡的字）
          文字照原樣比對（這裡 ref / hyp / biasing word 都轉小寫，避免大小寫不一致）
official  COALA 官方程式碼的定義（scripts/evaluation.py + score.py）：
          hyp、ref 都先經 Whisper EnglishTextNormalizer，再轉大寫
          biasing word = 句子裡屬於「all_rare_words − common_words（兩者都正規化過）」的字
          （這個定義會漏掉 OOV 詞，見 實作論文差異 B8）

兩種都用 fbai score.py 的 EditDistance 對齊（COALA 官方的 score.py 是同一份演算法），
所以 fbai 那一行的數字和直接跑 fbai score.py 相同（差別只在先轉小寫）。

用法
    python src/score_wer.py --hyps exp/hyp/test-clean.s2.nobias.tsv \\
        --refs /mnt/disk2/M11515125/fbai-speech/is21_deep_bias/ref/test-clean.biasing_1000.tsv
    （B-WER 只看 ref 第 3 欄，和清單大小 N 無關，任何 N 的 ref 檔都可以）
"""

import argparse
import importlib.util
import json
import time
from pathlib import Path

from rare_words import COMMON_WORDS_FILE, RARE_WORDS_FILE

FBAI_SCORE = Path("/mnt/disk2/M11515125/fbai-speech/is21_deep_bias/score.py")


def load_fbai(path):
    """用檔案路徑載入 fbai 的 score.py（避免和其他叫 score 的模組撞名）。"""
    spec = importlib.util.spec_from_file_location("fbai_score", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def get_normalizer():
    """官方 code 用 whisper_normalizer 套件；沒裝的話改用 transformers 內建的同一套規則。"""
    try:
        from whisper_normalizer.english import EnglishTextNormalizer
        return EnglishTextNormalizer(), "whisper_normalizer.english.EnglishTextNormalizer"
    except ImportError:
        from transformers import WhisperTokenizer
        tok = WhisperTokenizer.from_pretrained("openai/whisper-large-v2")
        return tok.normalize, "transformers WhisperTokenizer.normalize"


def read_refs(path):
    """fbai ref 檔：utt_id \\t 文字 \\t E⁺(JSON) \\t 清單(JSON) → {utt_id: (文字, E⁺)}"""
    refs = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            a = line.rstrip("\n").split("\t")
            if len(a) >= 3:
                refs[a[0]] = (a[1], json.loads(a[2]))
    return refs


def read_hyps(path):
    hyps = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            a = line.rstrip("\n").split("\t")
            if a[0]:
                hyps[a[0]] = a[1] if len(a) >= 2 else ""
    return hyps


def official_rare_set(norm, common_path, rare_path):
    """官方 code 的 rare word 集合：正規化(all_rare_words) − 正規化(common_words)，大寫。"""
    def load(p):
        with open(p, encoding="utf-8") as f:
            return {norm(w.strip()).upper() for w in f if w.strip()}
    common = load(common_path)
    return {w for w in load(rare_path) if w not in common}


def count(fb, items):
    """items: [(ref_tokens, hyp_tokens, biasing_set), ...] → (wer, u_wer, b_wer) 三個 fbai WordError。"""
    C = fb.Code
    wer, u, b = fb.WordError(), fb.WordError(), fb.WordError()
    for ref, hyp, bw in items:
        res = fb.EditDistance().align(ref, hyp)
        for code, ri, hi in zip(res.codes, res.refs, res.hyps):
            if code == C.insertion:
                wer.errors[code] += 1
                (b if hyp[hi] in bw else u).errors[code] += 1
                continue
            tgt = b if ref[ri] in bw else u
            wer.ref_words += 1
            tgt.ref_words += 1
            if code != C.match:
                wer.errors[code] += 1
                tgt.errors[code] += 1
    return wer, u, b


def rate(w):
    return w.get_wer() if w.ref_words else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyps", required=True)
    ap.add_argument("--refs", required=True, help="fbai 官方 ref 檔（任何 N 都可以）")
    ap.add_argument("--style", default="both", choices=["both", "fbai", "official"])
    ap.add_argument("--lenient", action="store_true", help="允許 hyp 少句子（只算有的）")
    ap.add_argument("--name", default=None, help="寫進 --tsv 的標籤，例如 100h.nobias.greedy")
    ap.add_argument("--tsv", default=None, help="把結果附加到這個 TSV")
    ap.add_argument("--fbai-score", default=str(FBAI_SCORE))
    ap.add_argument("--common-words", default=str(COMMON_WORDS_FILE))
    ap.add_argument("--rare-words", default=str(RARE_WORDS_FILE))
    args = ap.parse_args()

    fb = load_fbai(args.fbai_score)
    refs, hyps = read_refs(args.refs), read_hyps(args.hyps)
    missing = [u for u in refs if u not in hyps]
    print(f"ref {len(refs)} 句，hyp {len(hyps)} 句，缺 {len(missing)} 句")
    if missing and not args.lenient:
        raise SystemExit(f"hyp 缺少 {missing[:3]} 等 {len(missing)} 句；確定要只算有的句子請加 --lenient")
    utts = [u for u in refs if u in hyps]

    results = []
    if args.style in ("both", "fbai"):
        items = [(refs[u][0].lower().split(), hyps[u].lower().split(), {w.lower() for w in refs[u][1]})
                 for u in utts]
        results.append(("fbai", *count(fb, items)))

    if args.style in ("both", "official"):
        norm, norm_name = get_normalizer()
        t0 = time.time()
        rare = official_rare_set(norm, args.common_words, args.rare_words)
        print(f"官方定義：正規化器 {norm_name}；rare word 集合 {len(rare):,} 個（{time.time() - t0:.0f} 秒）")
        items = []
        for u in utts:
            ref = norm(refs[u][0]).upper().split()
            hyp = norm(hyps[u]).upper().split()
            items.append((ref, hyp, {w for w in ref if w in rare}))
        results.append(("official", *count(fb, items)))

    print()
    for style, wer, u, b in results:
        print(f"{style:<9} B-WER {rate(b):6.2f}  (WER {rate(wer):5.2f} / U-WER {rate(u):5.2f})"
              f"   biasing word {b.ref_words:,} / 全部 {wer.ref_words:,}")
    print("\n論文 COALA（test-clean / test-other）：w/o Biasing 23.39 (4.54 / 2.34) / 39.49 (8.75 / 5.48)、"
          "Oracle 1.77 (1.90 / 1.92) / 6.12 (4.86 / 4.73)")

    if args.tsv:
        p = Path(args.tsv)
        new = not p.exists()
        with open(p, "a", encoding="utf-8") as f:
            if new:
                f.write("name\tstyle\tB-WER\tWER\tU-WER\thyps\n")
            for style, wer, u, b in results:
                f.write(f"{args.name or Path(args.hyps).stem}\t{style}\t{rate(b):.2f}\t{rate(wer):.2f}\t"
                        f"{rate(u):.2f}\t{args.hyps}\n")
        print(f"附加到 {p}")


if __name__ == "__main__":
    main()
