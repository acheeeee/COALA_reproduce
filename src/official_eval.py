"""用 COALA 官方的評分程式碼（scripts/evaluation.py + scripts/score.py）原封不動算我們的 hyp 檔。

目的：排除「我們的評分方式和論文不同」這個可能。score_wer.py 的 official 那一行是我們照官方 code
重寫的；這支則直接 import 官方的檔案來算，不經過我們的實作。

官方的評分流程（github.com/Guo0911/COALA，commit 7b2e008）：
  1. reference = Whisper EnglishTextNormalizer(原始逐字稿).upper()      ← scripts/extract_feature2pt.py 第 41 行
  2. rare word = 正規化(all_rare_words) − 正規化(common_words_5k)        ← extract_feature2pt.py 第 73–83 行
  3. 每句的 biasing word = bias_list ∩ reference 的字                    ← evaluation.py 第 54 行
     （bias_list = 這句的 rare word + 干擾詞；干擾詞不會出現在句子裡，所以交集 = 句子裡的 rare word）
  4. hypothesis = EnglishTextNormalizer(hyp).upper()                     ← evaluation.py 第 51 行
  5. 用官方 score.py 對齊、算 WER / U-WER / B-WER

這支程式只負責做出官方 evaluation.py 吃的 results/*.json，然後呼叫官方的 main()。

用法（伺服器）
    git clone https://github.com/Guo0911/COALA.git /mnt/disk2/M11515125/COALA_official   # 只要做一次
    pip install whisper-normalizer                                                         # 官方用的正規化套件
    python src/official_eval.py --hyps exp/hyp/960h.test-clean.nobias.greedy_norep.tsv \\
        --refs /mnt/disk2/M11515125/fbai-speech/is21_deep_bias/ref/test-clean.biasing_1000.tsv \\
        --name 960h.clean.nobias.greedy_norep --tsv exp/wer_all.tsv
"""

import argparse
import importlib.util
import json
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

OFFICIAL_DIR = Path("/mnt/disk2/M11515125/COALA_official")
WORDS_DIR = Path("/mnt/disk2/M11515125/fbai-speech/is21_deep_bias/words")


def load_official(official_dir):
    """import 官方的 scripts/evaluation.py（它內部會 `from score import score`，所以要把 scripts/ 放進路徑）。"""
    scripts = Path(official_dir) / "scripts"
    if not (scripts / "evaluation.py").exists():
        raise SystemExit(f"找不到 {scripts}/evaluation.py，請先 git clone https://github.com/Guo0911/COALA.git {official_dir}")
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location("coala_official_evaluation", scripts / "evaluation.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def official_rare_words(normalizer, common_path, rare_path):
    """和官方 extract_feature2pt.py 第 73–83 行相同。"""
    common = set()
    with open(common_path) as f:
        for line in f.readlines():
            common.add(normalizer(line.strip()).upper())
    rare = set()
    with open(rare_path) as f:
        for line in f.readlines():
            w = normalizer(line.strip()).upper()
            if w not in common:
                rare.add(w)
    return rare


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyps", required=True, help="我們的 hyp 檔：utt_id \\t 轉寫")
    ap.add_argument("--refs", required=True, help="fbai ref 檔，只拿第 2 欄的逐字稿（任何 N 都可以）")
    ap.add_argument("--official-dir", default=str(OFFICIAL_DIR))
    ap.add_argument("--common-words", default=str(WORDS_DIR / "common_words_5k.txt"))
    ap.add_argument("--rare-words", default=str(WORDS_DIR / "all_rare_words.txt"))
    ap.add_argument("--name", default=None, help="寫進 --tsv 的標籤")
    ap.add_argument("--tsv", default=None, help="把結果附加到這個 TSV（style 欄寫 official_code）")
    ap.add_argument("--lenient", action="store_true", help="允許 hyp 少句子（只算有的）")
    args = ap.parse_args()

    from whisper_normalizer.english import EnglishTextNormalizer   # 和官方同一個套件，故意不給替代品
    normalizer = EnglishTextNormalizer()
    official = load_official(args.official_dir)

    refs = {}
    with open(args.refs, encoding="utf-8") as f:
        for line in f:
            a = line.rstrip("\n").split("\t")
            if len(a) >= 2:
                refs[a[0]] = a[1]
    hyps = {}
    with open(args.hyps, encoding="utf-8") as f:
        for line in f:
            a = line.rstrip("\n").split("\t")
            if a[0]:
                hyps[a[0]] = a[1] if len(a) >= 2 else ""
    missing = [u for u in refs if u not in hyps]
    print(f"ref {len(refs)} 句，hyp {len(hyps)} 句，缺 {len(missing)} 句")
    if missing and not args.lenient:
        raise SystemExit(f"hyp 缺少 {len(missing)} 句；確定只算有的句子請加 --lenient")

    rare = official_rare_words(normalizer, args.common_words, args.rare_words)
    print(f"官方 rare word 集合：{len(rare):,} 個")

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "results"
        out.mkdir()
        n_bias = 0
        for utt, text in refs.items():
            if utt not in hyps:
                continue
            reference = normalizer(text).upper()                                  # extract_feature2pt.py 第 41 行
            blist = list(rare.intersection(set(reference.split(" "))))            # extract_feature2pt.py 第 48 行
            n_bias += len(blist)
            result = {
                "reference": reference,
                "hypothesis": hyps[utt],                                         # 官方 evaluation.py 自己會正規化
                "bias_list": blist,
                "bias_scores": {w: 0.0 for w in blist},                          # 只給 Recall 用，這裡不看
            }
            (out / f"{utt}.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")

        official.main(Namespace(result_path=tmp))                               # 官方的 evaluation.main()
        metrics = json.loads((Path(tmp) / "metrics.json").read_text(encoding="utf-8"))

    asr = metrics["ASR"]
    b, w, u = asr["Biased"], asr["Overall"], asr["Unbiased"]
    print(f"\nofficial_code  B-WER {b['wer']:6.2f}  (WER {w['wer']:5.2f} / U-WER {u['wer']:5.2f})"
          f"   biasing word {b['ref']:,} / 全部 {w['ref']:,}")
    print("論文 COALA（test-clean / test-other）：w/o Biasing 23.39 (4.54 / 2.34) / 39.49 (8.75 / 5.48)、"
          "Oracle 1.77 (1.90 / 1.92) / 6.12 (4.86 / 4.73)")

    if args.tsv:
        p = Path(args.tsv)
        new = not p.exists()
        with open(p, "a", encoding="utf-8") as f:
            if new:
                f.write("name\tstyle\tB-WER\tWER\tU-WER\thyps\n")
            f.write(f"{args.name or Path(args.hyps).stem}\tofficial_code\t{b['wer']:.2f}\t{w['wer']:.2f}\t"
                    f"{u['wer']:.2f}\t{args.hyps}\n")
        print(f"附加到 {p}")


if __name__ == "__main__":
    main()
