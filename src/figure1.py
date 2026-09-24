
"""重現論文 Figure 1：統計 dev-clean 每句含幾個 rare word 的分布。"""
from pathlib import Path

from parse_trans import parse_trans_file
from rare_words import RARE_WORDS_FILE, load_rare_words, find_rare_words

SPLIT_DIR = Path("data/LIBRISPEECH/LibriSpeech/dev-clean")

BUCKET_ORDER = ["0", "1", "2-5", "6+"]

"""把 rare word 數量分到四個桶之一。"""
def bucket_of(n):

    if n == 0:
        return "0"
    if n == 1:
        return "1"
    if n <= 5:
        return "2-5"
    return "6+"


def main():
    rare_set = load_rare_words(RARE_WORDS_FILE)
    print(f"載入 {len(rare_set)} 個 rare words")

    counts = {"0": 0, "1": 0, "2-5": 0, "6+": 0}
    total = 0
    max_n = 0
    max_utt = None
    n_lt_10 = 0

    trans_files = sorted(SPLIT_DIR.rglob("*.trans.txt"))
    print(f"找到 {len(trans_files)} 個 trans.txt 檔")

    for trans_file in trans_files:
        for utt_id, text in parse_trans_file(trans_file):
            n = len(find_rare_words(text, rare_set))

            counts[bucket_of(n)] += 1
            total += 1

            if n < 10:
                n_lt_10 += 1

            if n > max_n:
                max_n = n
                max_utt = utt_id
    
    print()
    print(f"{'rare words':<12}{'utterances':>12}{'ratio':>10}")
    print("-" * 34)
    for name in BUCKET_ORDER:
        c = counts[name]
        print(f"{name:<12}{c:>12}{c / total * 100:>9.2f}%")
    print("-" * 34)
    print(f"{'total':<12}{total:>12}{100.0:>9.2f}%")
    print()
    print(f"最多的句子: {max_utt}，含 {max_n} 個 rare word")
    print(f"少於 10 個的佔比: {n_lt_10 / total * 100:.2f}%   (論文 3.4 節說 99%)")

if __name__ == "__main__":
       main()
