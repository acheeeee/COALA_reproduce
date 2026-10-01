
"""從逐字稿中找出 rare words（= 論文 Eq.3/4/5 的 E⁺）。

依 fbai-speech/is21_deep_bias/README.md：
  - 判定「哪些字是 rare word」→ 用 common_words_5k.txt 的補集
    （不在 5000 常見詞裡的都算 rare）。因為測試集有些字根本沒出現在
    訓練語料，不會在 all_rare_words.txt 裡。
  - all_rare_words.txt → 只用來採樣干擾詞 E⁻。
"""

from pathlib import Path

WORDS_DIR = Path("/mnt/disk2/M11515125/fbai-speech/is21_deep_bias/words")
COMMON_WORDS_FILE = WORDS_DIR / "common_words_5k.txt"
RARE_WORDS_FILE = WORDS_DIR / "all_rare_words.txt"


def load_word_list(path):
    """讀一行一個詞的文字檔，回傳 set。"""
    words = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            word = line.strip()
            if word:
                words.add(word)
    return words


def load_common_words(path=COMMON_WORDS_FILE):
    """載入 5000 個常見詞（用來判定 rare word）。"""
    return load_word_list(path)


def load_rare_words(path=RARE_WORDS_FILE):
    """載入 rare word 詞池（用來採樣 E⁻，不是用來判定）。"""
    return load_word_list(path)


def find_rare_words(text, common_set):
    """回傳 text 裡所有「不在 common_words_5k 裡」的字，去重 + 字母排序。

    注意第二個參數是 common set，不是 rare set。
    """
    found = set()
    for word in text.lower().split():
        if word not in common_set:
            found.add(word)
    return sorted(found)


def main():
    common_set = load_common_words()
    print(f"載入 {len(common_set)} 個 common words")

    tests = [
        (
            "2830-3980-0017",
            "when i was a young man i thought paul was making too much of his call",
            [],
        ),
        (
            "237-134493-0004",
            "the air and the earth are curiously mated and intermingled "
            "as if the one were the breath of the other",
            ["intermingled", "mated"],
        ),
        ("8455-210777-0015-部分", "craswellers", ["craswellers"]),
    ]

    for utt_id, text, expected in tests:
        got = find_rare_words(text, common_set)
        mark = "PASS" if got == expected else "FAIL"
        print(f"[{mark}] {utt_id}")
        print(f"       我的結果: {got}")
        print(f"       官方答案: {expected}")


if __name__ == "__main__":
    main()