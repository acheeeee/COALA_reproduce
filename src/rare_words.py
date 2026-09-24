from pathlib import Path

RARE_WORDS_FILE = Path(
    "/mnt/disk2/M11515125/fbai-speech/is21_deep_bias/words/all_rare_words.txt"
    )

def load_rare_words(path):
    rare_set = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            word = line.strip()
            if word:
                rare_set.add(word)
    return rare_set

def find_rare_words(text, rare_set):
    found = set()
    for word in text.lower().split():
        if word in rare_set:
            found.add(word)
    return sorted(found)

def main():
    rare_set = load_rare_words(RARE_WORDS_FILE)
    print(f"載入 {len(rare_set)} 個 rare words")

    test = [
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
    ]

    for utt_id, text, expected in test:
        got = find_rare_words(text, rare_set)
        mark = "Passed" if got == expected else "Failed"
        print(f"[{mark}] {utt_id}")
        print(f"       我的結果: {got}")
        print(f"       官方答案: {expected}")

if __name__ == "__main__":
    main()