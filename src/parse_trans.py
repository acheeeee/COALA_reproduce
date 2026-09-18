"""解析 LibriSpeech 的 trans.txt，印出 utterance ID 和逐字稿。"""

from pathlib import Path

TRANS = Path(
    "data/LIBRISPEECH/LibriSpeech/dev-clean/1272/128104/1272-128104.trans.txt"
)


def parse_trans_file(path):
    """讀一個 trans.txt，回傳 [(utt_id, text), ...]。"""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            utt_id, text = line.split(" ", 1)
            items.append((utt_id, text))
    return items


def main():
    items = parse_trans_file(TRANS)
    print(f"這個檔案有 {len(items)} 句")
    for utt_id, text in items[:3]:
        print(f"  {utt_id} | {text[:50]}...")


if __name__ == "__main__":
    main()



