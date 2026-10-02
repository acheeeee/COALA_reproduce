"""為一個 LibriSpeech split 建立 manifest TSV.

輸出欄位（tab 分隔）：
    utt_id        1272-128104-0000
    audio_path    flac 路徑（相對於 repo 根目錄）
    duration      秒數
    n_words       逐字稿單字數
    text          逐字稿，已轉小寫
    rare_words    JSON 陣列，這句話裡的 rare words（= 論文的 E+）
"""

import argparse
import json
from pathlib import Path

import soundfile as sf
from tqdm import tqdm

from parse_trans import parse_trans_file
from rare_words import load_common_words, find_rare_words

LIBRISPEECH_ROOT = Path("data/LIBRISPEECH/LibriSpeech")
MANIFEST_DIR = Path("data/manifest")

FIELDS = ["utt_id", "audio_path", "duration", "n_words", "text", "rare_words"]


def build_manifest(split, common_set):
    """走訪一個 split，回傳 list of dict。"""
    split_dir = LIBRISPEECH_ROOT / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"找不到 split 目錄: {split_dir}")

    rows = []
    trans_files = sorted(split_dir.rglob("*.trans.txt"))

    for trans_file in tqdm(trans_files, desc=split, unit="chapter"):
        for utt_id, text in parse_trans_file(trans_file):
            audio_path = trans_file.parent / f"{utt_id}.flac"
            if not audio_path.is_file():
                raise FileNotFoundError(f"找不到音檔: {audio_path}")

            text = text.lower()
            rows.append(
                {
                    "utt_id": utt_id,
                    "audio_path": str(audio_path),
                    "duration": round(sf.info(str(audio_path)).duration, 3),
                    "n_words": len(text.split()),
                    "text": text,
                    "rare_words": json.dumps(find_rare_words(text, common_set)),
                }
            )
    return rows


def write_manifest(rows, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\t".join(FIELDS) + "\n")
        for r in rows:
            f.write("\t".join(str(r[k]) for k in FIELDS) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "splits",
        nargs="+",
        help="例如: dev-clean test-clean train-clean-100",
    )
    args = ap.parse_args()

    common_set = load_common_words()
    print(f"載入 {len(common_set)} 個 common words")

    for split in args.splits:
        rows = build_manifest(split, common_set)
        out_path = MANIFEST_DIR / f"{split}.tsv"
        write_manifest(rows, out_path)

        hours = sum(r["duration"] for r in rows) / 3600
        n_multi = sum(1 for r in rows if len(json.loads(r["rare_words"])) >= 2)
        n_zero = sum(1 for r in rows if len(json.loads(r["rare_words"])) == 0)
        print(
            f"[{split}] {len(rows)} utts, {hours:.2f} h | "
            f"無 rare word {n_zero} ({n_zero / len(rows) * 100:.2f}%) | "
            f"多目標 {n_multi} ({n_multi / len(rows) * 100:.2f}%) -> {out_path}"
        )


if __name__ == "__main__":
    main()
