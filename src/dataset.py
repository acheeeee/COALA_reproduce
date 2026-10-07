"""LibriSpeech Dataset：把 manifest 的一行變成模型輸入。"""

import json
from pathlib import Path

import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import WhisperFeatureExtractor, AutoTokenizer

WHISPER_ID = "openai/whisper-large-v2"
LM_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"

MAX_DURATION = 30.0  # Whisper 的硬上限，超過的會被靜默截斷

CTC_CHARS = " 'abcdefghijklmnopqrstuvwxyz"  # 28 個，實測得到
CTC_BLANK_ID = 0
# char → id 的對照表，id 0 保留給 blank
CTC_CHAR2ID = {c: i + 1 for i, c in enumerate(CTC_CHARS)}


def text_to_ctc_ids(text):
    """把逐字稿轉成 CTC 的 id 序列（跳過表外字元）。"""
    return [CTC_CHAR2ID[c] for c in text if c in CTC_CHAR2ID]


class LibriSpeechDataset(Dataset):
    def __init__(self, manifest_path, feature_extractor, max_duration=MAX_DURATION):
        self.fe = feature_extractor
        self.rows = []

        n_skip = 0
        with open(manifest_path, "r", encoding="utf-8") as f:
            header = f.readline().rstrip("\n").split("\t")
            idx = {name: i for i, name in enumerate(header)}
            for line in f:
                p = line.rstrip("\n").split("\t")
                dur = float(p[idx["duration"]])
                if dur > max_duration:
                    n_skip += 1
                    continue
                self.rows.append({
                    "utt_id": p[idx["utt_id"]],
                    "audio_path": p[idx["audio_path"]],
                    "duration": dur,
                    "text": p[idx["text"]],
                    "rare_words": json.loads(p[idx["rare_words"]]),
                })
        print(f"{manifest_path}: 載入 {len(self.rows)} 句，跳過 {n_skip} 句（>{max_duration}s）")

    def __len__(self):
        # TODO: 回傳總筆數
        ...

    def __getitem__(self, i):
        r = self.rows[i]

        # TODO 1: 用 sf.read 讀出波形和取樣率
        # TODO 2: 用 self.fe(wav, sampling_rate=sr, return_tensors="pt")
        #         取出 .input_features，並用 [0] 去掉 batch 維 → (80, 3000)
        # TODO 3: 用 text_to_ctc_ids 把 r["text"] 轉成 id list，
        #         再轉成 torch.tensor（dtype=torch.long）

        return {
            "utt_id": r["utt_id"],
            "input_features": ...,  # (80, 3000)
            "text": r["text"],
            "ctc_ids": ...,  # (n_chars,)
            "rare_words": r["rare_words"],
        }


def collate_fn(batch):
    """把一批 sample 疊成 batch tensor。"""
    # TODO 4: input_features 全部等長 (80, 3000) → 用 torch.stack 疊成 (B, 80, 3000)

    # ctc_ids 長度不一，要 pad。CTC loss 需要知道每句的真實長度
    ctc_lens = torch.tensor([len(b["ctc_ids"]) for b in batch], dtype=torch.long)
    max_len = int(ctc_lens.max())
    ctc_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, b in enumerate(batch):
        ctc_ids[i, : len(b["ctc_ids"])] = b["ctc_ids"]

    return {
        "utt_id": [b["utt_id"] for b in batch],
        "input_features": ...,  # TODO 4 的結果
        "text": [b["text"] for b in batch],
        "ctc_ids": ctc_ids,  # (B, max_len)
        "ctc_lens": ctc_lens,  # (B,)
        "rare_words": [b["rare_words"] for b in batch],
    }


def main():
    fe = WhisperFeatureExtractor.from_pretrained(WHISPER_ID)
    ds = LibriSpeechDataset("data/manifest/dev-clean.tsv", fe)

    s = ds[0]
    print(f"\n單筆: {s['utt_id']}")
    print(f"  input_features {tuple(s['input_features'].shape)}")
    print(f"  ctc_ids        {tuple(s['ctc_ids'].shape)}")
    print(f"  text           {s['text'][:50]}...")
    print(f"  rare_words     {s['rare_words']}")

    dl = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_fn)
    b = next(iter(dl))
    print(f"\n一個 batch:")
    print(f"  input_features {tuple(b['input_features'].shape)}")
    print(f"  ctc_ids        {tuple(b['ctc_ids'].shape)}")
    print(f"  ctc_lens       {b['ctc_lens'].tolist()}")


if __name__ == "__main__":
    main()