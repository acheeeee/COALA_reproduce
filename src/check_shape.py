import json
import torch
from pathlib import Path
import soundfile as sf

from transformers import (
    AutoModelForCausalLM, 
    WhisperModel,
    WhisperFeatureExtractor
)

MANIFEST = Path("data/manifest/dev-clean.tsv")

WHISPER_ID = "openai/whisper-large-v2"
LM_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"

def pack_two_utterances(manifest_path):
    """從 MANIFEST 中挑出最短跟最長的兩句 然後回傳[(utt_id, path, duration....]"""
    rows = []

    with open(manifest_path, "r", encoding = "utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        i_utt = header.index("utt_id")
        i_path = header.index("audio_path")
        i_duration = header.index("duration")
        
        for line in f:
            parts = line.rstrip("\n").split("\t")
            rows.append((parts[i_utt], parts[i_path], float(parts[i_duration])))
    
    rows.sort(key = lambda r: r[2]) 
                #把rows排序 然後依照r'['2']'也就是"duration排" 
                #lambda 在這邊就是當作一個函數在call

    return [rows[0], rows[-1]]

def main():
    fe = WhisperFeatureExtractor.from_pretrained(WHISPER_ID)
    encoder = WhisperModel.from_pretrained(WHISPER_ID).get_encoder() #先拿whisper 裡面的encoder
    encoder.eval()  #模型切到 evaluation 模式(相對於 train() 模式)

    lm = AutoModelForCausalLM.from_pretrained(LM_ID)
    emb = lm.get_input_embeddings().weight
    print(f"\n[LM embedding查表] shape = {tuple(emb.shape)}")
    print(f"   → vocab_size = {emb.shape[0]}, hidden_size(D) = {emb.shape[1]}")
    print(f"   →Adapter的輸出維度必須是 = {emb.shape[1]}") #因為這樣才可以接到ＬＭ裡面

    for utt_id, audio_path, duration in pack_two_utterances(MANIFEST):

        print(f"\n {'=' * 66}")
        print(f"ID = {utt_id}, 時長 = {duration}")
        print(f"\n {'=' * 66}")

        wave, sr = sf.read(audio_path)
        print(f"[1] wave shape = {wave.shape} sr = {sr}")
        print(f"   → {duration} 秒 × {sr} Hz = {len(wave)} 個取樣點")

        feats = fe(wave, sampling_rate=sr, return_tensors="pt")
        #把wave跟頻率丟給fe(前處理器)他就會算好然後丟pt回來 pt表示是pythoch格式
        #feats算是一種dict(字典) 儲存方式就是一個key一個value
        mel = feats.input_features #也可以寫成 mel = feats["input_features"]

        print(f"[2] log-mel shape = {tuple(mel.shape)}")
        print(f"   → (batch, n_mels, n_frames)")

        with torch.no_grad():
            out = encoder(mel)

        hid = out.last_hidden_state

        print(f"[3] encoder 輸出    shape = {tuple(hid.shape)}")
        print(f"    → (batch, T, d_model),T = {hid.shape[1]} 個 audio token")
        print(f"    → adapter 的輸入維度是 {hid.shape[2]}")

if __name__ == "__main__":
    main()
