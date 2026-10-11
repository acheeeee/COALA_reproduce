import torch
from torch.utils.data import DataLoader
from transformers import WhisperFeatureExtractor
from dataset import LibriSpeechDataset, collate_fn
from model import COALAModel, WHISPER_ID

dev = torch.device("cuda")
fe = WhisperFeatureExtractor.from_pretrained(WHISPER_ID)
ds = LibriSpeechDataset("data/manifest/dev-clean.tsv", fe)
dl = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_fn)
feats = next(iter(dl))["input_features"].to(dev)

model = COALAModel().to(dev)
model.encoder.eval()

with torch.no_grad():
    a = model.encoder(feats).last_hidden_state
    with torch.autocast("cuda", dtype=torch.bfloat16):
        b = model.encoder(feats).last_hidden_state
    b = b.float()

rel = ((a - b).abs() / (a.abs() + 1e-6)).mean().item()
print(f"fp32 範圍   [{a.min():.3f}, {a.max():.3f}]  std {a.std():.4f}")
print(f"bf16 範圍   [{b.min():.3f}, {b.max():.3f}]  std {b.std():.4f}")
print(f"平均絕對誤差 {(a - b).abs().mean().item():.6f}")
print(f"平均相對誤差 {rel * 100:.3f}%")
print(f"餘弦相似度   {torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item():.6f}")
