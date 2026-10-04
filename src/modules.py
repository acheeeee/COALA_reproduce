"""COALA 的自訂模組：audio adapter 與 CTC head。"""

import torch
import torch.nn as nn

class AudioAdapter(nn.Module):
    """把 Whisper encoder 輸出接到 LM 的 embedding 空間。

      (B, 1500, 1280) → (B, 375, 576)

      做法：frame stacking —— 把相鄰 4 個 frame 的向量接成一個長向量
      （無資訊損失，只是重新排列），再用 Linear 壓到 576 維。
    """

    def __init__(self, d_in=1280, d_out=576, stack=4):
        # d_in 對應whisper encode的輸出 1280
        # d_out 對應adapter會送出來的目標維度 576
        
        super().__init__() #呼叫父類別（nn.Module)把原本寫好的init跑完
        self.stack = stack
        self.proj = nn.Linear(d_in*stack, d_out) #會把輸入 d_in*stack = 5120 的tensor變成 d_out = 576維
    def forward(self, x):
        #x =  (B, T, d_in)
        B, T, d_in = x.shape
        x = x.reshape(B, T // self.stack, d_in*self.stack)
        return self.proj(x)

class CTCHead(nn.Module):
    """把 adapter 輸出投影到 CTC 的字元詞彙。

    (B, L, 576) → (B, L, 29)

    詞彙只有 29 個（blank + a-z + 空格 + 撇號），不是 LM 的 49152。
    理由：576 × 49152 = 28.3M 參數，遠超過 5.7M 的可訓練預算。
    """

    def __init__(self, d_in=576, vocab_size=29):
        super().__init__()
        self.proj = nn.Linear(d_in, vocab_size)

    def forward(self, h):
        return self.proj(h)

def main():
    x = torch.randn(2, 1500, 1280)         
    # randn(產程隨機的數字0-1之間，n是常態分布，整體平均是0標準差是1),(batch, T, d_model)＝(一次處理兩個音檔, 總共有1500個audio token, 1280是whisper encod的輸出維度)

    adapter = AudioAdapter()
    h = adapter(x)
    print(f"adapter: {tuple(x.shape)} -> {tuple(h.shape)}")

    ctc = CTCHead()
    logits = ctc(h)
    print(f"ctc    : {tuple(h.shape)} -> {tuple(logits.shape)}")

    n_a = sum(p.numel() for p in adapter.parameters())
    n_c = sum(p.numel() for p in ctc.parameters())
    print(f"\nadapter 的數量有 {n_a:,}個")
    print(f"ctc 的數量有 {n_c:,}個")
    print(f"total {n_a + n_c:,} ,buget are 5,700,000")

if __name__ == "__main__":
    main()