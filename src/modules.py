"""COALA 的自訂模組：audio adapter 與 CTC head。"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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

def ctc_compress(hidden, logits, blank_id=0):
    """用 CTC 預測結果壓縮序列：去掉 blank 與連續重複的 frame。

      Args:
          hidden: (L, 576)  adapter 輸出
          logits: (L, 29)   CTC head 輸出
      Returns:
          (L', 576)  其中 L' 遠小於 L

      步驟：
        1. 對 logits 取 argmax，得到每個 frame 預測的字元 → (L,)
        2. 決定哪些 frame 要留：預測不是 blank，且和前一個 frame 的預測不同
        3. 用那些位置去取 hidden 的對應列
    """
   
    pred = logits.argmax(dim=-1)    
    #logits 是 (L, 29)的矩陣
    # L 是dim =0 , 29是dim =1（也可以寫-1)
    #argmax是在對那一列取最大值所在的位置 pred = logits.argmax(dim=-1)
    #最後取完 形狀會等於 (L,) 因為29裡面只會保留最大那個

    not_blank = pred != blank_id
    #因為要把所有空白刪掉 所以pred只要我預測出來的詞的位置跟blank id位置不同就設TRUE
    
    change = pred[1:] != pred[:-1]
    #pred[n:m] 表示法就是從第 'n' 印倒第 'm-1' 包頭不包包尾
    #True就是換字了

    first = torch.ones(1, dtype=torch.bool, device=pred.device)
    #因為 len(change) = 7對不齊

    changed = torch.cat([first, change])
    #cat = concatenate(串接)
    #changed = [True] + [False, True, False, False, True, True, False]
    #        = [True, False, True, False, False, True, True, False]

    keep = not_blank & changed
    #兩個都是true才是true因為有可能換到新的字但新的字是空格這樣我仍然不要
    
    return hidden[keep]

def compute_ctc_loss(logits, targets, target_lengths, blank_id=0):
    """算 CTC loss。

    Args:
        logits:         (B, T, C)  CTCHead 的輸出
        targets:        (B, S)     padded 的字元 id（來自 collate 的 ctc_ids）
        target_lengths: (B,)       每句真實字元數（來自 collate 的 ctc_lens）
        blank_id:       int        blank 的 id

    Returns:
        純量 loss

    注意三件事：
    1. CTCLoss 要 (T, B, C) 的維度順序 → 要 transpose(0, 1)
    2. 要先 log_softmax(dim=-1)，它吃的是 log 機率不是 logits
    3. input_lengths 全部等於 T（Whisper 固定 30 秒，沒有 padding）
    """
    # TODO 1: log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
    #         → (T, B, C)

    log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
    # .log_softmax()就是把分數變機率,然後再取LOG
    #eg X=[1,2,3] -> X.log_softmax() = [1/6, 2/6, 3/6]
    # .transpose(m, n)就是把第m, n維的順序互換

    # TODO 2: B, T = logits.shape[0], logits.shape[1]
    #         input_lengths = torch.full((B,), T, dtype=torch.long,
    #                                    device=logits.device)
    B, T = logits.shape[0], logits.shape[1]
    input_lengths = torch.full((B,), T, dtype=torch.long, device=logits.device)
            
    
    
    # TODO 3: 用 torch.nn.functional.ctc_loss(...)
    #         參數：log_probs, targets, input_lengths, target_lengths,
    #              blank=blank_id, reduction="mean", zero_infinity=True
    return F.ctc_loss(log_probs, targets, input_lengths, target_lengths,
                blank=blank_id, reduction="mean", zero_infinity=True)

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

    print("\n--- CTC loss ---")
    B, T, C = 4, 375, 29
    logits = torch.randn(B, T, C)
    targets = torch.randint(1, C, (B, 172))      # id 0 保留給 blank，所以從 1 開始
    target_lengths = torch.tensor([89, 63, 172, 133])

    loss = compute_ctc_loss(logits, targets, target_lengths)
    print(f"loss = {loss.item():.4f}")
    print(f"隨機初始化的理論值約 ln({C}) = {torch.log(torch.tensor(float(C))).item():.4f}")

if __name__ == "__main__":
    main()
