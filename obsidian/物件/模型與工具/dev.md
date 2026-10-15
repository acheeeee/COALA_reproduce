---
tags: [COALA, 物件, 模型與工具]
類別: 模型與工具
型別: torch.device
---
# dev

← [[物件 總覽]]

**是什麼**：tensor 和模型要放在哪個裝置上。

| 項目 | 內容 |
|---|---|
| 型別 | `torch.device` |
| 值 | `torch.device("cuda")`；[[model.main()]] 會在沒有 GPU 時退回 `"cpu"` |

## 實際是哪張卡
`"cuda"` = 「看得到的第 0 張」。看得到哪張由環境變數決定：
```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID      # 不加的話編號和 nvidia-smi 不一致
export CUDA_VISIBLE_DEVICES=1            # RTX 3090
```

## 函式裡的 `dev`
[[build_stage1_sample()]]、[[build_gen_prompt()]] 內的 `dev = audio_tokens.device` —— 不是傳進來的，而是**跟著輸入走**。原則：函式裡新建的 tensor 一律 `device=某個輸入.device`。
