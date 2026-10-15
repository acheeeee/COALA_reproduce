---
tags: [COALA, 物件, 常數]
類別: 常數
型別: int
---
# D_PRIME

← [[物件 總覽]]

**存什麼**：discriminative projector 中間層的寬度，論文 Eq. 2 的 `D′`。

| 項目 | 內容 |
|---|---|
| 型別 | `int` |
| 值 | `576`（自選，論文未給） |
| 定義於 | [[projector.py]] |

## 怎麼決定的
```
projector 參數量 = D′·D + D′ + D′ + 1 = D′(D+2) + 1
剩餘預算 ≈ 5.7M − 2,413,469（Stage 1）− 921,600（LoRA scoring）≈ 2,364,931
→ D′ ≤ 2,364,931 / 578 ≈ 4091
選 D′ = 576 → 332,929 參數（已實測吻合）
```
5.7M 本身是由四捨五入的 777M 反推的，所以 4091 只是粗略上限。

## 用在哪裡
- [[DiscriminativeProjector.__init__()]] 的 `d_prime` 預設值
