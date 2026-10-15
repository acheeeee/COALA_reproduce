---
tags: [COALA, 函式, 模型層, LoRA]
種類: 函式
檔案: lora_setup.py
---
# which_gets_grad()

← [[lora_setup.py]] · [[函式 總覽]]

**功能**：真的做一次 forward + backward，看哪一組參數**實際拿到梯度**。比數 `requires_grad` 更可靠。

```python
def which_gets_grad(model, tag):
```

## 步驟
1. `model.zero_grad(set_to_none=True)` → 所有 grad 清成 `None`
2. `x = torch.randn(1, 20, 576)` → `model(inputs_embeds=x).logits.pow(2).mean()` 當假 loss
3. `loss.backward()`
4. 數 `p.grad is not None` 的參數名分別屬於哪組
5. 再清一次 grad

## 用到的物件
[[logits (LM 輸出)]]（`(1, 20, 49152)`）

## 為什麼需要它
兩組 LoRA 設定一樣 → 參數量一樣（921,600）→ 光看數字分不出現在訓練的是哪組。而 `requires_grad=True` 也不保證這組有參與 forward（沒啟用的 adapter 不會被用到，backward 後 grad 還是 None）。

被 [[lora_setup.main()]] 呼叫。
