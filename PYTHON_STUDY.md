# 有空時的自學方向（依優先序）

目標不是「學完 Python」，而是**補掉會卡住 COALA 實作的那幾個洞**。

---

## 優先 1：NumPy shape 操作 ⭐ 最重要

B 層（模型）的 bug 九成是 shape 對不上。只要會這六件事：

```python
a.shape                    # 看形狀
a.reshape(4, -1)           # 改形狀（-1 = 自動推算）
a[:, 2:5, :]               # 切片（含左不含右！）
a.mean(axis=1)             # 沿某維聚合（sum / max 同理，該維消失）
a.transpose(1, 0, 2)       # 換維度順序
a + b                      # broadcasting（形狀不同時如何自動對齊）
```

**練習方法：不要看影片。** 開直譯器造假資料，**先猜 shape 再印出來對答案**，猜錯就想為什麼。一小時勝過看三小時教學。

```python
import numpy as np
a = np.zeros((4, 10, 576))     # (batch, seq, hidden)

# 每行先在心裡說出答案再執行
a.mean(axis=1).shape
a.mean(axis=0).shape
a[:, 0, :].shape
a[:, 0:1, :].shape        # 和上一行不同！經典考點
a.reshape(4, -1).shape
a.transpose(1, 0, 2).shape
(a + np.zeros((576,))).shape
```

## 優先 2：Python 補三個洞

- **`dict`**：建立、讀取、`.get(key, 預設值)`、`.items()`、當計數器用
- **`enumerate()`** 和 **`zip()`**
- **`class`**：只需要 `__init__`、`self`、定義方法這三件事（不用碰繼承）

## 優先 3：PyTorch 最小集合

```python
torch.tensor / torch.zeros           # 建 tensor
tensor.shape / .dtype / .device      # 三個必看屬性
nn.Module + __init__ + forward       # 模型骨架（就是一個 class）
nn.Linear / nn.ReLU                  # 最基本的層
loss.backward() / optimizer.step()   # 訓練三行式
```

---

## 不用學（省時間）

資料庫 / SQL、網頁框架、async / await、decorator、類別繼承與多型、型別註解。這個專案幾乎不碰。

---

## 已踩過的坑（自己的錯誤筆記）

| 坑 | 教訓 |
|---|---|
| `nums.sort()` 改掉了外面的 list | `.sort()` 原地修改、回傳 None；`sorted()` 做新的。**預設一律用 `sorted()`** |
| `set = Path(...)` | 不要用 `set` / `list` / `dict` / `str` / `id` 等內建名稱當變數名 |
| `counts[w] += 1` 報 KeyError | dict 的鍵不存在不能直接累加，要先建立或用 `.get(w, 0) + 1` |
| 迴圈變數在迴圈結束後還存在 | 用 `_` 丟掉的變數，可能被上一個迴圈的殘留值取代，**不報錯但答案錯** |
| `a[:, 2:5, :]` 是 3 個不是 4 個 | Python 切片含左不含右 |
| `nn.Linear(576, 1)` 吃 `(2,8,576)` → `(2,8,1)` | Linear **只作用最後一維**，前面維度保留 |
| `nn.Linear(576, 1)` 有 577 個參數 | 別忘了 bias。公式：`in × out + out` |
