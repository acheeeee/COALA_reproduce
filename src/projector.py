"""discriminative projector（論文 Eq. 1、2）"""
import torch
import torch.nn as nn

D = 576         # SmolLM2 hidden_size（已實測） 所以每個LM維度都是576
D_PRIME = 576   # 論文未給；預算上限約 4091


class DiscriminativeProjector(nn.Module):   #繼承“nn.Module”的一種
    def __init__(self, d=D, d_prime=D_PRIME):
        super().__init__()
        # TODO(1) Eq.2：Linear(d, d_prime) → ReLU → Linear(d_prime, 1)
        self.mlp = nn.Sequential(nn.Linear(d, d_prime,), nn.ReLU(), nn.Linear(d_prime, 1))

    def forward(self, h, mask):
        """
        h:    (N, L, D)  N 個 entity 對應位置的 last hidden state，已 pad 到 L
        mask: (N, L)     1 = 有效 token，0 = pad
        回傳: s (N,)     每個 entity 一個純量分數
        """
        # TODO(2) Eq.2：token-level 分數 f(h_ij) → (N, L)
        f = self.mlp(h)
        f = f.squeeze(-1)

        # TODO(3) Eq.1：只對有效 token 取平均 → (N,)
        total = (f * mask).sum(dim = 1)
        count = mask.sum(dim = 1)
        avg = total / count
        return avg

def main():
    torch.manual_seed(0)
    proj = DiscriminativeProjector()

    n = sum(p.numel() for p in proj.parameters())
    #proj.parameters() 一個一個交出模型裡的參數 tensor
    #p.numel() 是這個 tensor 裡有幾個數字(number of elements)
    # n 就是計算裡面所有參數到底有多少個
    print(f"params: {n:,}  (預期 {D_PRIME * (D + 2) + 1:,})")

    # 檢查 2：shape
    N, L = 121, 7
    h = torch.randn(N, L, D)
    lens = torch.randint(1, L + 1, (N,))
    #跟range一樣 下限包含、上限不包含。 所以生成範圍是 1~L
    #(N, ) tuple的寫法

    # 檢查 1：參數量
    mask = (torch.arange(L)[None, :] < lens[:, None]).float()

    s = proj(h, mask)
    print("s.shape:", tuple(s.shape), " (預期 (121,))")

    # 檢查 3：把 pad 位置改成垃圾值，分數不能變
    h2 = h.clone()
    h2[mask == 0] = 999.0
    print("pad 不影響分數:", torch.allclose(s, proj(h2, mask)))

    # 檢查 4：跟「每個 entity 單獨算、完全不 pad」的結果一致
    s_ref = torch.stack([
        proj(h[i:i + 1, :lens[i]], mask[i:i + 1, :lens[i]]).squeeze(0)
        for i in range(N)
    ])
    print("與逐個計算一致:", torch.allclose(s, s_ref, atol=1e-6))

    # 檢查 5：梯度到得了每個參數
    s.sum().backward()
    print("全部參數有梯度:", all(p.grad is not None for p in proj.parameters()))


if __name__ == "__main__":
    main()