"""三個 loss（論文 Eq. 3、4、5-7）—— 完全不依賴模型"""
import torch
import torch.nn.functional as F


def disc_loss(s, pos):
    """
    Eq.3 discriminative loss（baseline，對全部 entity 做 softmax）
    s:   (K,)  一個 utterance 的 K = M+1 個分數（index 0 是 <unbiased>）
    pos: (K,)  bool，True = E⁺
    回傳: 純量
    """
    log_p = F.log_softmax(s, dim=0)
    log_p_pos = log_p[pos]
    loss = -log_p_pos.mean()
    return loss

def mpd_loss(s, pos):
    """Eq.4 MPD-Loss：每個正樣本各自只跟 E⁻ 做 softmax"""
    pos_scores = s[pos]   # 正樣本的分數,(|E⁺|,)
    neg_scores = s[~pos]   # 負樣本的分數,(|E⁻|,)
    lse_neg = torch.logsumexp(neg_scores, dim=0)      # c = log Σ exp(s_j),純量
    lse_pair = torch.logaddexp(pos_scores, lse_neg)# log(exp(s_i) + exp(c)),(|E⁺|,)
    loss = (lse_pair - pos_scores).mean()         # (lse_pair - pos_scores) 的平均
    return loss


def dpd_loss(s, pos):
    """Eq.5-7 DPD-Loss：每個 entity 各自做二元分類，P(E⁺) + N(E⁻)"""
    pos_scores = s[pos]   # 正樣本的分數 s_i, i ∈ E⁺,shape (|E⁺|,)
    neg_scores = s[~pos]   # 負樣本的分數 s_j, j ∈ E⁻,shape (|E⁻|,)
    loss_pos = -F.logsigmoid(pos_scores).mean()     # 𝒫(E⁺) = -(1/|E⁺|) Σ log σ(s_i),純量
    loss_neg = -F.logsigmoid(-neg_scores).mean()     # 𝒩(E⁻) = -(1/|E⁻|) Σ log σ(-s_j),純量
    loss = loss_pos+loss_neg         # loss_pos + loss_neg,純量
    return loss


def check(name, got, want):
    ok = abs(got.item() - want) < 1e-5
    print(f"{name:<20} {got.item():.6f}  (預期 {want:.6f})  {'✅' if ok else '❌'}")


def main():
    # 論文圖 2(b) 的例子：<unbiased>, Watson, solved, survey, collect
    s = torch.tensor([-2.0, 3.0, 1.0, -1.0, -3.0], requires_grad=True) #每個entity的分數
    pos = torch.tensor([False, True, True, False, False])

    print("── 測試 1：手算答案")
    check("disc", disc_loss(s, pos), 1.150889)
    check("mpd", mpd_loss(s, pos), 0.106171)
    check("dpd", dpd_loss(s, pos), 0.343850)

    print("── 測試 2：只有 <unbiased> 一個正樣本時，disc 必須等於 mpd")
    pos1 = torch.tensor([True, False, False, False, False])
    check("disc (單一正樣本)", disc_loss(s, pos1), 5.150889)
    check("mpd  (單一正樣本)", mpd_loss(s, pos1), 5.150889)

    print("── 測試 3：極端分數不能出現 inf / nan")
    s_big = torch.tensor([-200.0, 300.0, 100.0, -100.0, -300.0])
    for fn in (disc_loss, mpd_loss, dpd_loss):
        v = fn(s_big, pos)
        print(f"{fn.__name__:<20} {v.item():.4f}  {'✅' if torch.isfinite(v) else '❌ inf/nan'}")

    print("── 測試 4：每個 entity 收到的梯度")
    for fn in (disc_loss, mpd_loss, dpd_loss):
        s.grad = None
        fn(s, pos).backward()
        print(f"{fn.__name__:<20} {[round(g, 4) for g in s.grad.tolist()]}")


if __name__ == "__main__":
    main()