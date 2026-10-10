import torch
from modules import compute_ctc_loss
logits = torch.zeros(1, 2, 3)
targets = torch.tensor([1])
targets_lengths = torch.tensor([1])
loss = compute_ctc_loss(logits, targets, targets_lengths)
print(loss.item())