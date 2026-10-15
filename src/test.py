import torch

x = torch.arange(5)[:,None]
y = torch.arange(3)[None, :]
z = torch.arange(4)
print(f"x shape = {x.shape}")

print(f"x column shape = {(x*y).shape }")

# print(f"x row  shape = {x[None, :].shape}")

