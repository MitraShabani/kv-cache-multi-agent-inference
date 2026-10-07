import torch
import torch.nn as nn


# A and B dimensions
A_KV_heads = 4
B_KV_heads = 6
head_dim = 16


"""Learnable parameters of the adapter for K 
    (𝐲=𝐱𝐖𝑇+𝐛, where 𝐱 is the input, 𝐖 is the learnable weight matrix, and 𝐛 is the bias)
    hint: Linear only transforms the last dimension."""
K_adapter = nn.Linear(
    A_KV_heads * head_dim,
    B_KV_heads * head_dim
)


# A cache for one layer with random numbers
batch = 1
seq_len = 5

K_A = torch.randn(
    batch,
    A_KV_heads,
    seq_len,
    head_dim
)

print("K_A:", K_A.shape)

"""Reshape and permute (put the dimensions we want to combine 
    next to each other.) the cache to apply the adapter"""
# [B, 4, S, 16]
#      ↓
# [B, S, 4*16]
K_A = K_A.permute(0, 2, 1, 3)

K_A = K_A.reshape(
    batch,
    seq_len,
    A_KV_heads * head_dim
)

print("Before adapter:", K_A.shape)


# Learned transformation
K_B = K_adapter(K_A)

print("After adapter:", K_B.shape)


# [B, S, 6*16]
#      ↓
# [B, 6, S, 16]
K_B = K_B.reshape(
    batch,
    seq_len,
    B_KV_heads,
    head_dim
)

K_B = K_B.permute(0, 2, 1, 3)

print("K_B:", K_B.shape)