import torch
import torch.nn as nn
import inspect_different_models

config_A = inspect_different_models.config_A
config_B = inspect_different_models.config_B
head_dim = config_A.hidden_size // config_A.num_attention_heads

# Head adapters
"""Learnable parameters of the adapter
    (𝐲=𝐱𝐖𝑇+𝐛, where 𝐱 is the input, 𝐖 is the learnable weight matrix, and 𝐛 is the bias)
    hint: Linear only transforms the last dimension."""

K_adapter = nn.Linear(
    config_A.num_key_value_heads * head_dim,
    config_B.num_key_value_heads * head_dim
)
V_adapter = nn.Linear(
    config_A.num_key_value_heads * head_dim,
    config_B.num_key_value_heads * head_dim
)



# Apply it one layer
K_A = inspect_different_models.mapped_K_A[0]
print("Before:", K_A.shape)

"""Reshape and permute (put the dimensions we want to combine
    next to each other.) the cache to apply the adapter"""
# [B, 4, S, 16]
#      ↓
# [B, S, 4*16]
K_A = K_A.permute(0, 2, 1, 3)

B, S, H, D = K_A.shape
K_A = K_A.reshape(B, S, H * D)


# Learned transformation
K_B = K_adapter(K_A)

# [B, S, 6*16]
#      ↓
# [B, 6, S, 16]

K_B = K_B.reshape(
    B, S,
    config_B.num_key_value_heads,
    head_dim
)

K_B = K_B.permute(0, 2, 1, 3)

print("After:", K_B.shape)
