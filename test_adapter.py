import torch
import torch.nn as nn
from inspect_different_models import models_and_caches

config_A, config_B, cache_A, cache_B = models_and_caches()

head_dim_A = config_A.hidden_size // config_A.num_attention_heads
head_dim_B = config_B.hidden_size // config_B.num_attention_heads

# Head adaptation
# Head adapters
"""Learnable parameters of the adapter
    (𝐲=𝐱𝐖𝑇+𝐛, where 𝐱 is the input, 𝐖 is the learnable weight matrix, and 𝐛 is the bias)
    hint: Linear only transforms the last dimension."""

K_adapter = nn.Linear(
    config_A.num_key_value_heads * head_dim_A,
    config_B.num_key_value_heads * head_dim_B
)
V_adapter = nn.Linear(
    config_A.num_key_value_heads * head_dim_A,
    config_B.num_key_value_heads * head_dim_B
)

def adapt_heads(cache_tensor, adapter):
    """Reshape and permute (put the dimensions we want to combine
    next to each other.) the cache to apply the adapter"""

    # [B, H_A, S, D_A]
    #      ↓
    # [B, S, H_A * D_A]
    x = cache_tensor.permute(0, 2, 1, 3)

    B, S, H, D = x.shape

    # [B, S, H_A * D_A]
    x = x.reshape(B, S, H * D)

    # [B, S, H_B * D_A]
    x = adapter(x)

    # [B, S, H_B, D_B]
    x = x.reshape(
        B,
        S,
        config_B.num_key_value_heads,
        config_B.hidden_size // config_B.num_attention_heads
    )

    # [B, H_B, S, D_B]
    x = x.permute(0, 2, 1, 3)

    return x

# Layer matching
# Proportional layer mapping
def create_layer_map(num_layers_A, num_layers_B):
    layer_map = []

    for b_layer in range(num_layers_B):
        """ we want the beginning and end to correspond so we just measure the distance between indices: intervals so num_layers_A - 1"""
        a_layer = round(
            (b_layer * (num_layers_A - 1)) / (num_layers_B - 1)
        )
        layer_map.append(a_layer)

    return layer_map

layer_map = create_layer_map(
    config_A.num_hidden_layers,
    config_B.num_hidden_layers
)
print(layer_map)

# main
adapted_K = []
adapted_V = []

for b_layer, a_layer in enumerate(layer_map):

    K_A = cache_A.layers[a_layer].keys
    V_A = cache_A.layers[a_layer].values

    K_B = adapt_heads(K_A, K_adapter)
    V_B = adapt_heads(V_A, V_adapter)

    adapted_K.append(K_B)
    adapted_V.append(V_B)

# Number of layers
num_layers_A = len(cache_A.layers)
num_layers_B = len(cache_B.layers)

for b_layer in range(num_layers_B):
    print(
        f"B layer {b_layer}: "
        f"K {adapted_K[b_layer].shape}, "
        f"V {adapted_V[b_layer].shape}"
    )

# Calculate MSE loss for the adapted keys and values
# Use B's actual cache as the target

""" Adam updates the Linear layers' weights and biases
to reduce the difference between predicted and actual B caches"""

optimizer = torch.optim.Adam(
    list(K_adapter.parameters()) + list(V_adapter.parameters()),
    lr=0.001
)

loss_fn = nn.MSELoss()

# Number of training iterations
for epoch in range(100):

    # clear the previous batch's gradients at the start of each epoch
    optimizer.zero_grad()

    total_K_loss = 0
    total_V_loss = 0

    for b_layer, a_layer in enumerate(layer_map):

        K_A = cache_A.layers[a_layer].keys
        V_A = cache_A.layers[a_layer].values

        predicted_K = adapt_heads(K_A, K_adapter)
        predicted_V = adapt_heads(V_A, V_adapter)

        target_K = cache_B.layers[b_layer].keys
        target_V = cache_B.layers[b_layer].values

        total_K_loss += loss_fn(predicted_K, target_K)
        total_V_loss += loss_fn(predicted_V, target_V)

    total_loss = total_K_loss + total_V_loss

    """ Calculate gradients for the adapter's parameters (PyTorch saves the gradients automatically inside each parameter.)
    and update the parameters using those gradients."""
    total_loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch {epoch + 1}: "
            f"Loss = {total_loss.item():.4f}"
        )
