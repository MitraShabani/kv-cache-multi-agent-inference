
import torch
import torch.nn as nn
from inspect_pretrained_models import models_info
import math

(config, model, cache, layers, get_key, get_value) = zip(*models_info())

source_config = config[1] # GPT-Neo configuration
target_config = config[0] # Qwen configuration

print("Source:", source_config.model_type)
print("Target:", target_config.model_type)

def get_model_dimensions(source_config, target_config):
    return {
        "source_layers": source_config.num_hidden_layers,
        "target_layers": target_config.num_hidden_layers,
        "source_heads": source_config.num_key_value_heads
            if hasattr(source_config, "num_key_value_heads")
            else source_config.num_attention_heads,
        "target_heads": target_config.num_key_value_heads
            if hasattr(target_config, "num_key_value_heads")
            else target_config.num_attention_heads,
        "source_head_dim": source_config.hidden_size
            // source_config.num_attention_heads,
        "target_head_dim": target_config.hidden_size
            // target_config.num_attention_heads,
    }

dimensions = get_model_dimensions(source_config, target_config)

# Proportional layer mapping
# create one output for each target layer so: target to source mapping
""" Make layer_map represent the continuous source-layer position
    for every target layer.We won't round it to an integer, because
    interpolation needs the two neighboring source layers"""

layer_map = [
    target_layer * (dimensions["source_layers"] - 1) / (dimensions["target_layers"] - 1)  # Linear interpolation of layer positions
    for target_layer in range(dimensions["target_layers"])
]

print("Target layer -> Source layer")
for target_layer, source_layer in enumerate(layer_map):

    print(f"{target_layer:2} -> {source_layer:2}")

print()

# head adaptation in each layer
class KVLayerAdapter(nn.Module):
    def __init__(self, source_heads, target_heads):
        super().__init__()

        self.key_proj = nn.Linear(source_heads, target_heads)
        self.value_proj = nn.Linear(source_heads, target_heads)

    def forward(self, key, value):
        # Input: [batch, source_heads, sequence, head_dim]

        key = key.permute(0, 2, 3, 1)       # [B, S, D, H_source]
        value = value.permute(0, 2, 3, 1)

        key = self.key_proj(key)            # [B, S, D, H_target]
        value = self.value_proj(value)

        key = key.permute(0, 3, 1, 2)       # [B, H_target, S, D]
        value = value.permute(0, 3, 1, 2)

        return key, value


source_heads = (dimensions["source_heads"])
target_heads = (dimensions["target_heads"])

adapter = KVLayerAdapter(source_heads, target_heads)

adapted_K = []
adapted_V = []

for target_layer, position in enumerate(layer_map):
    """ One adapted KV representation for every target layer.
        - find the corresponding Source layer position
        - interpolate the source tensors
        - adapt the tensors for target """

    lower = math.floor(position)
    upper = math.ceil(position)
    alpha = position - lower

    K_lower = get_key[1](layers[1][lower])
    V_lower = get_value[1](layers[1][lower])

    if lower == upper:
        K_source = K_lower
        V_source = V_lower
    else:
        K_upper = get_key[1](layers[1][upper])
        V_upper = get_value[1](layers[1][upper])

        K_source = (1 - alpha) * K_lower + alpha * K_upper
        V_source = (1 - alpha) * V_lower + alpha * V_upper

    K_target, V_target = adapter(K_source, V_source)

    adapted_K.append(K_target)
    adapted_V.append(V_target)

    print(
        f"Qwen layer {target_layer} <- GPT-Neo position {position:.2f}: "
        f"K {K_target.shape}, V {V_target.shape}"
    )

print("\nValidation")
print("Expected target layers:", dimensions["target_layers"])
print("Adapted key tensors:", len(adapted_K))
print("Adapted value tensors:", len(adapted_V))
print()
print("First key shape:", adapted_K[0].shape)
print("Last key shape:", adapted_K[-1].shape)

assert len(adapted_K) == len(layer_map)
assert len(adapted_V) == len(layer_map)

assert all(
    K.shape[1] == dimensions["target_heads"]
    and V.shape[1] == dimensions["target_heads"]
    for K, V in zip(adapted_K, adapted_V)
)

