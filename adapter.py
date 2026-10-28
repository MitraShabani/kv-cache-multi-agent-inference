
import torch
import torch.nn as nn
from inspect_pretrained_models import models_info

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

# Proportional layer mapping from source to target

layer_map = [
    round(gpt_layer * (dimensions["target_layers"] - 1) / (dimensions["source_layers"] - 1))  # linear interpolation
    for gpt_layer in range(dimensions["source_layers"])
]

print("Source layer -> Target layer")
for source_layer, target_layer in enumerate(layer_map):

    print(f"{source_layer:2} -> {target_layer:2}")

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

for a_layer, b_layer in enumerate(layer_map):

    K_A = get_key[1](layers[1][a_layer])
    V_A = get_value[1](layers[1][a_layer])

    K_B, V_B = adapter(K_A, V_A)

    adapted_K.append(K_B)
    adapted_V.append(V_B)

    print(
        f"Source layer {a_layer} -> Target layer {layer_map[a_layer]}: "
        f"K {tuple(K_B.shape)}, V {tuple(V_B.shape)}"
    )

print("Adapted layers:", len(adapted_K))
print("Expected source layers:", dimensions["source_layers"])
print("Expected target KV heads:", dimensions["target_heads"])

assert len(adapted_K) == len(layer_map)
assert len(adapted_V) == len(layer_map)

assert all(
    K.shape[1] == dimensions["target_heads"]
    and V.shape[1] == dimensions["target_heads"]
    for K, V in zip(adapted_K, adapted_V)
)

