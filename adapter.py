
import torch.nn as nn
from inspect_pretrained_models import models_info

results = models_info()

source_config = results[1][0] # GPT-Neo configuration
target_config = results[0][0] # Qwen configuration

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


