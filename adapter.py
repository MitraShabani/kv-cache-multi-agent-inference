
import torch.nn as nn
from inspect_pretrained_models import models_info

results = models_info()

qwen_config = results[0][0]
gpt_config = results[1][0]

print("Source:", gpt_config.model_type)
print("Target:", qwen_config.model_type)

# Proportional layer mapping from GPT-Neo to Qwen

gpt_layers = gpt_config.num_hidden_layers
qwen_layers = qwen_config.num_hidden_layers

layer_map = [
    round(gpt_layer * (qwen_layers - 1) / (gpt_layers - 1))  # linear interpolation
    for gpt_layer in range(gpt_layers)
]

print("GPT-Neo layer -> Qwen layer")
for gpt_layer, qwen_layer in enumerate(layer_map):

    print(f"{gpt_layer:2} -> {qwen_layer:2}")
