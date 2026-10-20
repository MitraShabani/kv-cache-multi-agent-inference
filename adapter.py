
# Proportional layer mapping from GPT-Neo to Qwen
# GPT-Neo (12 layers) -> Qwen (24 layers)

gpt_layers = 12
qwen_layers = 24

layer_map = [
    round(gpt_layer * (qwen_layers - 1) / (gpt_layers - 1))  # linear interpolation
    for gpt_layer in range(gpt_layers)
]

print("GPT-Neo layer -> Qwen layer")
for gpt_layer, qwen_layer in enumerate(layer_map):

    print(f"{gpt_layer:2} -> {qwen_layer:2}")

