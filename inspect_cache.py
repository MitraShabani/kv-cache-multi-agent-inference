import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

model.eval()

text = "The cat sat on the"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(
        **inputs,
        use_cache=True,
        return_dict=True,
    )

cache = outputs.past_key_values

print("Cache type:", type(cache))
print("Number of layers:", len(cache))

for layer_idx in range(len(cache)):
    key, value = cache.layers[layer_idx].keys, cache.layers[layer_idx].values

    print(f"\nLayer {layer_idx}")
    print("  Key shape:  ", key.shape)
    print("  Value shape:", value.shape)