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

print("\nModel configuration:")
print("hidden_size:", model.config.hidden_size)
print("num_hidden_layers:", model.config.num_hidden_layers)
print("num_attention_heads:", model.config.num_attention_heads)
print("num_key_value_heads:", model.config.num_key_value_heads)
print("head_dim:", model.config.head_dim)

print("\nChecking cache against config:")

print(
    "Number of cache layers:",
    len(cache.layers),
    "==",
    model.config.num_hidden_layers
)

print(
    "Number of KV heads:",
    cache.layers[0].keys.shape[1],
    "==",
    model.config.num_key_value_heads
)

print(
    "Head dimension:",
    cache.layers[0].keys.shape[-1],
    "==",
    model.config.head_dim
)

print(
    "Sequence length:",
    cache.layers[0].keys.shape[-2],
    "==",
    inputs["input_ids"].shape[1]
)


for layer_idx in range(len(cache)):
    key, value = cache.layers[layer_idx].keys, cache.layers[layer_idx].values

    print(f"\nLayer {layer_idx}")
    print("  Key shape:  ", key.shape)
    print("  Value shape:", value.shape)


# Inspect the cache after adding one more token to the sequence
next_token_id = torch.tensor([[100]])

with torch.no_grad():
    outputs2 = model(
        input_ids=next_token_id,
        past_key_values=cache,
        use_cache=True,
        return_dict=True,
    )

cache2 = outputs2.past_key_values

key, value = cache2.layers[0].keys, cache2.layers[0].values

print("\nAfter adding one token:")
print("Key shape:", key.shape)
print("Value shape:", value.shape)

# Compare the cache before and after adding one token
all_preserved = True
for old_layer, new_layer in zip(cache.layers, cache2.layers):
    old_key = old_layer.keys
    old_value = old_layer.values

    new_key = new_layer.keys
    new_value = new_layer.values

    keys_same = torch.allclose(
        old_key,
        new_key[:, :, :old_key.shape[2], :]
    )

    values_same = torch.allclose(
        old_value,
        new_value[:, :, :old_value.shape[2], :]
    )

    all_preserved = all_preserved and keys_same and values_same

print("Previous cache preserved:", all_preserved)