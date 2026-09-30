import torch
from transformers import LlamaConfig, LlamaForCausalLM

config_A = LlamaConfig(
    hidden_size=128,
    intermediate_size=256,
    num_hidden_layers=4,
    num_attention_heads=8,
    num_key_value_heads=4,
)

config_B = LlamaConfig(
    hidden_size=192,
    intermediate_size=384,
    num_hidden_layers=6,
    num_attention_heads=12,
    num_key_value_heads=6,
)

model_A = LlamaForCausalLM(config_A)
model_B = LlamaForCausalLM(config_B)

model_A.eval()
model_B.eval()

input_ids = torch.tensor([[1, 2, 3, 4, 5]])

with torch.no_grad():
    output_A = model_A(
        input_ids=input_ids,
        use_cache=True,
        return_dict=True,
    )

    output_B = model_B(
        input_ids=input_ids,
        use_cache=True,
        return_dict=True,
    )

cache_A = output_A.past_key_values
cache_B = output_B.past_key_values

print("Model A")
print("Number of layers:", len(cache_A.layers))
print("Layer 0 K:", cache_A.layers[0].keys.shape)
print("Layer 0 V:", cache_A.layers[0].values.shape)

print("\nModel B")
print("Number of layers:", len(cache_B.layers))
print("Layer 0 K:", cache_B.layers[0].keys.shape)
print("Layer 0 V:", cache_B.layers[0].values.shape)

""" These are two independently initialized models,
so of course their learned representations will differ. This is exactly
the reason that we should transform representations not merely change the shape! """

print("Model A first KV head, first token:")
print(cache_A.layers[0].keys[0, 0, 0, :])

print("\nModel B first KV head, first token:")
print(cache_B.layers[0].keys[0, 0, 0, :])