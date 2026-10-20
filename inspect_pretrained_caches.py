
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_IDS = [
    "Qwen/Qwen2.5-0.5B",
    "EleutherAI/gpt-neo-125m",
]

TEXT = "Artificial intelligence can help people understand complex information."

for model_id in MODEL_IDS:
    print(f"\n{'=' * 60}")
    print(f"Loading: {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.eval()

    inputs = tokenizer(
        TEXT,
        return_tensors="pt",
        truncation=True,
        max_length=32,
    )

    with torch.no_grad():
        output = model(**inputs, use_cache=True)

    cache = output.past_key_values

    print("Model type:", model.config.model_type)
    print("Input token count:", inputs["input_ids"].shape[1])
    print("Cache type:", type(cache).__name__)

    # newer DynamicCache objects and older tuple caches
    if hasattr(cache, "layers"):
        layers = cache.layers
        get_key = lambda layer: layer.keys
        get_value = lambda layer: layer.values
    else:
        layers = cache
        get_key = lambda layer: layer[0]
        get_value = lambda layer: layer[1]

    print("Number of cached layers:", len(layers))

    for i in sorted(set([0, len(layers) // 2, len(layers) - 1])):
        key = get_key(layers[i])
        value = get_value(layers[i])

        print(f"Layer {i}:")
        print("  Key shape:", tuple(key.shape))
        print("  Value shape:", tuple(value.shape))
        print("  Key dtype:", key.dtype)

    del output, cache, model, tokenizer
    print("Finished:", model_id)
