
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

def models_info():

    MODEL_IDS = [
        "Qwen/Qwen2.5-0.5B",
        "EleutherAI/gpt-neo-125m",
    ]

    TEXT = "Artificial intelligence can help people understand complex information."

    results = []
    for model_id in MODEL_IDS:

        # Inspect model configurations
        config = AutoConfig.from_pretrained(model_id)

        # Inspect KV caches
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

        # newer DynamicCache objects and older tuple caches
        if hasattr(cache, "layers"):
            layers = cache.layers
            get_key = lambda layer: layer.keys
            get_value = lambda layer: layer.values
        else:
            layers = cache
            get_key = lambda layer: layer[0]
            get_value = lambda layer: layer[1]

        results.append((config, model, cache, layers, get_key, get_value))
    return results


if __name__ == "__main__":

    results = models_info()

    for config, model, cache, layers, get_key, get_value in results:

        num_layers = getattr(
            config, "num_hidden_layers",
            getattr(config, "num_layers", None)
        )
        num_heads = getattr(
            config, "num_attention_heads",
            getattr(config, "num_heads", None)
        )
        num_kv_heads = getattr(
            config, "num_key_value_heads",
            num_heads
        )
        head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
            head_dim = config.hidden_size // num_heads

        print("Number of layers:", num_layers)
        print("Number of attention heads:", num_heads)
        print("Number of key-value heads:", num_kv_heads)
        print("Head dimension:", head_dim)

        print("\nModel type:", model.config.model_type)
        print("Cache type:", type(cache).__name__)
        print("Number of cached layers:", len(layers))

        for i in sorted(set([0, len(layers) // 2, len(layers) - 1])):
            key = get_key(layers[i])
            value = get_value(layers[i])

            print(f"Layer {i}:")
            print("  Key shape:", tuple(key.shape))
            print("  Value shape:", tuple(value.shape))
            print("  Key dtype:", key.dtype)

