
import gc
import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
)

MODEL_IDS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "microsoft/Phi-3.5-mini-instruct",
]

TEXTS = [
    "Artificial intelligence can help people understand complex information.",
    "Large language models generate text based on patterns learned from data.",
    "The researchers evaluated the system using several different benchmarks.",
    "Trustworthy AI requires reliable evaluation and transparent behavior.",
    "I am working on cross-model KV cache adaptation.",
]


def get_key(layer):
    return layer[0]


def get_value(layer):
    return layer[1]


def extract_cache_layers(past_key_values):
    """Extract a portable CPU copy of each layer's K and V tensors."""

    layers = []

    if hasattr(past_key_values, "layers"):
        for layer in past_key_values.layers:
            key = layer.keys.detach().cpu()
            value = layer.values.detach().cpu()
            layers.append((key, value))

    elif hasattr(past_key_values, "key_cache"):
        for key, value in zip(
            past_key_values.key_cache,
            past_key_values.value_cache,
        ):
            layers.append((
                key.detach().cpu(),
                value.detach().cpu(),
            ))

    else:
        # Legacy tuple-based cache
        for key, value in past_key_values:
            layers.append((
                key.detach().cpu(),
                value.detach().cpu(),
            ))

    return layers

import os

def save_cache(model_id, layers):
    safe_name = model_id.replace("/", "_")

    save_dir = "/content/drive/MyDrive/KVBridge/caches"
    os.makedirs(save_dir, exist_ok=True)

    path = os.path.join(save_dir, f"{safe_name}_cache.pt")

    torch.save(layers, path)

    print(f"Cache saved to: {path}")

def inspect_models(
    load_weights=False,
    device_map="auto",
    cache_text=None,
):
    results = {}

    for model_id in MODEL_IDS:

        print(f"\nInspecting {model_id}")

        config = AutoConfig.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        attention_heads = config.num_attention_heads

        dimensions = {
            "layers": config.num_hidden_layers,
            "attention_heads": attention_heads,
            "kv_heads": getattr(
                config, "num_key_value_heads", attention_heads
            ),
            "hidden_size": config.hidden_size,
            "head_dim": getattr(
                config,
                "head_dim",
                config.hidden_size // attention_heads,
            ),
        }

        tokenized_texts = []

        for text in TEXTS:
            tokens = tokenizer(
                text,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )

            tokenized_texts.append({
                "input_ids": tokens["input_ids"],
                "tokens": tokenizer.convert_ids_to_tokens(
                    tokens["input_ids"]
                ),
                "offsets": tokens["offset_mapping"],
            })

        model = None
        cache = None
        layers = None

        if load_weights:
            print("Loading model weights...")

            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype="auto",
                device_map=device_map,
            )

            model.eval()

            text = cache_text or TEXTS[0]

            inputs = tokenizer(
                text,
                return_tensors="pt",
            ).to(model.device)

            with torch.no_grad():
                outputs = model(
                    **inputs,
                    use_cache=True,
                )

            layers = extract_cache_layers(
                outputs.past_key_values
            )
            save_cache(model_id, layers)

            # Portable CPU cache representation
            cache = layers

            print("Extracted cache layers:", len(layers))
            print("First key shape:", layers[0][0].shape)
            print("First value shape:", layers[0][1].shape)

            # Release model weights before loading the next model
            del outputs, inputs, model
            model = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            gc.collect()

        results[model_id] = {
            "config": config,
            "tokenizer": tokenizer,
            "dimensions": dimensions,
            "tokenized_texts": tokenized_texts,
            "model": model,
            "cache": cache,
            "layers": layers,
            "get_key": get_key,
            "get_value": get_value,
        }

        print("Dimensions:", dimensions)

    print("\nTOKEN ALIGNMENT")

    for i, text in enumerate(TEXTS):
        print(f"\nText {i + 1}: {text}")

        for model_id in MODEL_IDS:
            offsets = results[model_id]["tokenized_texts"][i]["offsets"]

            print(
                f"{model_id.split('/')[0]:12} "
                f"Tokens: {len(offsets):2} | "
                f"Offsets: {offsets}"
            )

        qwen = results[MODEL_IDS[0]]["tokenized_texts"][i]["offsets"]
        mistral = results[MODEL_IDS[1]]["tokenized_texts"][i]["offsets"]
        phi = results[MODEL_IDS[2]]["tokenized_texts"][i]["offsets"]

        print("Qwen <-> Mistral:", qwen == mistral)
        print("Qwen <-> Phi:", qwen == phi)

    return results


if __name__ == "__main__":
    results = inspect_models(load_weights=False)
