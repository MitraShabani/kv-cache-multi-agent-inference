from transformers import AutoConfig, AutoTokenizer

MODEL_IDS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "microsoft/Phi-3.5-mini-instruct",
]

TEXTS = ["Artificial intelligence can help people understand complex information.",
        "Large language models generate text based on patterns learned from data.",
        "The researchers evaluated the system using several different benchmarks.",
        "Trustworthy AI requires reliable evaluation and transparent behavior.",
        "I am working on cross-model KV cache adaptation.",
    ]


def inspect_models():

    results = {}

    for model_id in MODEL_IDS:

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
                config, "head_dim",
                config.hidden_size // attention_heads
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

        results[model_id] = {
            "config": config,
            "tokenizer": tokenizer,
            "dimensions": dimensions,
            "tokenized_texts": tokenized_texts,
        }

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

        qwen_offsets = results[MODEL_IDS[0]]["tokenized_texts"][i]["offsets"]
        mistral_offsets = results[MODEL_IDS[1]]["tokenized_texts"][i]["offsets"]
        phi_offsets = results[MODEL_IDS[2]]["tokenized_texts"][i]["offsets"]

        print("Qwen ↔ Mistral:", qwen_offsets == mistral_offsets)
        print("Qwen ↔ Phi:", qwen_offsets == phi_offsets)


if __name__ == "__main__":
    results = inspect_models()