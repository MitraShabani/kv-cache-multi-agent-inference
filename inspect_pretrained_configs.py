
from transformers import AutoConfig

# Qwen uses grouped-query attention, while GPT-Neo uses a different attention architecture
model_ids = [
    "Qwen/Qwen2.5-0.5B",
    "EleutherAI/gpt-neo-125m",
]

for model_id in model_ids:
    config = AutoConfig.from_pretrained(model_id)

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
    head_dim = getattr(
        config, "head_dim",
        config.hidden_size // num_heads
    )

    print(f"\n--- {model_id} ---")
    print("Architecture:", config.architectures)
    print("Model type:", config.model_type)
    print("Hidden size:", config.hidden_size)
    print("Layers:", num_layers)
    print("Attention heads:", num_heads)
    print("KV heads:", num_kv_heads)
    print("Head dimension:", head_dim)
