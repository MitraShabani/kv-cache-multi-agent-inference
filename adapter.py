
import torch
import torch.nn as nn
from inspect_task_models import MODEL_IDS,load_saved_caches
import math
from transformers.cache_utils import DynamicCache

cache_dir = "./caches"
model_caches, model_dimensions = load_saved_caches(cache_dir)


def get_model_dimensions(model_dimensions, source_id, target_id):

    source = model_dimensions[source_id]
    target = model_dimensions[target_id]
    return {
        "source_layers": source["layers"],
        "target_layers": target["layers"],
        "source_heads": source["kv_heads"],
        "target_heads": target["kv_heads"],
        "source_head_dim": source["head_dim"],
        "target_head_dim": target["head_dim"],
    }


# head & head dimension adaptation (in each layer)
# head-wise projection
class KVLayerAdapter(nn.Module):

    def __init__(self, source_heads, target_heads, source_head_dim, target_head_dim):
        super().__init__()

        self.key_head_proj = nn.Linear(source_heads, target_heads)
        self.value_head_proj = nn.Linear(source_heads, target_heads)

        self.key_dim_proj = nn.Linear(source_head_dim, target_head_dim)
        self.value_dim_proj = nn.Linear(source_head_dim, target_head_dim)

    def forward(self, key, value):
        # Input: [batch, source_heads, sequence, head_dim]

        key = key.permute(0, 2, 3, 1)       # [B, S, D_source, H_source]
        value = value.permute(0, 2, 3, 1)

        key = self.key_head_proj(key)            # [B, S, D_source, H_target]
        value = self.value_head_proj(value)

        # Put head dimension last for nn.Linear
        key = key.permute(0, 1, 3, 2)       # [B, S ,H_target, D_source]
        value = value.permute(0, 1, 3, 2)

        # Adapt the head dimension
        key = self.key_dim_proj(key)
        value = self.value_dim_proj(value)

        # Return to cache layout
        # [B, S, H_target, D_target] -> [B, H_target, S, D_target]
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)

        return key, value


def adapt_cache(source_cache, dimensions, adapter):

    source_layers = dimensions["source_layers"]
    target_layers = dimensions["target_layers"]

    # Proportional layer mapping
    # create one output for each target layer so: target to source mapping
    """ Make layer_map represent the continuous source-layer position
        for every target layer.We won't round it to an integer, because
        interpolation needs the two neighboring source layers"""

    layer_map = [
    target_layer * (source_layers - 1) / (target_layers - 1)
    if target_layers > 1 else 0
    for target_layer in range(target_layers)
]

    """ One adapted KV representation for every target layer.
    - find the corresponding Source layer position
    - interpolate the source tensors
    - adapt the tensors for target """

    adapted_cache = []
    for source_layer_position in layer_map:

        lower = math.floor(source_layer_position)
        upper = math.ceil(source_layer_position)
        alpha = source_layer_position - lower

        K_lower, V_lower = source_cache[lower]

        if lower == upper:
            K_source, V_source = K_lower, V_lower
        else:
            K_upper, V_upper = source_cache[upper]

            K_source = (1 - alpha) * K_lower + alpha * K_upper
            V_source = (1 - alpha) * V_lower + alpha * V_upper

        K_target, V_target = adapter(
            K_source.float(),
            V_source.float()
        )

        adapted_cache.append((K_target, V_target))

    return adapted_cache

# Test for six pairs (source → target combinations)
for source_id in MODEL_IDS:
    for target_id in MODEL_IDS:

        if source_id == target_id:
            continue

        dimensions = get_model_dimensions(
            model_dimensions,
            source_id,
            target_id
        )

        source_heads = dimensions["source_heads"]
        target_heads = dimensions["target_heads"]
        source_head_dim = dimensions["source_head_dim"]
        target_head_dim = dimensions["target_head_dim"]

        adapter = KVLayerAdapter(source_heads, target_heads, source_head_dim, target_head_dim)

        adapted_cache = adapt_cache(
            model_caches[source_id],
            dimensions,
            adapter
        )

        assert len(adapted_cache) == dimensions["target_layers"]

        assert all(
            K.shape[1] == dimensions["target_heads"]
            and K.shape[3] == dimensions["target_head_dim"]
            and V.shape[1] == dimensions["target_heads"]
            and V.shape[3] == dimensions["target_head_dim"]
            for K, V in adapted_cache
        )

        print(f"\n{source_id} -> {target_id}")
        print("Adapted layers:", len(adapted_cache))
        print("First key shape:", adapted_cache[0][0].shape)
        print("First value shape:", adapted_cache[0][1].shape)



