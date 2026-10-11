"""Kolibri 1 model configuration."""

from transformers import Qwen3MoeConfig


class Kolibri1Config(Qwen3MoeConfig):
    """Qwen3-MoE fields plus layer_types, sliding_window and
    shared_expert_intermediate_size, which config.json carries as extras."""

    model_type = "kolibri1"
