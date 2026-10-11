# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Inference-only Kolibri 1 (Aleph-Alpha/Kolibri-1), a Qwen3-MoE variant with
4:1 sliding-window/full attention, RoPE only on the sliding-window layers,
sandwich norms, an ungated shared expert and logit+bias expert selection."""

from functools import partial
from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.layers.layer_boundary import (
    append_stages,
    declare_attn,
    declare_ffn,
)
from sglang.srt.layers.layer_boundary.output import OutputTransform
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe import get_moe_a2a_backend
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.layers.moe.utils import filter_moe_weight_param_global_expert
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.qwen3_moe import (
    Qwen3MoeAttention,
    Qwen3MoeDecoderLayer,
    Qwen3MoeForCausalLM,
    Qwen3MoeMLP,
    Qwen3MoeModel,
)
from sglang.srt.models.utils import apply_qk_norm
from sglang.srt.runtime_context import get_exec, get_parallel
from sglang.srt.utils import LazyValue, add_prefix, is_cuda
from sglang.srt.utils.hf_transformers_utils import get_rope_config

_is_cuda = is_cuda()

if _is_cuda:
    from sglang.kernels.ops.attention.fused_qknorm_rope import (
        can_use_fused_qk_norm_rope,
    )


def kolibri1_routing(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    *,
    correction_bias: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # The bias shifts the logits before selection; the weights are the
    # unbiased sigmoid. DeepSeek-style routing adds the bias after the sigmoid.
    logits = gating_output.float()
    topk_ids = torch.topk(logits + correction_bias, k=topk, dim=-1)[1]
    topk_weights = torch.sigmoid(logits.gather(1, topk_ids))
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights, topk_ids.to(torch.int32)


class Kolibri1SparseMoeBlock(nn.Module):
    def __init__(
        self,
        layer_id: int,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        if get_parallel().moe_tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {get_parallel().moe_tp_size} is greater "
                f"than the number of experts {config.num_experts}."
            )
        if not get_moe_a2a_backend().is_none():
            raise NotImplementedError(
                "Kolibri 1 supports only the default MoE all-to-all backend."
            )

        # The checkpoint keeps the router in bf16; selection runs on fp32 logits.
        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            params_dtype=torch.float32,
            quant_config=None,
            prefix=add_prefix("gate", prefix),
        )
        self.e_score_correction_bias = nn.Parameter(
            torch.zeros(config.num_experts, dtype=torch.float32),
            requires_grad=False,
        )
        self.topk = TopK(
            top_k=config.num_experts_per_tok,
            renormalize=config.norm_topk_prob,
            layer_id=layer_id,
            custom_routing_function=partial(
                kolibri1_routing, correction_bias=self.e_score_correction_bias
            ),
        )
        self.experts = get_moe_impl_class(quant_config)(
            num_experts=config.num_experts + get_exec().moe.ep_num_redundant_experts,
            top_k=config.num_experts_per_tok,
            layer_id=layer_id,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            quant_config=quant_config,
            prefix=add_prefix("experts", prefix),
        )
        self.shared_experts = Qwen3MoeMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.shared_expert_intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            reduce_results=False,
            prefix=add_prefix("shared_experts", prefix),
        )

    def get_moe_weights(self):
        return [
            x.data
            for name, x in self.experts.named_parameters()
            if name not in ["correction_bias"]
            and filter_moe_weight_param_global_expert(
                name, x, self.experts.num_local_experts
            )
        ]

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: Optional[ForwardBatch] = None,
    ) -> torch.Tensor:
        # Before the routed experts, which may overwrite hidden_states in place.
        shared_output = self.shared_experts(hidden_states)
        if hidden_states.shape[0] > 0:
            router_logits, _ = self.gate(hidden_states.to(torch.float32))
            topk_output = self.topk(hidden_states, router_logits)
        else:
            topk_output = self.topk.empty_topk_output(hidden_states.device)
        return self.experts(hidden_states, topk_output) + shared_output


class Kolibri1Attention(Qwen3MoeAttention):
    """Qwen3MoeAttention with a per-layer sliding window. Only the
    sliding-window layers apply RoPE; the full-attention layers use none."""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        is_sliding: bool,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        nn.Module.__init__(self)
        attn_tp_size = get_parallel().attn_tp_size
        self.config = config
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % attn_tp_size == 0
        self.num_heads = self.total_num_heads // attn_tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= attn_tp_size:
            assert self.total_num_kv_heads % attn_tp_size == 0
        else:
            assert attn_tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // attn_tp_size)
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=config.attention_bias,
            quant_config=quant_config,
            parallel_group="attn_tp",
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
            quant_config=quant_config,
            parallel_group="attn_tp",
            reduce_results=False,
            prefix=add_prefix("o_proj", prefix),
        )

        if is_sliding:
            self.rope_theta, rope_scaling = get_rope_config(config)
            self.rotary_emb = get_rope(
                self.head_dim,
                rotary_dim=self.head_dim,
                max_position=config.max_position_embeddings,
                base=self.rope_theta,
                rope_scaling=rope_scaling,
            )
            self.use_fused_qk_norm_rope = (
                get_exec().kernel.enable_fused_qk_norm_rope
                and _is_cuda
                and can_use_fused_qk_norm_rope(
                    self.head_dim,
                    self.rotary_emb.is_neox_style,
                    torch.bfloat16,
                    False,
                )
            )
        else:
            self.rotary_emb = None
            self.use_fused_qk_norm_rope = False
        # The fused KV-buffer write rides on the RoPE kernel.
        self.compatible_with_fused_kv_buffer = is_sliding
        self.use_fused_qk_norm_rope_cpu = False
        self._used_fused_qk_norm_rope_last_call = False

        # HF counts the current token in sliding_window; RadixAttention does not.
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            sliding_window_size=config.sliding_window - 1 if is_sliding else -1,
            prefix=add_prefix("attn", prefix),
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.alt_stream = alt_stream

    def forward_prepare_npu(self, positions, hidden_states, forward_batch):
        raise NotImplementedError("Kolibri 1 is not supported on NPU.")

    def apply_qk_norm_rope(self, qkv, positions, forward_batch):
        if self.rotary_emb is not None:
            return super().apply_qk_norm_rope(qkv, positions, forward_batch)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = apply_qk_norm(
            q=q,
            k=k,
            q_norm=self.q_norm,
            k_norm=self.k_norm,
            head_dim=self.head_dim,
            alt_stream=self.alt_stream,
        )
        return q, k, v


class Kolibri1DecoderLayer(Qwen3MoeDecoderLayer):
    """Qwen3MoeDecoderLayer with Kolibri1Attention and the MoE block, and
    sandwich norms on the attention and MoE outputs."""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        start_layer: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_id = layer_id
        layer_type = config.layer_types[layer_id]
        if layer_type not in ("sliding_attention", "full_attention"):
            raise ValueError(
                f"Unsupported layer_types[{layer_id}]={layer_type}; "
                "expected 'sliding_attention' or 'full_attention'"
            )

        self.self_attn = Kolibri1Attention(
            config=config,
            layer_id=layer_id,
            is_sliding=layer_type == "sliding_attention",
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
            alt_stream=alt_stream,
        )
        self.is_layer_sparse = True
        self.mlp = Kolibri1SparseMoeBlock(
            layer_id=layer_id,
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        attn, ffn = self.stage_facts(
            config,
            layer_id,
            post_attn_norm=self.post_attn_norm,
            post_ffn_norm=self.post_ffn_norm,
        )
        self.attn_boundary, self.ffn_boundary = append_stages(
            (attn, self.input_layernorm),
            (ffn, self.post_attention_layernorm),
        )

    @staticmethod
    def stage_facts(
        config: PretrainedConfig,
        layer_id: int,
        *,
        post_attn_norm=None,
        post_ffn_norm=None,
    ):
        """The stages a Kolibri 1 layer declares, from the config alone: the
        model's shared declaration function, which the layer declares with
        too (see make_layers). Its output norms are the output transforms."""
        return (
            declare_attn(output_transform=OutputTransform(post_attn_norm)),
            declare_ffn(
                sparse=True,
                next_layer_sparse=True,
                output_transform=OutputTransform(post_ffn_norm),
            ),
        )


class Kolibri1Model(Qwen3MoeModel):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            config=config,
            quant_config=quant_config,
            prefix=prefix,
            decoder_layer_type=Kolibri1DecoderLayer,
        )


class Kolibri1ForCausalLM(Qwen3MoeForCausalLM):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.pp_group = get_parallel().pp_group
        self.config = config
        self.quant_config = quant_config
        self.model = Kolibri1Model(
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
            use_attn_tp_group=get_parallel().enable_dp_lm_head,
        )
        self.logits_processor = LogitsProcessor(config)
        self.capture_aux_hidden_states = False
        self.routed_experts_weights_of_layer = LazyValue(
            lambda: {
                layer_id: self.model.layers[layer_id].mlp.get_moe_weights()
                for layer_id in range(self.start_layer, self.end_layer)
            }
        )

    def get_attention_sliding_window_size(self) -> int:
        return self.config.sliding_window - 1

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # torchtitan's name for the selection bias.
        return super().load_weights(
            (
                name.replace(".moe.router.expert_bias", ".mlp.e_score_correction_bias"),
                weight,
            )
            for name, weight in weights
        )


EntryClass = Kolibri1ForCausalLM
