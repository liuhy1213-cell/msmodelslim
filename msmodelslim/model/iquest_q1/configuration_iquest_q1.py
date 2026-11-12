#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

MindStudio is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
For more details, please refer to the LICENSE file in the project root.

THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
-------------------------------------------------------------------------
"""

"""IQuest-Q1 的 transformers 配置类。

IQuest-Q1 是自定义架构，``config.json`` 里的 ``model_type`` 是 ``iquest_q1``，
transformers 不认识它 —— 直接 ``AutoConfig.from_pretrained()`` 会报
"The checkpoint you are trying to load has model type `iquest_q1` but Transformers
does not recognize this architecture"。

这里提供最小可用的 ``PretrainedConfig`` 子类并注册到 ``AutoConfig``，
字段全部来自 checkpoint 的 config.json（``PretrainedConfig.__init__`` 会把
未知 kwargs 原样设为实例属性），因此不需要随包携带一份 config 代码。

字段定义与 vllm-ascend 的 ``vllm_ascend/models/iquest_q1/configs.py`` 保持一致。
"""

from transformers import PretrainedConfig


class IQuestQ1Config(PretrainedConfig):
    """IQuest-Q1 配置（text-only MoE + learned sink attention + 混合滑窗注意力）。"""

    model_type = "iquest_q1"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 160000,
        hidden_size: int = 3072,
        intermediate_size: int = 1536,
        dense_intermediate_size: int = 12288,
        num_hidden_layers: int = 88,
        num_attention_heads: int = 48,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        hidden_act: str = "silu",
        max_position_embeddings: int = 524288,
        rms_norm_eps: float = 1e-6,
        num_experts: int = 256,
        num_experts_per_tok: int = 8,
        mlp_only_layers: list[int] | None = None,
        use_hybrid_layers: bool = True,
        first_layers_types: list[str] | None = None,
        hybrid_layers_types_block: list[str] | None = None,
        num_hybrid_layers_block: int = 21,
        last_layers_types: list[str] | None = None,
        sliding_window: int | None = 4096,
        use_sliding_window: bool = True,
        rope_theta: float = 1000000.0,
        swa_rope_theta: float = 10000.0,
        rotary_dim: int | None = 32,
        no_rope_layers: list[int] | None = None,
        enable_sink_attention: bool = True,
        use_over_encoding: bool = False,
        shared_kv_num_layers: int = 0,
        shared_kv_source_begin: int = 0,
        shared_kv_target_begin: int = 0,
        attn_out_scale: float = 1.0,
        ffn_out_scale: float = 0.53881590608,
        first_layer_attn_out_scale: float = 1.0,
        first_layer_ffn_out_scale: float = 1.0,
        softmax_scale: float | None = None,
        logit_scale: float = 1.0,
        moe_router_dtype: str = "fp32",
        enable_lm_head_fp32: bool = False,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.dense_intermediate_size = dense_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.mlp_only_layers = mlp_only_layers if mlp_only_layers is not None else [0]
        self.use_hybrid_layers = use_hybrid_layers
        self.first_layers_types = first_layers_types if first_layers_types is not None else ["full_attention"]
        self.hybrid_layers_types_block = (
            hybrid_layers_types_block
            if hybrid_layers_types_block is not None
            else ["full_attention", "sliding_attention", "sliding_attention", "sliding_attention"]
        )
        self.num_hybrid_layers_block = num_hybrid_layers_block
        self.last_layers_types = (
            last_layers_types
            if last_layers_types is not None
            else ["full_attention", "full_attention", "full_attention"]
        )
        self.sliding_window = sliding_window
        self.use_sliding_window = use_sliding_window
        self.rope_theta = rope_theta
        self.swa_rope_theta = swa_rope_theta
        self.rotary_dim = rotary_dim
        self.no_rope_layers = no_rope_layers if no_rope_layers is not None else []
        self.enable_sink_attention = enable_sink_attention
        self.use_over_encoding = use_over_encoding
        self.shared_kv_num_layers = shared_kv_num_layers
        self.shared_kv_source_begin = shared_kv_source_begin
        self.shared_kv_target_begin = shared_kv_target_begin
        self.attn_out_scale = attn_out_scale
        self.ffn_out_scale = ffn_out_scale
        self.first_layer_attn_out_scale = first_layer_attn_out_scale
        self.first_layer_ffn_out_scale = first_layer_ffn_out_scale
        self.softmax_scale = softmax_scale
        self.logit_scale = logit_scale
        self.moe_router_dtype = moe_router_dtype
        self.enable_lm_head_fp32 = enable_lm_head_fp32
        self.tie_word_embeddings = tie_word_embeddings
        super().__init__(**kwargs)


__all__ = ["IQuestQ1Config"]
