# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

MindStudio is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

"""Pure-torch calibration model for IQuest-Q1 (W8A8C8-MXFP8 quantization).

逐行对齐上游推理实现：
``https://github.com/IQuestLab/vllm-iquest-q1/blob/main/src/vllm_iquest_q1/model.py``

对齐点（写错任何一条，标定统计就是错的）：
  * 层结构：layer0 是 sandwich norm（多一个 ``ffn_out_norm``），其余层 residual
    走 ``x + attn_out_norm(attn) * attn_out_scale`` / ``h + ffn_out * ffn_out_scale``；
  * 注意力是 **causal**，滑窗层再叠加 ``sliding_window``（含当前 token），
    和 vLLM 的 ``per_layer_sliding_window`` 同口径；
  * QK 都是 per-head RMSNorm 之后做 **partial neox RoPE**（rotary_dim=32），
    滑窗层用 ``swa_rope_theta``；
  * learned sink 是 softmax 的额外一列，永远可见；
  * MoE 用 fp32 router（``moe_router_dtype=fp32``）+ softmax + top-k 重归一化。

模块/参数名与 checkpoint 1:1（``model.layers.N.*`` / ``lm_head``），这样
``load_state_dict`` 是直接拷贝，ascendv1_saver 写出的 key 也和部署一致。

MoE 专家在 checkpoint 里是融合张量（``mlp.experts.fc`` [E,2I,H] /
``mlp.experts.proj`` [E,H,I]），这里既展开成逐专家 ``nn.Linear``（量化器要看到
纯线性层），又把融合张量作为普通属性留一份给前向 einsum 用。
"""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


UNSUPPORTED_SHARED_KV = "shared_kv_num_layers>0 未适配：该层没有 k_norm 且 KV 复用源层"


def get_layer_sliding_window_size(config, layer_idx: int) -> Optional[int]:
    """与上游 ``get_layer_sliding_window_size`` 完全一致。

    first_layers_types -> hybrid_layers_types_block 重复 num_hybrid_layers_block 次
    -> last_layers_types。
    """
    if not getattr(config, "use_hybrid_layers", False):
        return None
    first = list(getattr(config, "first_layers_types", []) or [])
    block = list(getattr(config, "hybrid_layers_types_block", []) or [])
    last = list(getattr(config, "last_layers_types", []) or [])
    num_layers = config.num_hidden_layers
    num_first = len(first)
    num_blocks = getattr(config, "num_hybrid_layers_block", 0) or 0
    num_block_layers = len(block) * num_blocks

    if layer_idx < num_first:
        layer_type = first[layer_idx]
    elif layer_idx < num_first + num_block_layers:
        layer_type = block[(layer_idx - num_first) % len(block)]
    else:
        offset = layer_idx - num_first - num_block_layers
        if offset >= len(last):
            raise ValueError(
                f"layer {layer_idx} 落在 first+block 之外且 last_layers_types 不够长 "
                f"(num_hidden_layers={num_layers}, first={num_first}, "
                f"block={num_block_layers}, last={len(last)})"
            )
        layer_type = last[offset]

    if layer_type == "full_attention":
        return None
    return int(config.sliding_window)


def layer_types(config):
    """[str] * num_hidden_layers，用于日志/自检。"""
    return [
        "sliding_attention" if get_layer_sliding_window_size(config, i) else "full_attention"
        for i in range(config.num_hidden_layers)
    ]


def _rotate_half_neox(x: torch.Tensor) -> torch.Tensor:
    """GPT-NeoX rotate_half: [-x2, x1]."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class IQuestQ1RMSNorm(nn.Module):
    """RMSNorm (T5 口径): fp32 统计 + weight，最后转回输入 dtype。"""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class IQuestQ1Attention(nn.Module):
    """GQA + learned sink + 因果/滑窗 mask + FA3 挂点。

    FA3 占位模块（``fa_q`` / ``fa_k`` / ``fa_v``）由适配器注入，调用位置和部署图
    一致：Q/K 在 qk-norm + RoPE 之后，V 在投影之后。
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        if getattr(config, "shared_kv_num_layers", 0):
            raise NotImplementedError(f"layer {layer_idx}: {UNSUPPORTED_SHARED_KV}")

        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.group = self.num_heads // self.num_kv_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.hidden_size = config.hidden_size
        self.eps = config.rms_norm_eps

        softmax_scale = getattr(config, "softmax_scale", None)
        self.scaling = float(softmax_scale) if softmax_scale else self.head_dim ** -0.5

        # 滑窗：None 表示 full attention（因果但对整条可见）。
        self.sliding_window = get_layer_sliding_window_size(config, layer_idx)

        self.q_proj = nn.Linear(self.hidden_size, self.q_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_size, bias=False)
        self.o_proj = nn.Linear(self.q_size, self.hidden_size, bias=False)
        self.q_norm = IQuestQ1RMSNorm(self.head_dim, eps=self.eps)
        self.k_norm = IQuestQ1RMSNorm(self.head_dim, eps=self.eps)

        self.enable_sink_attention = getattr(config, "enable_sink_attention", False)
        if self.enable_sink_attention:
            # checkpoint key 是 ``self_attn.sink_k``（没有 .weight 后缀）
            self.sink_k = nn.Parameter(
                torch.zeros(self.num_kv_heads, self.head_dim), requires_grad=False
            )

        # partial neox RoPE：只转前 rotary_dim 个通道
        no_rope = getattr(config, "no_rope_layers", None) or []
        if layer_idx in no_rope:
            raise NotImplementedError(f"layer {layer_idx}: no_rope_layers 未适配")
        self.rotary_dim = int(getattr(config, "rotary_dim", self.head_dim))
        self.rope_theta = float(config.rope_theta)
        if self.sliding_window is not None:
            self.rope_theta = float(getattr(config, "swa_rope_theta", self.rope_theta))
        inv_freq = 1.0 / (
            self.rope_theta
            ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) / self.rotary_dim)
        )
        # 普通属性：不能被 saver / .to() 当成 buffer 或 parameter
        self._inv_freq = inv_freq

    # -- RoPE -------------------------------------------------------------

    def _rope_cos_sin(self, positions: torch.Tensor, dtype):
        freqs = positions.to(torch.float32)[:, None] * self._inv_freq.to(positions.device)[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)  # neox: 前后半相同
        return emb.cos().to(dtype), emb.sin().to(dtype)

    @staticmethod
    def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """x: [T, H, D]；只旋转前 ``rotary_dim`` 个通道。"""
        rot_dim = cos.shape[-1]
        x_rot, x_pass = x[..., :rot_dim], x[..., rot_dim:]
        x_rot = x_rot * cos[:, None, :] + _rotate_half_neox(x_rot) * sin[:, None, :]
        return torch.cat((x_rot, x_pass), dim=-1)

    def _attn_mask(self, positions: torch.Tensor, dtype) -> torch.Tensor:
        """[T, T] bool：True 表示可见（causal + 可选滑窗，含当前 token）。"""
        q_pos = positions[:, None]
        k_pos = positions[None, :]
        keep = k_pos <= q_pos
        if self.sliding_window is not None:
            keep = keep & (k_pos > (q_pos - self.sliding_window))
        return keep

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim)

        # per-head QK norm，在 RoPE 之前（上游 IQuestQ1Attention.forward）
        q = self.q_norm(q)
        k = self.k_norm(k)

        cos, sin = self._rope_cos_sin(positions, q.dtype)
        q = self._apply_rope(q, cos, sin)
        k = self._apply_rope(k, cos, sin)

        # [B, H, T, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # ===== FA3 挂点（qk-norm/RoPE 之后，attention kernel 之前）=====
        if hasattr(self, "fa_q"):
            q = self.fa_q(q)
        if hasattr(self, "fa_k"):
            k = self.fa_k(k)
        if hasattr(self, "fa_v"):
            v = self.fa_v(v)
        # ===============================================================

        # GQA：query head h 属于 kv head h // group（kv-major 布局）
        k_full = k.repeat_interleave(self.group, dim=1)
        v_full = v.repeat_interleave(self.group, dim=1)

        scores = torch.matmul(q.float(), k_full.float().transpose(-2, -1)) * self.scaling
        keep = self._attn_mask(positions, scores.dtype)
        scores = scores.masked_fill(~keep, float("-inf"))

        probs = torch.softmax(scores, dim=-1)

        if self.enable_sink_attention:
            # sink 是额外一列（value 为 0），永远可见，只贡献一个 logit
            q_kv = q.float().view(bsz, self.num_kv_heads, self.group, seq_len, self.head_dim)
            sink_logit = (
                (q_kv * self.sink_k.float().view(1, self.num_kv_heads, 1, 1, self.head_dim)).sum(-1)
                * self.scaling
            )  # [B, KV, G, T] -> [B, H, T]
            sink_logit = sink_logit.view(bsz, self.num_heads, seq_len, 1)
            probs = torch.softmax(torch.cat((scores, sink_logit), dim=-1), dim=-1)[..., :seq_len]

        attn_output = torch.matmul(probs.to(v_full.dtype), v_full)  # [B, H, T, D]
        attn_output = attn_output.transpose(1, 2).reshape(bsz, seq_len, self.q_size)
        return self.o_proj(attn_output)


class IQuestQ1DenseMLP(nn.Module):
    """稠密 MLP（layer 0）：silu(gate) * up -> down。"""

    def __init__(self, config):
        super().__init__()
        hidden = config.hidden_size
        inter = config.dense_intermediate_size
        if getattr(config, "hidden_act", "silu") != "silu":
            raise ValueError(f"unsupported hidden_act {config.hidden_act}")
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class IQuestQ1MoE(nn.Module):
    """top-k softmax 路由的 MoE。

    量化面（与 checkpoint key 同名）：
        router                        -> nn.Linear（不量化）
        experts.fc.<eid>.gate_proj    -> nn.Linear（W8A8_MXFP8）
        experts.fc.<eid>.up_proj      -> nn.Linear
        experts.proj.<eid>.down_proj  -> nn.Linear

    前向走 top-k dispatch（unselected 专家权重为 0，结果与全专家一致，但快
    256/top_k 倍）。融合张量 ``_gate3/_up3/_down3`` 由适配器在加载时捕获。
    """

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.router = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        self.experts = nn.Module()
        self.experts.fc = nn.ModuleDict({str(e): nn.Module() for e in range(self.num_experts)})
        self.experts.proj = nn.ModuleDict({str(e): nn.Module() for e in range(self.num_experts)})
        for e in range(self.num_experts):
            fc = self.experts.fc[str(e)]
            fc.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
            fc.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
            self.experts.proj[str(e)].down_proj = nn.Linear(
                self.intermediate_size, self.hidden_size, bias=False
            )

        self._gate3: Optional[torch.Tensor] = None
        self._up3: Optional[torch.Tensor] = None
        self._down3: Optional[torch.Tensor] = None

    def capture_fused_weights(self, fc3d: torch.Tensor, proj3d: torch.Tensor) -> None:
        """fc3d: [E, 2I, H]（gate 在前），proj3d: [E, H, I]。"""
        self._gate3 = fc3d[:, : self.intermediate_size, :]
        self._up3 = fc3d[:, self.intermediate_size :, :]
        self._down3 = proj3d

    def _apply(self, fn, recurse=True):
        ret = super()._apply(fn, recurse)
        # 融合权重跟随模块的设备/offload 生命周期；只有被搬到 meta（卸载）时才丢弃，
        # 普通 .to(cpu) 必须保留——标定前向在 LoadProcessor 之后才跑。
        for name in ("_gate3", "_up3", "_down3"):
            tensor = self.__dict__.get(name)
            if isinstance(tensor, torch.Tensor):
                try:
                    moved = fn(tensor)
                except Exception:
                    moved = None
                if moved is None or moved.device.type == "meta":
                    self.__dict__.pop(name, None)
                else:
                    self.__dict__[name] = moved
        return ret

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        x = hidden_states.view(-1, self.hidden_size)
        num_tokens = x.shape[0]
        device = x.device

        # fp32 router + softmax + top-k 重归一化（renormalize=True）
        logits = F.linear(x.to(torch.float32), self.router.weight.to(torch.float32))
        probs = torch.softmax(logits, dim=-1)
        top_w, top_e = torch.topk(probs, self.top_k, dim=-1)
        top_w = top_w / top_w.sum(dim=-1, keepdim=True)

        gate3 = self.__dict__.get("_gate3")
        up3 = self.__dict__.get("_up3")
        down3 = self.__dict__.get("_down3")
        if not (
            isinstance(gate3, torch.Tensor)
            and isinstance(up3, torch.Tensor)
            and isinstance(down3, torch.Tensor)
        ):
            raise RuntimeError(
                "MoE 融合权重不在计算设备上；该层必须走适配器的按层加载路径。"
            )

        flat_e = top_e.reshape(-1)
        flat_w = top_w.reshape(-1).to(torch.float32)
        flat_tok = torch.arange(num_tokens, device=device).repeat_interleave(self.top_k)
        order = torch.argsort(flat_e, stable=True)
        sorted_e = flat_e[order]
        sorted_tok = flat_tok[order]
        sorted_w = flat_w[order]
        counts = torch.bincount(sorted_e, minlength=self.num_experts)

        final = torch.zeros(num_tokens, self.hidden_size, dtype=torch.float32, device=device)
        start = 0
        for e in range(self.num_experts):
            n = int(counts[e].item())
            if n == 0:
                continue
            idx = sorted_tok[start : start + n]
            rows = x[idx].to(gate3.dtype)
            act = F.silu(rows @ gate3[e].t()) * (rows @ up3[e].t())
            final.index_add_(0, idx, (act @ down3[e].t()).float() * sorted_w[start : start + n].unsqueeze(-1))
            start += n
        return final.to(x.dtype).view(orig_shape)


class IQuestQ1DecoderLayer(nn.Module):
    """IQuest-Q1 decoder layer，残差结构与上游逐字一致。"""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        hidden = config.hidden_size
        eps = config.rms_norm_eps

        self.self_attn = IQuestQ1Attention(config, layer_idx)
        self.use_sandwich_norm = layer_idx == 0

        if layer_idx in (getattr(config, "mlp_only_layers", None) or []):
            self.mlp = IQuestQ1DenseMLP(config)
        else:
            self.mlp = IQuestQ1MoE(config)

        self.attention_norm = IQuestQ1RMSNorm(hidden, eps=eps)
        self.attn_out_norm = IQuestQ1RMSNorm(hidden, eps=eps)
        self.feed_forward_norm = IQuestQ1RMSNorm(hidden, eps=eps)
        if self.use_sandwich_norm:
            self.ffn_out_norm = IQuestQ1RMSNorm(hidden, eps=eps)

        if layer_idx == 0:
            self.attn_out_scale = getattr(config, "first_layer_attn_out_scale", 1.0)
            self.ffn_out_scale = getattr(config, "first_layer_ffn_out_scale", 1.0)
        else:
            self.attn_out_scale = getattr(config, "attn_out_scale", 1.0)
            self.ffn_out_scale = getattr(config, "ffn_out_scale", 1.0)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.use_sandwich_norm:
            norm_hidden_states = self.attention_norm(hidden_states)
            attn_output = self.self_attn(positions, norm_hidden_states)
            h = hidden_states + self.attn_out_norm(attn_output) * self.attn_out_scale
            ffn_out = self.mlp(self.feed_forward_norm(h))
            return h + self.ffn_out_norm(ffn_out) * self.ffn_out_scale

        x = self.attention_norm(hidden_states)
        attn_output = self.self_attn(positions, x)
        h = x + self.attn_out_norm(attn_output) * self.attn_out_scale
        ffn_out = self.mlp(self.feed_forward_norm(h))
        return h + ffn_out * self.ffn_out_scale


class IQuestQ1Inner(nn.Module):
    """checkpoint 的 ``model.*`` 命名空间：embed / layers / 最终 norm。

    必须真的挂在 ``IQuestQ1Model.model`` 下面（不能平铺到根），否则
    LinearQuantProcessor 之类用 ``set_submodule("model.layers.N...")`` 定位子模块
    的处理器会报 "has no attribute `model`"。
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        # init_model 里 config.num_hidden_layers 被临时改成 1：layer0 当模板，
        # 其余层由适配器 generate_decoder_layer 按需 append。
        self.layers = nn.ModuleList(
            IQuestQ1DecoderLayer(config, i) for i in range(config.num_hidden_layers)
        )
        self.norm = IQuestQ1RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, inputs_embeds: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(positions, hidden_states)
        return hidden_states


class IQuestQ1Model(nn.Module):
    """``ForCausalLM`` 形态的标定模型：``model.*`` + ``lm_head``。

    ``init_model`` 只建一层（layer 0 当模板，同时把 embed/norm/lm_head 的真实权重
    载进来）；其余层由 ``generate_decoder_layer`` 按需构造并 append。
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = IQuestQ1Inner(config)
        if getattr(config, "tie_word_embeddings", False):
            raise NotImplementedError("tie_word_embeddings 未适配（IQuest-Q1 是 False）")
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    # 便捷访问
    @property
    def embed_tokens(self):
        return self.model.embed_tokens

    @property
    def layers(self):
        return self.model.layers

    @property
    def norm(self):
        return self.model.norm

    def forward(self, input_ids: torch.Tensor, positions: Optional[torch.Tensor] = None):
        if positions is None:
            positions = torch.arange(input_ids.shape[-1], device=input_ids.device)
        hidden_states = self.model.embed_tokens(input_ids)
        hidden_states = self.model(hidden_states, positions)
        return self.model.norm(hidden_states)
