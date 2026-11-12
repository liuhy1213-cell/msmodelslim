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

Module/parameter names mirror the checkpoint 1:1 so that:
  - ``load_state_dict`` is a plain 1:1 copy from ``model.safetensors.index.json``;
  - ascendv1_saver writes output keys identical to the checkpoint names
    (e.g. ``model.layers.N.self_attn.q_proj.weight``,
    ``model.layers.N.mlp.experts.fc.<eid>.gate_proj.weight``);
  - LinearQuantProcessor quantizes exactly the modules the deployment graph
    expects (separate q/k/v/o projections, fused-stored experts, fp32 router).

The MoE experts are stored in the checkpoint as fused 3-D tensors
(``mlp.experts.fc`` [E, 2I, H] with gate rows first, ``mlp.experts.proj``
[E, H, I]).  They are exposed as per-expert ``nn.Linear`` submodules (so the
quantizer can deploy them) while the calibration forward computes all experts
with a stacked einsum over the original fused weights (kept as plain tensor
attributes, device-tracked via ``_apply`` so they follow the layer's NPU
residency and are dropped when the layer is offloaded).

Reference forward math: vllm-iquest-q1 ``src/vllm_iquest_q1/model.py``.
"""

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class IQuestQ1RMSNorm(nn.Module):
    """RMSNorm (T5-style): fp32 statistics, weight multiply, cast back."""

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


def _layer_type(config, layer_idx: int) -> str:
    """Resolve 'full_attention' / 'sliding_attention' for a layer index."""
    first = list(getattr(config, "first_layers_types", []) or [])
    block = list(getattr(config, "hybrid_layers_types_block", []) or [])
    last = list(getattr(config, "last_layers_types", []) or [])
    num_layers = config.num_hidden_layers
    if layer_idx < len(first):
        return first[layer_idx]
    last_start = num_layers - len(last)
    if layer_idx >= last_start:
        return last[layer_idx - last_start]
    pos = layer_idx - len(first)
    if block:
        return block[pos % len(block)]
    return "full_attention"


def _rotate_half_neox(x: torch.Tensor) -> torch.Tensor:
    """GPT-NeoX style rotate_half over the last dim: [-x2, x1]."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class IQuestQ1CalibAttention(nn.Module):
    """Eager GQA attention with learned sink key and FA3 hook points.

    FA3 placeholders (``fa_q`` / ``fa_k`` / ``fa_v``) are injected by the
    adapter via ``set_submodule``; the forward calls them when present, after
    q/k norm + RoPE (Q/K) and right after the projection (V), mirroring the
    vllm-ascend deployment graph.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.group = self.num_heads // self.num_kv_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.hidden_size = hidden_size
        self.scaling = self.head_dim ** -0.5
        self.eps = config.rms_norm_eps

        self.q_proj = nn.Linear(hidden_size, self.q_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.kv_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.kv_size, bias=False)
        self.o_proj = nn.Linear(self.q_size, hidden_size, bias=False)
        self.q_norm = IQuestQ1RMSNorm(self.head_dim, eps=self.eps)
        self.k_norm = IQuestQ1RMSNorm(self.head_dim, eps=self.eps)

        # Learned sink key, [num_kv_heads, head_dim] raw parameter (checkpoint
        # key ``self_attn.sink_k`` has no ".weight" suffix).
        self.sink_k = nn.Parameter(torch.zeros(self.num_kv_heads, self.head_dim), requires_grad=False)

        # RoPE: partial neox (rotary_dim / head_dim), theta depends on the
        # layer type (full vs sliding window).
        self.rotary_dim = getattr(config, "rotary_dim", self.head_dim)
        theta = config.rope_theta
        if _layer_type(config, layer_idx) == "sliding_attention":
            theta = getattr(config, "swa_rope_theta", theta)
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) / self.rotary_dim)
        )
        # Plain attribute on purpose: must NOT be registered as buffer/param so
        # it is never picked up by the saver or by .to(device) recursion.
        self._inv_freq = inv_freq

    # -- plain-attribute RoPE tables -------------------------------------

    def _rope_tables(self, seq_len: int, device, dtype):
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = t[:, None] * self._inv_freq.to(device)[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    @staticmethod
    def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """x: [T, H, D] (first ``rotary_dim`` channels rotated, neox style)."""
        rot_dim = cos.shape[-1]
        x_rot, x_pass = x[..., :rot_dim], x[..., rot_dim:]
        x_rot = x_rot * cos[:, None, :] + _rotate_half_neox(x_rot) * sin[:, None, :]
        return torch.cat((x_rot, x_pass), dim=-1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim)

        # Per-head QK norm before RoPE (vLLM IQuestQ1Attention.forward).
        q = self.q_norm(q)
        k = self.k_norm(k)

        cos, sin = self._rope_tables(seq_len, q.device, q.dtype)
        q = self._apply_rope(q, cos, sin)
        k = self._apply_rope(k, cos, sin)

        # [B, H, T, D] layout for FA3 branches and attention.
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # ===== FA3 placeholders (after norm/RoPE, before attention kernel) =====
        if hasattr(self, "fa_q"):
            q = self.fa_q(q)
        if hasattr(self, "fa_k"):
            k = self.fa_k(k)
        if hasattr(self, "fa_v"):
            v = self.fa_v(v)
        # =======================================================================

        # GQA: query head h belongs to kv head h // group (kv-major layout).
        k_full = k.repeat_interleave(self.group, dim=1)
        v_full = v.repeat_interleave(self.group, dim=1)

        scores = torch.matmul(q.float(), k_full.float().transpose(-2, -1)) * self.scaling

        # Learned sink: an always-visible entry with zero value; it only adds
        # one logit per query head.  Implemented as an extra softmax column.
        q_kv = q.float().view(bsz, self.num_kv_heads, self.group, seq_len, self.head_dim)
        sink_logit = (
            (q_kv * self.sink_k.float().view(1, self.num_kv_heads, 1, 1, self.head_dim)).sum(-1) * self.scaling
        )  # [B, KV, G, T]
        sink_logit = sink_logit.view(bsz, self.num_heads, seq_len, 1)
        probs = torch.softmax(torch.cat((scores, sink_logit), dim=-1), dim=-1)
        probs_v = probs[..., :seq_len]

        attn_output = torch.matmul(probs_v.to(v_full.dtype), v_full)  # [B, H, T, D]
        attn_output = attn_output.transpose(1, 2).reshape(bsz, seq_len, self.q_size)
        return self.o_proj(attn_output)


class IQuestQ1CalibDenseMLP(nn.Module):
    """Dense MLP (layer 0) with separate gate/up/down projections."""

    def __init__(self, config):
        super().__init__()
        inter = config.dense_intermediate_size
        hidden = config.hidden_size
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class IQuestQ1CalibMoE(nn.Module):
    """Top-k softmax MoE with checkpoint-shaped expert containers.

    Children (quantization surface, 1:1 with checkpoint keys):
        router                       -> nn.Linear (kept FLOAT)
        experts.fc.<eid>.gate_proj   -> nn.Linear (W8A8_MXFP8)
        experts.fc.<eid>.up_proj     -> nn.Linear (W8A8_MXFP8)
        experts.proj.<eid>.down_proj -> nn.Linear (W8A8_MXFP8)

    The fused checkpoint tensors are kept as plain attributes (``_gate3`` /
    ``_up3`` / ``_down3``) and the calibration forward runs all experts with a
    stacked einsum; the per-expert Linear modules are only consumed by the
    quantizer (data-free MXFP8 weight quantization at deploy time).
    """

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        hidden = config.hidden_size
        inter = config.intermediate_size
        self.hidden_size = hidden
        self.intermediate_size = inter

        self.router = nn.Linear(hidden, self.num_experts, bias=False)
        self.experts = nn.Module()
        self.experts.fc = nn.ModuleDict({str(e): nn.Module() for e in range(self.num_experts)})
        self.experts.proj = nn.ModuleDict({str(e): nn.Module() for e in range(self.num_experts)})
        for e in range(self.num_experts):
            fc = self.experts.fc[str(e)]
            fc.gate_proj = nn.Linear(hidden, inter, bias=False)
            fc.up_proj = nn.Linear(hidden, inter, bias=False)
            pj = self.experts.proj[str(e)]
            pj.down_proj = nn.Linear(inter, hidden, bias=False)

        # Fused weights captured by the adapter at load time (plain tensors).
        self._gate3: Optional[torch.Tensor] = None
        self._up3: Optional[torch.Tensor] = None
        self._down3: Optional[torch.Tensor] = None

    def capture_fused_weights(self, fc3d: torch.Tensor, proj3d: torch.Tensor) -> None:
        """Keep views into the fused checkpoint tensors for the einsum path.

        ``fc3d``: [E, 2I, H] (gate rows first), ``proj3d``: [E, H, I].
        """
        self._gate3 = fc3d[:, : self.intermediate_size, :]
        self._up3 = fc3d[:, self.intermediate_size:, :]
        self._down3 = proj3d

    def _apply(self, fn, recurse=True):
        ret = super()._apply(fn, recurse)
        # Track the fused weights with the module's device residency.  They are
        # only dropped when the module is offloaded to meta (parameter-less
        # shell); a plain ``.to(cpu)`` from LoadProcessor must keep them, since
        # the calibration forward (fa3_quant) runs *after* that load step and
        # reads ``_gate3/_up3/_down3`` instead of the per-expert Linear modules.
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

        # fp32 router (moe_router_dtype=float32), softmax + top-k renormalize.
        # moe_router_dtype=float32: both operands must be fp32 (the residual
        # stream is bf16 and the router weight stays unquantized bf16).
        logits = F.linear(x.to(torch.float32), self.router.weight.to(torch.float32))
        probs = torch.softmax(logits, dim=-1)
        top_w, top_e = torch.topk(probs, self.top_k, dim=-1)
        top_w = top_w / top_w.sum(dim=-1, keepdim=True)

        gate3 = self.__dict__.get("_gate3")
        up3 = self.__dict__.get("_up3")
        down3 = self.__dict__.get("_down3")
        if not (isinstance(gate3, torch.Tensor) and isinstance(up3, torch.Tensor)
                and isinstance(down3, torch.Tensor)):
            raise RuntimeError(
                "MoE fused weights are not available on the compute device; "
                "the layer must be loaded through the adapter's on-demand path."
            )

        # Top-k dispatch instead of a dense all-expert pass.  Mathematically
        # identical (unselected experts contribute zero weight), but 256/top_k
        # times cheaper -- the dense version cost ~78 min per CPU layer, which
        # made an 87-layer calibration run infeasible.
        flat_e = top_e.reshape(-1)                                # [T*K]
        flat_w = top_w.reshape(-1).to(torch.float32)              # [T*K]
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
                start += n
                continue
            idx = sorted_tok[start:start + n]
            rows = x[idx].to(gate3.dtype)                              # [n, H]
            act = F.silu(rows @ gate3[e].t()) * (rows @ up3[e].t())    # [n, I]
            expert_out = act @ down3[e].t()                            # [n, H]
            final.index_add_(0, idx, expert_out.float() * sorted_w[start:start + n].unsqueeze(-1))
            start += n
        return final.to(x.dtype).view(orig_shape)


class IQuestQ1CalibDecoderLayer(nn.Module):
    """IQuest-Q1 decoder layer.

    Layer 0 uses a sandwich structure (extra ``ffn_out_norm`` on the MLP
    output); layers 1+ add the raw MLP output scaled by ``ffn_out_scale``,
    and the residual stream after attention is the *normed* input ``x``.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        hidden = config.hidden_size
        eps = config.rms_norm_eps

        self.self_attn = IQuestQ1CalibAttention(config, layer_idx)
        self.use_sandwich_norm = layer_idx == 0

        if layer_idx in (getattr(config, "mlp_only_layers", None) or []):
            self.mlp = IQuestQ1CalibDenseMLP(config)
        else:
            self.mlp = IQuestQ1CalibMoE(config)

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

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.use_sandwich_norm:
            norm_hidden_states = self.attention_norm(hidden_states)
            attn_output = self.self_attn(norm_hidden_states)
            h = hidden_states + self.attn_out_norm(attn_output) * self.attn_out_scale

            ffn_out = self.mlp(self.feed_forward_norm(h))
            return h + self.ffn_out_norm(ffn_out) * self.ffn_out_scale

        x = self.attention_norm(hidden_states)
        attn_output = self.self_attn(x)
        h = x + self.attn_out_norm(attn_output) * self.attn_out_scale

        ffn_out = self.mlp(self.feed_forward_norm(h))
        return h + ffn_out * self.ffn_out_scale


class IQuestQ1CalibInner(nn.Module):
    """Decoder body: embedding / decoder layers / final norm.

    对应 checkpoint 里的 ``model.*`` 命名空间。必须真实嵌在 ``IQuestQ1CalibModel.model``
    下（而不是平铺在根上），否则 ``LinearQuantProcessor`` 之类的处理器用
    ``set_submodule("model.layers.N....")`` 在根模型上定位子模块时会报
    ``has no attribute `model```。
    """

    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList()
        self.norm = IQuestQ1RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class IQuestQ1CalibModel(nn.Module):
    """Shell model (``ForCausalLM`` 形态): ``model.*`` + ``lm_head``。

    embedding / final norm / lm_head 由适配器在 ``init_model`` 里加载真实权重；
    decoder 层按需追加（layer-wise 流式）。
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = IQuestQ1CalibInner(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    # 便捷访问（适配器内部使用）
    @property
    def embed_tokens(self):
        return self.model.embed_tokens

    @property
    def layers(self):
        return self.model.layers

    @property
    def norm(self):
        return self.model.norm
