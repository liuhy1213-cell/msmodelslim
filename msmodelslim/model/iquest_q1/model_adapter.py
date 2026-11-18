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

"""modelslim_v1 适配器：IQuest-Q1（MoE + learned sink + 混合滑窗）。

结构和 ``msmodelslim/model/deepseek_v4`` 同一套：

  1. ``init_model`` 只建 **1 层** 的壳（layer0 即模板），用
     ``get_state_dict`` 载入 embed / norm / lm_head / layer0 的真实权重；
  2. ``generate_decoder_layer`` 按需构造并加载第 N 层，visit 与 forward 共用
     同一个生成器 —— visit 和 forward 的顺序天然一致；
  3. ``generate_model_forward`` 先跑一次真实前向，用 pre-hook 在 layer0 入口
     截住输入（抛 ``TransformersForwardBreak``），再逐层 yield 真实的
     ``(positions, hidden_states)``。前向就是模型自己的 forward，不存在
     "标定一份实现、部署另一份实现" 的风险。

与 deepseek_v4 的差异只有一处：IQuest-Q1 的 MoE 专家在 checkpoint 里是融合
张量（``mlp.experts.fc`` [E,2I,H] / ``mlp.experts.proj`` [E,H,I]），而量化器
要看到纯线性层，所以这层加载时把融合张量展开成逐专家 ``nn.Linear``，同时把
融合张量留给前向 einsum 用（见 ``IQuestQ1MoE.capture_fused_weights``）。

保存阶段仍由 ``ExpertFuser`` 把逐专家张量融回 vLLM-ascend 的
``routed_experts`` 布局，一条命令直接产出可部署权重。
"""

import os
from collections import defaultdict
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple
from unittest.mock import patch

import torch
from safetensors import safe_open
from torch import nn
from tqdm import tqdm

from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.core.const import DeviceType
from msmodelslim.core.quant_service.modelslim_v1.save.interface import AscendV1SaveTransformInterface
from msmodelslim.model.common.layer_wise_forward import (
    TransformersForwardBreak,
    generated_decoder_layer_visit_func,
)
from msmodelslim.model.common.transformers import TransformersModel
from msmodelslim.model.common.weight_helper import get_state_dict
from msmodelslim.model.interface_hub import (
    ModelInfoInterface,
    ModelSlimPipelineInterfaceV1,
    FA3QuantAdapterInterface,
    FA3QuantPlaceHolder,
)
from msmodelslim.utils.exception import InvalidModelError
from msmodelslim.utils.logging import logger_setter, get_logger
from msmodelslim.utils.security import get_valid_read_path, json_safe_load, MAX_READ_FILE_SIZE_512G

from .expert_fusion import ExpertFuser
from .iquest_q1_model import (
    IQuestQ1Attention,
    IQuestQ1DecoderLayer,
    IQuestQ1MoE,
    IQuestQ1Model,
    layer_types,
)

# 标定序列长度上限。qwen3_cot_w4a4.json 最长 2661 token，4096 等于不截断；
# 换成更长的标定集时可用 IQUEST_CALIB_SEQ_LEN 收窄（CPU 上 T^2 很贵）。
MAX_CALIB_SEQ_LEN = int(os.environ.get("IQUEST_CALIB_SEQ_LEN", "4096"))


@contextmanager
def default_dtype(dtype):
    """临时改 torch 默认 dtype（建模块用），退出时还原。"""
    original = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(original)


@logger_setter()
class IQuestQ1ModelAdapter(
    TransformersModel,
    ModelInfoInterface,
    ModelSlimPipelineInterfaceV1,
    FA3QuantAdapterInterface,
    AscendV1SaveTransformInterface,
):
    def __init__(self, model_type: str, model_path: Path, trust_remote_code: bool = False):
        super().__init__(model_type, model_path, trust_remote_code)
        self._expert_fuser = ExpertFuser(self.config.num_experts)
        types = layer_types(self.config)
        get_logger().info(
            "IQuest-Q1: %d layers (%d full / %d sliding, window=%s), %d experts top-%d",
            len(types),
            types.count("full_attention"),
            types.count("sliding_attention"),
            getattr(self.config, "sliding_window", None),
            self.config.num_experts,
            self.config.num_experts_per_tok,
        )

    # ===== ModelInfoInterface =====

    def get_model_pedigree(self) -> str:
        return "iquest_q1"

    def get_model_type(self) -> str:
        return self.model_type

    def get_global_model_torch_dtype(self) -> torch.dtype:
        # config.json 的 dtype 是 bfloat16；checkpoint 也是 bf16。
        return torch.bfloat16

    def _load_tokenizer(self, trust_remote_code: bool = False):
        """标定集 tokenize 时 padding=True，必须有 pad_token。"""
        tokenizer = super()._load_tokenizer(trust_remote_code=trust_remote_code)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer

    # ===== checkpoint 读取 =====

    @lru_cache(maxsize=1)
    def _get_weight_map(self) -> Dict[str, str]:
        return json_safe_load(os.path.join(str(self.model_path), "model.safetensors.index.json"))["weight_map"]

    def _read_checkpoint_tensors(self, names: List[str], desc: str) -> Dict[str, torch.Tensor]:
        weight_map = self._get_weight_map()
        groups: Dict[str, List[str]] = defaultdict(list)
        for name in names:
            if name not in weight_map:
                raise InvalidModelError(
                    f"checkpoint 里没有 {name}",
                    action="请确认模型目录与 IQuest-Q1 架构一致。",
                )
            groups[weight_map[name]].append(name)

        tensors: Dict[str, torch.Tensor] = {}
        for file_name, keys in tqdm(groups.items(), desc=desc, leave=False):
            path = get_valid_read_path(
                os.path.join(str(self.model_path), file_name),
                extensions="safetensors",
                size_max=MAX_READ_FILE_SIZE_512G,
            )
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in keys:
                    tensors[key] = f.get_tensor(key)
        return tensors

    def _layer_state_dict(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Optional[Tuple]]:
        """第 idx 层的 state_dict（名字相对层）+ 融合专家张量（MoE 层才有）。"""
        prefix = f"model.layers.{idx}"
        weight_map = self._get_weight_map()
        ckpt_keys = {k for k in weight_map if k.startswith(prefix + ".")}

        fused_fc = f"{prefix}.mlp.experts.fc"
        fused_proj = f"{prefix}.mlp.experts.proj"
        fused_keys = {fused_fc, fused_proj} & ckpt_keys

        state_dict: Dict[str, torch.Tensor] = {}
        fc3d = proj3d = None
        if fused_keys:
            inter = self.config.intermediate_size
            fused = self._read_checkpoint_tensors(sorted(fused_keys), f"layer {idx} experts")
            fc3d, proj3d = fused[fused_fc], fused[fused_proj]
            want = (self.config.num_experts, 2 * inter, self.config.hidden_size)
            if tuple(fc3d.shape) != want:
                raise InvalidModelError(
                    f"{fused_fc} 形状 {tuple(fc3d.shape)} != {want}",
                    action="请检查 checkpoint 的 MoE 融合布局（gate 行在前）。",
                )
            for e in range(self.config.num_experts):
                state_dict[f"mlp.experts.fc.{e}.gate_proj.weight"] = fc3d[e, :inter, :]
                state_dict[f"mlp.experts.fc.{e}.up_proj.weight"] = fc3d[e, inter:, :]
                state_dict[f"mlp.experts.proj.{e}.down_proj.weight"] = proj3d[e]

        plain = sorted(ckpt_keys - fused_keys)
        for key, tensor in self._read_checkpoint_tensors(plain, f"layer {idx}").items():
            state_dict[key[len(prefix) + 1 :]] = tensor
        return state_dict, (fc3d, proj3d)

    # ===== 建层 / 载层 =====

    def load_decoder_if_not_exist(self, model: nn.Module, idx: int) -> nn.Module:
        name = f"model.layers.{idx}"
        try:
            return model.get_submodule(name)
        except AttributeError:
            pass

        get_logger().info("Creating decoder layer %s", idx)
        # reset_parameters 会把 256 个专家的 Linear 全初始化一遍，纯浪费
        with torch.device("cpu"), default_dtype(torch.bfloat16):
            with patch.object(nn.Linear, "reset_parameters", lambda _self: None):
                layer = IQuestQ1DecoderLayer(self.config, idx)

        state_dict, fused = self._layer_state_dict(idx)
        layer.load_state_dict(state_dict, strict=True)
        layer.eval()
        if isinstance(layer.mlp, IQuestQ1MoE):
            if fused[0] is None:
                raise InvalidModelError(
                    f"layer {idx} 是 MoE 层但 checkpoint 里没有融合专家张量",
                    action="请确认 checkpoint 含 mlp.experts.fc / mlp.experts.proj。",
                )
            layer.mlp.capture_fused_weights(*fused)
        model.model.layers.append(layer)
        get_logger().info("Decoder layer %s loaded", idx)
        return layer

    def generate_decoder_layer(self, model: nn.Module) -> Generator[Tuple[str, nn.Module], None, None]:
        for idx in range(self._visit_limit()):
            yield f"model.layers.{idx}", self.load_decoder_if_not_exist(model, idx)

    def _visit_limit(self) -> int:
        """层数。``IQUEST_MAX_LAYERS`` 只用于开发期冒烟（真实量化必须不设）。"""
        limit = int(os.environ.get("IQUEST_MAX_LAYERS", "0") or "0")
        if limit <= 0:
            return self.config.num_hidden_layers
        get_logger().warning("IQUEST_MAX_LAYERS=%d：只量化前 %d 层，产物不完整", limit, limit)
        return min(limit, self.config.num_hidden_layers)

    # ===== ModelSlimPipelineInterfaceV1 =====

    def handle_dataset(self, dataset: Any, device: DeviceType = DeviceType.NPU) -> List[Any]:
        """tokenize 标定集，每条返回 ``[input_ids, attention_mask]``。"""
        dev = "npu" if device is DeviceType.NPU else "cpu"
        processed: List[Any] = []
        for item in tqdm(dataset, desc="Processing calibration dataset"):
            if isinstance(item, dict):
                # lab_calib 的 jsonl 用 inputs_pretokenized，json 是裸字符串列表
                text = item.get("text") or item.get("prompt") or item.get("inputs_pretokenized")
                if text is None:
                    text = str(item)
            else:
                text = str(item)
            enc = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_CALIB_SEQ_LEN,
            )
            if enc["input_ids"].shape[1] < 8:
                get_logger().warning("Skip calibration sample shorter than 8 tokens: %r", text[:32])
                continue
            processed.append([enc["input_ids"].to(dev), enc["attention_mask"].to(dev)])
        if not processed:
            raise InvalidModelError(
                "标定集为空或全部太短。",
                action="请提供非空的标定数据。",
            )
        get_logger().info(
            "Prepared %d calibration samples (device=%s, max_len=%d)", len(processed), dev, MAX_CALIB_SEQ_LEN
        )
        return processed

    def init_model(self, device: DeviceType = DeviceType.NPU) -> nn.Module:
        """只建 1 层壳，其余层按需 append。"""
        origin = self.config.num_hidden_layers
        get_logger().info("Initializing IQuest-Q1 shell with 1 template layer (%d total)", origin)
        try:
            self.config.num_hidden_layers = 1
            with torch.device("cpu"), default_dtype(torch.bfloat16):
                with patch.object(nn.Linear, "reset_parameters", lambda _self: None):
                    model = IQuestQ1Model(self.config)
        finally:
            self.config.num_hidden_layers = origin

        state_dict = get_state_dict(self.model_path, model)
        missing = set(dict(model.named_parameters())) - set(state_dict)
        if missing:
            raise InvalidModelError(
                f"壳模型有参数没在 checkpoint 里找到: {sorted(missing)[:5]}",
                action="请检查 config.json 与权重是否匹配。",
            )
        model.load_state_dict(state_dict)
        model.eval()

        embed_dtype = model.embed_tokens.weight.dtype
        if embed_dtype != torch.bfloat16:
            raise InvalidModelError(
                f"壳模型 dtype 必须是 bf16，实际 {embed_dtype}",
                action="请保持 init_model 与 get_global_model_torch_dtype 一致。",
            )
        get_logger().info("Shell ready (dtype=%s, layers appended on demand)", embed_dtype)
        return model

    def generate_model_visit(self, model: nn.Module) -> Generator[ProcessRequest, Any, None]:
        return generated_decoder_layer_visit_func(
            model, transformer_blocks=self.generate_decoder_layer(model)
        )

    def generate_model_forward(self, model: nn.Module, inputs: Any) -> Generator[ProcessRequest, Any, None]:
        """跑一次真实前向截住 layer0 的入参，再逐层 yield。"""
        if isinstance(inputs, dict):
            input_ids = inputs["input_ids"]
        elif isinstance(inputs, (list, tuple)):
            input_ids = inputs[0]
        else:
            input_ids = inputs

        first_layer = model.model.layers[0]
        captured: Dict[str, Any] = {}

        def break_hook(module, hook_args, hook_kwargs):
            captured["args"] = hook_args
            captured["kwargs"] = hook_kwargs
            raise TransformersForwardBreak()

        handle = first_layer.register_forward_pre_hook(break_hook, with_kwargs=True, prepend=True)
        try:
            model(input_ids)
        except TransformersForwardBreak:
            pass
        finally:
            handle.remove()

        if "args" not in captured:
            raise InvalidModelError(
                "没截到 layer0 的输入",
                action="请检查 IQuestQ1Model.forward 是否会依次调用 self.model.layers。",
            )

        positions, hidden_states = captured["args"]
        for name, layer in self.generate_decoder_layer(model):
            hidden_states = yield ProcessRequest(name, layer, (positions, hidden_states), {})

    def enable_kv_cache(self, model: nn.Module, need_kv_cache: bool) -> None:
        # 标定前向是整条序列的 eager 注意力，没有 KV cache；只同步配置位。
        model.model.config.use_cache = need_kv_cache
        get_logger().info("KV cache %s", "enabled" if need_kv_cache else "disabled")

    # ===== FA3QuantAdapterInterface =====

    def inject_fa3_placeholders(
        self, root_name: str, root_module: nn.Module, should_inject: Callable[[str], bool]
    ) -> None:
        for name, module in root_module.named_modules():
            if not isinstance(module, IQuestQ1Attention):
                continue
            full_name = f"{root_name}.{name}" if root_name else name
            if not should_inject(full_name) or hasattr(module, "fa_q"):
                continue
            prefix = f"{name}." if name else ""
            root_module.set_submodule(f"{prefix}fa_q", FA3QuantPlaceHolder(ratio=1.0))
            root_module.set_submodule(f"{prefix}fa_k", FA3QuantPlaceHolder(ratio=1.0))
            root_module.set_submodule(f"{prefix}fa_v", FA3QuantPlaceHolder(ratio=1.0))
            get_logger().info("Injected FA3 placeholders into %s", full_name)

    # ===== AscendV1SaveTransformInterface =====

    def ascendv1_transform_save_entry(
        self, prefix: str, desc: str, tensor: torch.Tensor, write_tensor: Callable, write_desc: Callable
    ) -> bool:
        """逐专家张量先缓存，整层到齐后按 routed_experts 布局融合写出。"""
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        if world_size > 1:
            raise InvalidModelError(
                "IQuest-Q1 的 routed_experts 融合要求同一进程看到全部专家，"
                f"当前 world_size={world_size}。",
                action="请用单进程 layer-wise 保存（不要开 DP/EP 分布）。",
            )
        return self._expert_fuser.consume(prefix, desc, tensor, write_tensor, write_desc)

    def ascendv1_flush_save_entries(self, write_tensor: Callable, write_desc: Callable) -> None:
        self._expert_fuser.flush(write_tensor, write_desc)
