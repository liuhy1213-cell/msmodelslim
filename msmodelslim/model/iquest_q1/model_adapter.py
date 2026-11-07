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

import os
from collections import defaultdict
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Set
from unittest.mock import patch

import torch
from safetensors import safe_open
from torch import nn
from tqdm import tqdm
from transformers import AutoTokenizer  # pylint: disable=no-name-in-module

from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.core.const import DeviceType
from msmodelslim.model.common.vlm_base import VLMBaseModelAdapter
from msmodelslim.model.interface_hub import (
    ModelInfoInterface,
    ModelSlimPipelineInterfaceV1,
    FA3QuantAdapterInterface,
    FA3QuantPlaceHolder,
)
from msmodelslim.utils.exception import InvalidModelError
from msmodelslim.utils.logging import logger_setter, get_logger
from msmodelslim.utils.security import get_valid_read_path, json_safe_load, MAX_READ_FILE_SIZE_512G

from .iquest_q1_model import (
    IQuestQ1CalibAttention,
    IQuestQ1CalibDecoderLayer,
    IQuestQ1CalibModel,
    IQuestQ1CalibMoE,
)

MAX_CALIB_SEQ_LEN = 1024


@contextmanager
def default_dtype(dtype):
    """Temporarily override the default dtype used for module construction."""
    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(original_dtype)


@logger_setter()
class IQuestQ1ModelAdapter(
    VLMBaseModelAdapter,
    ModelInfoInterface,
    ModelSlimPipelineInterfaceV1,
    FA3QuantAdapterInterface,
):
    """Layer-wise modelslim_v1 adapter for IQuest-Q1.

    - ``init_model`` builds the shell (embedding / final norm / lm_head on CPU
      with real weights); decoder layers are appended on demand by
      ``generate_model_visit`` / ``generate_model_forward``.
    - Module names mirror the checkpoint 1:1 so ascendv1_saver emits the
      expected quantized-key layout (incl. ``mlp.experts.fc.<eid>.*``).
    - MoE calibration runs all experts with stacked einsums over the fused
      checkpoint tensors captured at layer load time.
    """

    def __init__(self, model_type: str, model_path: Path, trust_remote_code: bool = False):
        super().__init__(model_type, model_path, trust_remote_code)
        self._tokenizer = AutoTokenizer.from_pretrained(  # nosec B615
            str(self.model_path), trust_remote_code=self.trust_remote_code, local_files_only=True
        )
        get_logger().info("IQuest-Q1 adapter ready (tokenizer=%s)", type(self._tokenizer).__name__)

    # ===== ModelInfoInterface =====

    def get_model_pedigree(self) -> str:
        return "iquest_q1"

    def get_model_type(self) -> str:
        return self.model_type

    def get_global_model_torch_dtype(self) -> torch.dtype:
        # config.json of IQuest-Q1 does not carry torch_dtype; the checkpoint
        # is bf16.
        return torch.bfloat16

    # ===== weight access helpers =====

    @lru_cache(maxsize=1)
    def _get_weight_map(self) -> Dict[str, str]:
        index_path = os.path.join(str(self.model_path), "model.safetensors.index.json")
        index_data = json_safe_load(index_path)
        return index_data["weight_map"]

    def _read_checkpoint_tensors(self, names: List[str], desc: str) -> Dict[str, torch.Tensor]:
        """Read the given checkpoint keys, grouped by safetensors shard."""
        weight_map = self._get_weight_map()
        file_groups: Dict[str, List[str]] = defaultdict(list)
        for name in names:
            if name not in weight_map:
                raise InvalidModelError(
                    f"Checkpoint key not found: {name}",
                    action="Please verify the model directory matches the IQuest-Q1 architecture.",
                )
            file_groups[weight_map[name]].append(name)

        tensors: Dict[str, torch.Tensor] = {}
        for file_name, keys in tqdm(file_groups.items(), desc=desc, leave=False):
            file_path = get_valid_read_path(
                os.path.join(str(self.model_path), file_name),
                extensions="safetensors",
                size_max=MAX_READ_FILE_SIZE_512G,
            )
            with safe_open(file_path, framework="pt", device="cpu") as f:
                for key in keys:
                    tensors[key] = f.get_tensor(key)
        return tensors

    def _layer_fused_keys(self, prefix: str) -> List[str]:
        return [f"{prefix}.mlp.experts.fc", f"{prefix}.mlp.experts.proj"]

    def _load_layer(self, idx: int) -> IQuestQ1CalibDecoderLayer:
        """Build decoder layer ``idx`` and load its checkpoint weights (CPU)."""
        prefix = f"model.layers.{idx}"
        get_logger().info("Loading decoder layer %d...", idx)

        with patch.object(nn.Linear, "reset_parameters", lambda _self: None), default_dtype(torch.bfloat16):
            layer = IQuestQ1CalibDecoderLayer(self.config, idx)

        # 1) Module-side parameter names (relative to the layer).
        module_param_keys: Set[str] = {name for name, _ in layer.named_parameters()}

        # 2) Checkpoint-side keys for this layer.
        weight_map = self._get_weight_map()
        prefix_dot = prefix + "."
        ckpt_keys: Set[str] = {k for k in weight_map if k.startswith(prefix_dot)}
        is_moe = isinstance(layer.mlp, IQuestQ1CalibMoE)
        fused_names = set(self._layer_fused_keys(prefix)) if is_moe else set()
        # Expand fused checkpoint keys into the per-expert parameter names the
        # modules expect (gate/up/down per expert id).
        #
        # NOTE: ``module_param_keys`` 取自 ``layer.named_parameters()``，是**相对层**
        # 的名字（如 ``attention_norm.weight``）；而 ``ckpt_keys`` 是全名
        # （``model.layers.N.attention_norm.weight``）。比对前必须把 expanded
        # 也剥掉 ``prefix_dot``，否则每一层都会误报 mismatch。
        expanded: Set[str] = set()
        inter = self.config.intermediate_size
        for key in ckpt_keys:
            if key == f"{prefix}.mlp.experts.fc":
                for e in range(self.config.num_experts):
                    expanded.add(f"mlp.experts.fc.{e}.gate_proj.weight")
                    expanded.add(f"mlp.experts.fc.{e}.up_proj.weight")
            elif key == f"{prefix}.mlp.experts.proj":
                for e in range(self.config.num_experts):
                    expanded.add(f"mlp.experts.proj.{e}.down_proj.weight")
            else:
                expanded.add(key[len(prefix_dot):])

        missing = module_param_keys - expanded
        unexpected = expanded - module_param_keys
        if missing or unexpected:
            raise InvalidModelError(
                f"Layer {idx} module/checkpoint mismatch: "
                f"missing-in-checkpoint={sorted(missing)[:5]}, "
                f"missing-in-model={sorted(unexpected)[:5]}",
                action="Please keep the calibration model in sync with the IQuest-Q1 checkpoint layout.",
            )
        if is_moe and not fused_names.issubset(ckpt_keys):
            raise InvalidModelError(
                f"Layer {idx} MoE layer is missing fused expert tensors {sorted(fused_names)}.",
                action="Please verify the checkpoint contains mlp.experts.fc / mlp.experts.proj.",
            )

        # 3) Read tensors.  Fused experts are read as whole 3-D tensors once;
        #    per-expert weights become views into them.
        state_dict: Dict[str, torch.Tensor] = {}
        read_names = sorted(ckpt_keys - fused_names)
        tensors = self._read_checkpoint_tensors(read_names, f"Loading layer {idx}")
        fc3d = proj3d = None
        if is_moe:
            fused_tensors = self._read_checkpoint_tensors(sorted(fused_names), f"Loading layer {idx} experts")
            fc3d = fused_tensors[f"{prefix}.mlp.experts.fc"]
            proj3d = fused_tensors[f"{prefix}.mlp.experts.proj"]
            for e in range(self.config.num_experts):
                state_dict[f"mlp.experts.fc.{e}.gate_proj.weight"] = fc3d[e, :inter, :]
                state_dict[f"mlp.experts.fc.{e}.up_proj.weight"] = fc3d[e, inter:, :]
                state_dict[f"mlp.experts.proj.{e}.down_proj.weight"] = proj3d[e]
        for full_key, tensor in tensors.items():
            state_dict[full_key[len(prefix_dot):]] = tensor

        # 4) Materialize and attach.
        layer.load_state_dict(state_dict, strict=True)
        layer.eval()
        if isinstance(layer.mlp, IQuestQ1CalibMoE):
            layer.mlp.capture_fused_weights(fc3d, proj3d)

        get_logger().info("Decoder layer %d loaded", idx)
        return layer

    def _ensure_layer(self, model: nn.Module, idx: int) -> nn.Module:
        layers = model.layers
        if len(layers) > idx:
            return layers[idx]
        layer = self._load_layer(idx)
        layers.append(layer)
        return layers[idx]

    # ===== ModelSlimPipelineInterfaceV1 =====

    def handle_dataset(self, dataset: Any, device: DeviceType = DeviceType.NPU) -> List[Any]:
        dev = device.value if isinstance(device, DeviceType) else str(device)
        processed: List[Dict[str, torch.Tensor]] = []
        for item in tqdm(dataset, desc="Processing calibration dataset"):
            if isinstance(item, dict):
                # lab_calib jsonl 约定键是 inputs_pretokenized（见 lab_calib/mix_calib.jsonl），
                # 其它模型适配器（minimax_m3 / step3_7_flash）同样兼容这三种键名。
                text = item.get("text") or item.get("prompt") or item.get("inputs_pretokenized")
                if text is None:
                    text = str(item)
            else:
                text = str(item)
            enc = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_CALIB_SEQ_LEN,
            )
            if enc["input_ids"].shape[1] < 8:
                get_logger().warning("Skipping calibration sample shorter than 8 tokens: %r", text[:32])
                continue
            processed.append(
                {
                    "input_ids": enc["input_ids"].to(dev),
                    "attention_mask": enc["attention_mask"].to(dev),
                }
            )
        if not processed:
            raise InvalidModelError(
                "No usable calibration samples after tokenization.",
                action="Please provide a non-empty calibration dataset.",
            )
        get_logger().info("Prepared %d calibration samples (device=%s)", len(processed), dev)
        return processed

    def init_model(self, device: DeviceType = DeviceType.NPU) -> nn.Module:
        """Shell model with real embedding / final norm / lm_head on CPU.

        Decoder layers are appended on demand (layer-wise streaming).
        """
        get_logger().info("Initializing IQuest-Q1 calibration shell (layer-wise)...")
        self.model_path = Path(get_valid_read_path(str(self.model_path), is_dir=True, check_user_stat=True))

        model = IQuestQ1CalibModel(self.config)

        root_keys = ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"]
        tensors = self._read_checkpoint_tensors(root_keys, "Loading shell weights")
        # 键名与 checkpoint 一致（model.* + lm_head），因为 IQuestQ1CalibModel
        # 现在按 ForCausalLM 形态嵌套了 ``.model`` 子层。
        model.load_state_dict(
            {
                "model.embed_tokens.weight": tensors["model.embed_tokens.weight"],
                "model.norm.weight": tensors["model.norm.weight"],
                "lm_head.weight": tensors["lm_head.weight"],
            },
            strict=True,
        )
        model.eval()
        get_logger().info("Shell model ready (layers appended on demand)")
        return model

    def generate_model_visit(self, model: nn.Module) -> Generator[ProcessRequest, Any, None]:
        """Visit pipeline: decoder layers in order (on-demand load)."""
        for idx in range(self.config.num_hidden_layers):
            layer = self._ensure_layer(model, idx)
            yield ProcessRequest(f"model.layers.{idx}", layer, tuple(), {})

    def generate_model_forward(self, model: nn.Module, inputs: Any) -> Generator[ProcessRequest, Any, None]:
        """Forward pipeline: embed -> layers (yielded one by one) -> final norm.

        The lm_head is not executed during calibration (no loss is needed);
        the module still exists on the shell so the saver exports it.
        """
        sample = inputs[0] if isinstance(inputs, (list, tuple)) else inputs
        input_ids = sample["input_ids"]

        hidden_states = model.embed_tokens(input_ids)
        for idx in range(self.config.num_hidden_layers):
            layer = self._ensure_layer(model, idx)
            hidden_states = yield ProcessRequest(
                f"model.layers.{idx}", layer, (hidden_states,), {}
            )
        hidden_states = model.norm(hidden_states)

    def enable_kv_cache(self, model: nn.Module, need_kv_cache: bool) -> None:
        # The calibration model is stateless (full-sequence eager attention);
        # only reflect the flag on the config for interface parity.
        model.config.use_cache = need_kv_cache
        get_logger().info("KV cache %s", "enabled" if need_kv_cache else "disabled")

    # ===== FA3QuantAdapterInterface =====

    def inject_fa3_placeholders(
        self, root_name: str, root_module: nn.Module, should_inject: Callable[[str], bool]
    ) -> None:
        """Install fa_q/fa_k/fa_v placeholders into the attention modules of
        the layer being visited, when its full name passes ``should_inject``."""
        for name, module in root_module.named_modules():
            if not isinstance(module, IQuestQ1CalibAttention):
                continue
            full_name = f"{root_name}.{name}" if root_name else name
            if not should_inject(full_name):
                continue
            if hasattr(module, "fa_q"):
                continue
            prefix = f"{name}." if name else ""
            root_module.set_submodule(f"{prefix}fa_q", FA3QuantPlaceHolder(ratio=1.0))
            root_module.set_submodule(f"{prefix}fa_k", FA3QuantPlaceHolder(ratio=1.0))
            root_module.set_submodule(f"{prefix}fa_v", FA3QuantPlaceHolder(ratio=1.0))
            get_logger().info("Injected FA3 placeholders into %s", full_name)
