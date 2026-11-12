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

"""把逐专家的量化张量融合成 vLLM-ascend 需要的 ``routed_experts`` 布局。

校准模型的模块命名与 HF checkpoint 一致，routed expert 是逐专家存放的::

    model.layers.{L}.mlp.experts.fc.{E}.gate_proj.{weight,weight_scale}
    model.layers.{L}.mlp.experts.fc.{E}.up_proj.{weight,weight_scale}
    model.layers.{L}.mlp.experts.proj.{E}.down_proj.{weight,weight_scale}

而 vLLM-ascend 的 IQuest-Q1 实现按融合后的整块专家加载，可部署 checkpoint 中必须是::

    model.layers.{L}.mlp.experts.routed_experts.w13_weight        [E, 2I, H]
    model.layers.{L}.mlp.experts.routed_experts.w13_weight_scale
    model.layers.{L}.mlp.experts.routed_experts.w2_weight         [E, H, I]
    model.layers.{L}.mlp.experts.routed_experts.w2_weight_scale

``ExpertFuser`` 挂在保存器的落盘钩子上：逐专家张量先在这一层内缓存，等一层写完后
一次性拼好写出。因此不需要在量化结束后对整个 checkpoint 再做一遍读写。
"""

import re
from typing import Callable, Dict, List, Optional

import torch

#: ``model.layers.{L}.mlp.experts.{fc|proj}.{E}.{proj}.{suffix}``
EXPERT_KEY_RE = re.compile(
    r"^(?P<base>model\.layers\.(?P<layer>\d+)\.mlp\.experts)"
    r"\.(?:fc|proj)\.(?P<expert>\d+)"
    r"\.(?P<proj>gate_proj|up_proj|down_proj)\.(?P<suffix>weight|weight_scale)$"
)

#: 融合后的张量名 -> 沿第 0 维拼接的投影（顺序即拼接顺序）
FUSED_PARTS: Dict[str, List[str]] = {
    "w13": ["gate_proj", "up_proj"],
    "w2": ["down_proj"],
}

#: 逐专家投影 -> 所属融合张量
PROJ_TO_FUSED: Dict[str, str] = {
    "gate_proj": "w13",
    "up_proj": "w13",
    "down_proj": "w2",
}

SUFFIXES = ("weight", "weight_scale")

WriteTensor = Callable[[str, str, torch.Tensor], None]
WriteDesc = Callable[[str, str], None]


class ExpertFuser:
    """缓存一层的逐专家张量，按 ``routed_experts`` 布局融合写出。

    用法（由模型适配器的 ``AscendV1SaveTransformInterface`` 实现驱动）：

    - 每个待落盘张量调用 :meth:`consume`，逐专家键返回 ``True`` 表示已接管；
    - 所有层写完后调用 :meth:`flush` 落盘剩余的层。

    ``write_tensor`` / ``write_desc`` 由保存器在每次 :meth:`consume` 时传入并缓存，
    因此层切换时可以直接把上一层的融合结果写出去。
    """

    def __init__(self, num_experts: int):
        self._num_experts = num_experts
        # layer -> fused_name -> expert_id -> {"gate_proj.weight": tensor, ...}
        self._pending: Dict[int, Dict[str, Dict[int, Dict[str, torch.Tensor]]]] = {}
        # (layer, fused_name, suffix) -> 量化类型描述（如 W8A8_MXFP8）
        self._desc: Dict[tuple, str] = {}
        self._write_tensor: Optional[WriteTensor] = None
        self._write_desc: Optional[WriteDesc] = None

    # ===== 保存器钩子 =====

    def consume(
        self, prefix: str, desc: str, tensor: torch.Tensor, write_tensor: WriteTensor, write_desc: WriteDesc
    ) -> bool:
        """接管逐专家张量；非专家键返回 ``False`` 交还保存器。"""
        match = EXPERT_KEY_RE.match(prefix)
        if match is None:
            return False

        self._write_tensor = write_tensor
        self._write_desc = write_desc

        layer = int(match["layer"])
        for pending_layer in [item for item in self._pending if item != layer]:
            # 层按顺序保存，遇到新层时上一层的专家已经到齐，可以落盘。
            self._flush_layer(pending_layer)

        fused = PROJ_TO_FUSED[match["proj"]]
        key = (layer, fused, match["suffix"])
        known = self._desc.setdefault(key, desc)
        if known != desc:
            raise RuntimeError(
                f"专家张量 {prefix} 的量化类型 {desc} 与同组其它专家 {known} 不一致，无法融合。"
            )

        entry = self._pending.setdefault(layer, {}).setdefault(fused, {}).setdefault(int(match["expert"]), {})
        entry[f"{match['proj']}.{match['suffix']}"] = tensor
        return True

    def flush(self, write_tensor: WriteTensor, write_desc: WriteDesc) -> None:
        """落盘所有还在缓存中的层（保存器关闭分片前调用）。"""
        self._write_tensor = write_tensor
        self._write_desc = write_desc
        for layer in sorted(self._pending):
            self._flush_layer(layer)

    # ===== 内部实现 =====

    def _flush_layer(self, layer: int) -> None:
        groups = self._pending.pop(layer)
        base = f"model.layers.{layer}.mlp.experts"

        for fused, parts in FUSED_PARTS.items():
            experts = groups.get(fused)
            if not experts:
                continue

            missing = sorted(set(range(self._num_experts)) - set(experts))
            if missing:
                raise RuntimeError(
                    f"{base} 的 {fused} 只收到 {len(experts)}/{self._num_experts} 个专家"
                    f"（缺 {missing[:5]}...），无法融合成 routed_experts 布局。"
                    f"若开启了分布式/EP 保存，需要先在单进程内收敛全部专家。"
                )

            desc_weight = self._desc.get((layer, fused, "weight"))
            for suffix in SUFFIXES:
                desc = self._desc.pop((layer, fused, suffix), None)
                if desc is None:
                    continue
                fused_tensor = self._stack(experts, parts, suffix)
                self._write_tensor(f"{base}.routed_experts.{fused}_{suffix}", desc, fused_tensor)

            # vLLM 通过 experts.0.* 的别名键查量化类型，只写描述不写张量。
            if desc_weight is not None:
                for proj in parts:
                    self._write_desc(f"{base}.0.{proj}.weight", desc_weight)

    def _stack(self, experts: Dict[int, Dict[str, torch.Tensor]], parts: List[str], suffix: str) -> torch.Tensor:
        """把逐专家分片按专家号堆叠成 ``[E, len(parts)*rows, cols]``。"""
        sample = experts[0][f"{parts[0]}.{suffix}"]
        rows, cols = sample.shape[0], sample.shape[1]
        stacked = torch.empty(
            (self._num_experts, rows * len(parts), cols), dtype=sample.dtype, device=sample.device
        )
        for expert in range(self._num_experts):
            entry = experts[expert]
            for index, proj in enumerate(parts):
                stacked[expert, index * rows: (index + 1) * rows] = entry[f"{proj}.{suffix}"]
        return stacked
