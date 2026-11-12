#!/usr/bin/env python
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

from abc import ABC, abstractmethod
from typing import Callable, Tuple

import torch
from torch import nn

#: ``(prefix, desc, tensor)`` writer supplied by the saver.
WriteTensor = Callable[[str, str, torch.Tensor], None]
#: ``(prefix, desc)`` description-only writer supplied by the saver (no tensor).
WriteDesc = Callable[[str, str], None]


class AscendV1GlobalModelDtypeInterface(ABC):
    """Interface for adapters that expose the global model torch dtype (e.g. for Saver to decide deq_scale int64)."""

    @abstractmethod
    def get_global_model_torch_dtype(self) -> torch.dtype:
        """
        Return the global torch dtype used for model loading/calibration.
        Used by Saver and other components to infer precision (e.g. whether bfloat16).
        """
        pass


class AscendV1SaveInterface(ABC):
    def ascendv1_save_postprocess(self, model: nn.Module, save_directory: str) -> None:
        """
        导出件后处理
        @param model: 量化模型
        @param save_directory: 包含导出件（如config.json，quant_model_description.json等）的量化模型存储路径
        """
        pass

    def ascendv1_save_module_preprocess(
        self, prefix: str, module: nn.Module, model: nn.Module
    ) -> Tuple[str, nn.Module]:
        """
        在保存模块前，对模块进行预处理，返回新的前缀和模块
        @param prefix: 模块的前缀路径
        @param module: 待处理的模块
        @param model: 模型
        @return: 返回(prefix, module)
        """
        pass


class AscendV1SaveTransformInterface(ABC):
    """在落盘前改写张量的适配器扩展点。

    有些部署格式把 checkpoint 中的多个张量合成一个（例如把逐专家的
    ``mlp.experts.fc.<e>.gate_proj`` 融合成 ``mlp.experts.routed_experts.w13_weight``）。
    适配器实现本接口后，``AscendV1Saver`` 会把每个即将落盘的张量先交给适配器：

    - ``ascendv1_transform_save_entry`` 返回 ``True`` 表示适配器已接管该张量
      （例如先缓存，等同一层的专家都到齐再一次性写出融合结果），保存器不再写入它；
    - ``ascendv1_flush_save_entries`` 在分片文件关闭前调用一次，供适配器写出缓存。

    适配器在 ``transform`` 中还会拿到 ``write_tensor`` / ``write_desc`` 两个回调，
    可直接用于写出自己拼好的张量（``write_desc`` 只写描述、不写张量，用于
    vLLM 读取的别名键）。
    """

    @abstractmethod
    def ascendv1_transform_save_entry(
        self, prefix: str, desc: str, tensor: torch.Tensor, write_tensor: WriteTensor, write_desc: WriteDesc
    ) -> bool:
        """返回 True 表示该张量已被适配器接管，保存器不再写入。"""
        pass

    @abstractmethod
    def ascendv1_flush_save_entries(self, write_tensor: WriteTensor, write_desc: WriteDesc) -> None:
        """分片关闭前落盘所有被接管的张量。"""
        pass
