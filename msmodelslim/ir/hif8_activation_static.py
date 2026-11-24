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

import torch
from torch import nn

from msmodelslim.ir.api import fake_quantize
from msmodelslim.ir.qal import QABCRegistry, QDType, QParam, QStorage
from msmodelslim.utils.logging import logger_setter
from .auto import AutoFakeQuantActivation
from .const import hif8_per_tensor_sym


@QABCRegistry.multi_register(dispatch_key=[hif8_per_tensor_sym], abc_type=AutoFakeQuantActivation)
@logger_setter()
class HIF8FakeQuantActivationPerTensor(AutoFakeQuantActivation):
    """FA Q/K/V 分支：HiFloat8 静态 per-tensor 伪量化。

    scale 为单标量，导出到 checkpoint 的 ``fa_{q,k,v}.scale``（float32）。
    运行时（QFA quant_mode=0）以 (1,) 的 descale 形式消费。
    """

    def __init__(self, x_q_param: QParam):
        super().__init__()
        self.x_q_scheme = x_q_param.scheme

        scale = x_q_param.ext.get("scale")
        if scale is None:
            raise ValueError(f"`scale` is needed in ext but is missing for {self.__class__.__name__}")

        self.input_scale = nn.Parameter(torch.as_tensor(scale, dtype=torch.float32), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_q_param = QParam(scheme=self.x_q_scheme, ext={"scale": self.input_scale.data})
        return fake_quantize(QStorage(QDType.FLOAT, x), x_q_param).value
