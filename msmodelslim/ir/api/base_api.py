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

from typing import Optional, Tuple

import torch

from msmodelslim.ir.qal.qbase import QDType, QParam, QScope
from msmodelslim.ir.qal.qbase import QStorage
from msmodelslim.ir.qal.qregistry import QFuncRegistry
from msmodelslim.utils.exception import SchemaValidateError


def _raise_if_non_finite(label: str, value, name: Optional[str] = None) -> None:
    """NaN/Inf 一旦进入 min/max 或 scale，会污染整个张量（甚至整个 checkpoint）。"""
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    if tensor.device.type == "meta" or tensor.numel() == 0:
        return
    if bool(torch.isfinite(tensor).all()):
        return
    nan = int(torch.isnan(tensor).sum())
    inf = int(torch.isinf(tensor).sum())
    where = f"（{name}）" if name else ""
    raise SchemaValidateError(
        f"{label}{where} 含非有限值：NaN={nan}, Inf={inf}",
        action="请检查标定数据或溢出；单个 NaN/Inf 会让该张量的 scale/offset 全部失效，"
        "进而污染导出的量化权重。",
    )


@QFuncRegistry.register_api(dispatch_key=Tuple[QDType, QScope, bool])
def calculate_qparam(
    min_val: torch.Tensor,
    max_val: torch.Tensor,
    q_dtype: QDType,
    q_scope: QScope,
    symmetric: bool,
    **kwargs,
) -> QParam:
    """
    计算量化参数函数，根据输入张量的最小值和最大值计算量化参数。

    Args:
        min_val: 输入张量的最小值，类型为torch.Tensor
        max_val: 输入张量的最大值，类型为torch.Tensor
        q_dtype: 量化数据类型，类型为QDType
        q_scope: 量化范围，类型为QScope
        symmetric: 是否使用对称量化，类型为bool

    Returns:
        QParam: 计算得到的量化参数

    Note:
        由于实现上的限制，并非所有的入参组合都是合法的，对于非法组合，QFuncRegistry.dispatch()接口将会抛出NotImplementedError异常。
    """
    layer_name = kwargs.pop("name", None)
    _raise_if_non_finite("量化统计量 min_val", min_val, layer_name)
    _raise_if_non_finite("量化统计量 max_val", max_val, layer_name)
    if torch.is_tensor(min_val) and torch.is_tensor(max_val) and min_val.shape == max_val.shape:
        if bool((min_val > max_val).any()):
            where = f"（{layer_name}）" if layer_name else ""
            raise SchemaValidateError(
                f"量化统计量 min_val > max_val{where}",
                action="请检查标定数据或 observer 统计逻辑，非法统计会让 scale 失去意义。",
            )
    q_param = QFuncRegistry.dispatch(
        "calculate_qparam", (q_dtype, q_scope, symmetric), *(min_val, max_val, q_dtype, q_scope, symmetric), **kwargs
    )
    ext = q_param.ext if isinstance(q_param.ext, dict) else {}
    for key in ("scale", "offset"):
        if key in ext:
            _raise_if_non_finite(f"量化参数 {key}", ext[key], layer_name)
    return q_param


@QFuncRegistry.register_api(dispatch_key=Tuple[QDType, QDType, QScope, bool])
def quantize(tensor: QStorage, q_param: QParam) -> QStorage:
    """
    量化函数，用于将浮点张量量化为量化张量。

    Args:
        tensor: 需要进行量化操作的张量，类型为QStorage

        q_param: 量化过程所需要的量化参数，类型为QParam

    Returns:
        QStorage: 量化张量，类型取决于q_param中的量化方案

    Note:
        由于实现上的限制，并非所有的入参组合都是合法的，对于非法组合，QFuncRegistry.dispatch()接口将会抛出NotImplementedError异常。

    """

    return QFuncRegistry.dispatch(
        "quantize",
        (tensor.dtype, q_param.scheme.dtype, q_param.scheme.scope, q_param.scheme.symmetric),
        *(tensor, q_param),
    )


@QFuncRegistry.register_api(dispatch_key=Tuple[QDType, QDType, QScope, bool])
def dequantize(tensor: QStorage, q_param: QParam) -> QStorage:
    """
    反量化函数，用于将量化张量反量化为浮点张量。

    Args:
        tensor: 需要进行反量化操作的量化张量，类型为QStorage

        q_param: 反量化过程所需要的量化参数，类型为QParam

    Returns:
        QStorage: 反量化后的浮点张量，类型取决于q_param中的量化方案

    Note:
        由于实现上的限制，并非所有的入参组合都是合法的，对于非法组合，QFuncRegistry.dispatch()接口将会抛出NotImplementedError异常。

    """

    return QFuncRegistry.dispatch(
        "dequantize",
        (tensor.dtype, q_param.scheme.dtype, q_param.scheme.scope, q_param.scheme.symmetric),
        *(tensor, q_param),
    )


@QFuncRegistry.register_api(dispatch_key=Tuple[QDType, QDType, QScope, bool])
def fake_quantize(tensor: QStorage, q_param: QParam) -> QStorage:
    """
    伪量化函数，将浮点张量以特定的量化参数进行量化然后反量化后，返回一个与输入张量shape/dtype一致的新张量。

    Args:
        tensor: 浮点张量，类型为QStorage
        q_param: 量化参数，类型为QParam

    Returns:
        QStorage: 伪量化后的浮点张量，dtype/shape与输入的tensor一致，与输入张量不共享存储空间。
    """

    return dequantize(quantize(tensor, q_param), q_param)
