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

HiFloat8 静态 per-tensor 量化。

元素编码（1 sign + dot(4b) + payload(3b)）：
  dot=0  DML: 指数 -22..-16，0 位尾数
  dot=1  D0 : 指数 0，3 位尾数
  dot=2/3 D1: 指数 ±1，3 位尾数
  dot=4..7 D2: 指数 ±2..±3，3 位尾数
  dot=8..11 D3: 指数 ±4..±7，2 位尾数
  dot=12..15 D4: 指数 ±8..±15，1 位尾数

注意：hif8 不是均匀定点格式，"3 位尾数窗口" 约在 |x| ∈ [0.125, 16)。
因此 per-tensor scale 不能沿用 int8/FP8 的 amax/max_norm，而是把 amax
压到窗口上沿附近：scale = amax / max_abs_ratio（默认 16）。

编码实现与 cann ops-transformer golden 的
``cvt_float32_to_hifuint8(round_mode="hybrid")`` 对齐（TA/SSR 混合舍入），
解码表按位布局解析，用于伪量化的反量化。
"""

import torch

from msmodelslim.ir.qal import QStorage
from msmodelslim.ir.qal.qbase import QDType, QScope, QParam, QScheme
from msmodelslim.ir.qal.qregistry import QFuncRegistry

# ---------------------------------------------------------------------------
# 编码常量（与 ops-transformer / vllm-ascend golden 一致）
# ---------------------------------------------------------------------------
FP32_FRACTION_BITS = 23

HIF8_EXP_ZERO_THRESHOLD = -23
HIF8_EXP_DML_MIN = -22
HIF8_EXP_DML_MAX = -15
HIF8_EXP_D0 = 0
HIF8_EXP_D1_BOUNDARY = 1
HIF8_EXP_D2_MIN, HIF8_EXP_D2_MAX = 2, 3
HIF8_EXP_D3_MIN, HIF8_EXP_D3_MAX = 4, 7
HIF8_EXP_D4_MIN, HIF8_EXP_D4_MAX = 8, 15

HIF8_DOT_DML = 0
HIF8_DOT_D0 = 1
HIF8_DOT_D1 = 2
HIF8_DOT_D2 = 4
HIF8_DOT_D3 = 8
HIF8_DOT_D4 = 12
HIF8_DOT_INVALID = -1

HIF8_FRAC_BITS_DML = 0
HIF8_FRAC_BITS_D0 = 3
HIF8_FRAC_BITS_D1 = 3
HIF8_FRAC_BITS_D2 = 3
HIF8_FRAC_BITS_D3 = 2
HIF8_FRAC_BITS_D4 = 1

HIF8_EXP_BITS_DML = 3
HIF8_EXP_BITS_D0 = 0
HIF8_EXP_BITS_D1 = 1
HIF8_EXP_BITS_D2 = 2
HIF8_EXP_BITS_D3 = 3
HIF8_EXP_BITS_D4 = 4

HIF8_ZERO = 0
HIF8_NAN = 128
HIF8_NEG_INF = 239
HIF8_NEG_MAX = 238
HIF8_POS_INF = 111
HIF8_POS_MAX = 110

HIF8_SIGN_MASK = 128
HIF8_DOT_BIT_SHIFT = 3
HIF8_DML_EXP_OFFSET = 23

HIF8_OVERFLOW_SCALE = 1.25
HIF8_MAX_FINITE_VALUE = 32768.0

SSR_T14_MASK = 16383
SSR_F14_OFFSET = 8192
SSR_DML_SHIFT = 10
SSR_RESERVED_BITS = 14
HYBRID_ROUND_EXP_THRESHOLD = 4

# per-tensor scale 的默认窗口：amax / 16 落在 3 位尾数窗口上沿
DEFAULT_MAX_ABS_RATIO = 16.0
_MIN_SCALE = 1e-12

# "round" = TA 舍入（golden 默认）；"hybrid" = TA/SSR 混合（设备 VF 的另一种口径）
DEFAULT_ROUND_MODE = "round"


def _classify(exponent):
    """按绝对值指数返回 (dot, exp_bits, frac_bits)；超出可表示范围返回 INVALID。"""
    abs_exp = torch.abs(exponent)
    dot = torch.full_like(exponent, HIF8_DOT_INVALID)
    exp_bits = torch.zeros_like(exponent)
    frac_bits = torch.zeros_like(exponent)

    m1 = exponent < HIF8_EXP_DML_MIN
    exp_bits = torch.where(m1, HIF8_EXP_BITS_DML, exp_bits)
    frac_bits = torch.where(m1, HIF8_FRAC_BITS_DML, frac_bits)

    m2 = (~m1) & (exponent < HIF8_EXP_DML_MAX)
    dot = torch.where(m2, HIF8_DOT_DML, dot)
    exp_bits = torch.where(m2, HIF8_EXP_BITS_DML, exp_bits)
    frac_bits = torch.where(m2, HIF8_FRAC_BITS_DML, frac_bits)

    m3 = exponent == HIF8_EXP_D0
    dot = torch.where(m3, HIF8_DOT_D0, dot)
    exp_bits = torch.where(m3, HIF8_EXP_BITS_D0, exp_bits)
    frac_bits = torch.where(m3, HIF8_FRAC_BITS_D0, frac_bits)

    m4 = abs_exp == HIF8_EXP_D1_BOUNDARY
    dot = torch.where(m4, HIF8_DOT_D1, dot)
    exp_bits = torch.where(m4, HIF8_EXP_BITS_D1, exp_bits)
    frac_bits = torch.where(m4, HIF8_FRAC_BITS_D1, frac_bits)

    m5 = (abs_exp >= HIF8_EXP_D2_MIN) & (abs_exp <= HIF8_EXP_D2_MAX)
    dot = torch.where(m5, HIF8_DOT_D2, dot)
    exp_bits = torch.where(m5, HIF8_EXP_BITS_D2, exp_bits)
    frac_bits = torch.where(m5, HIF8_FRAC_BITS_D2, frac_bits)

    m6 = (abs_exp >= HIF8_EXP_D3_MIN) & (abs_exp <= HIF8_EXP_D3_MAX)
    dot = torch.where(m6, HIF8_DOT_D3, dot)
    exp_bits = torch.where(m6, HIF8_EXP_BITS_D3, exp_bits)
    frac_bits = torch.where(m6, HIF8_FRAC_BITS_D3, frac_bits)

    m7 = (abs_exp >= HIF8_EXP_D4_MIN) & (abs_exp <= HIF8_EXP_D4_MAX)
    dot = torch.where(m7, HIF8_DOT_D4, dot)
    exp_bits = torch.where(m7, HIF8_EXP_BITS_D4, exp_bits)
    frac_bits = torch.where(m7, HIF8_FRAC_BITS_D4, frac_bits)

    return dot, exp_bits, frac_bits


def _round_ta(fraction_int, frac_bits, exponent):
    """TA 舍入（与 golden fp32_ta_round_to_hif8 一致）。"""
    zero_thresh = exponent == HIF8_EXP_ZERO_THRESHOLD
    carry = zero_thresh.clone()

    shift_bits = torch.clamp(FP32_FRACTION_BITS - (frac_bits + 1), min=0)
    tmp = fraction_int >> shift_bits

    pow_frac = torch.pow(torch.tensor(2, dtype=torch.int64, device=tmp.device), frac_bits + 1) - 1
    carry = torch.where((~zero_thresh) & (tmp == pow_frac), torch.ones_like(carry), carry)
    frac_val = torch.zeros_like(fraction_int)
    odd = (~zero_thresh) & (tmp != pow_frac) & (tmp != 0) & (tmp % 2 == 1)
    even = (~zero_thresh) & (tmp != pow_frac) & (tmp != 0) & (tmp % 2 == 0)
    frac_val = torch.where(odd, (tmp + 1) >> 1, frac_val)
    frac_val = torch.where(even, tmp >> 1, frac_val)
    return carry, frac_val


def _round_ssr(fraction_int, frac_bits, exponent):
    """SSR 舍入（与 golden fp32_ssr_round_to_hif8 一致）。"""
    zero_thresh = exponent == HIF8_EXP_ZERO_THRESHOLD

    f14_v1 = (fraction_int >> SSR_DML_SHIFT) + SSR_F14_OFFSET
    t14_v1 = fraction_int & SSR_T14_MASK
    hif8_v1 = torch.zeros_like(fraction_int)

    s_bits = torch.clamp(FP32_FRACTION_BITS - frac_bits, min=0)
    hif8_v2 = fraction_int >> s_bits
    f14_t14 = fraction_int - (hif8_v2 << s_bits)
    s_bits_f14 = torch.clamp(FP32_FRACTION_BITS - frac_bits - SSR_RESERVED_BITS, min=0)
    f14_v2 = f14_t14 >> s_bits_f14
    t14_v2 = f14_t14 & SSR_T14_MASK

    f14 = torch.where(zero_thresh, f14_v1, f14_v2)
    t14 = torch.where(zero_thresh, t14_v1, t14_v2)
    hif8_value = torch.where(zero_thresh, hif8_v1, hif8_v2)

    ge = f14 >= t14
    pow_frac = torch.pow(torch.tensor(2, dtype=torch.int64, device=f14.device), frac_bits) - 1
    carry = ge & (hif8_value == pow_frac)
    frac_val = torch.where(ge & (~carry), hif8_value + 1, torch.zeros_like(hif8_value))
    frac_val = torch.where(~ge, hif8_value, frac_val)
    return carry, frac_val


def encode_hif8(x: torch.Tensor, round_mode: str = DEFAULT_ROUND_MODE) -> torch.Tensor:
    """float32 → HiFloat8 8bit 码（uint8）。SAT 语义：NaN→0，溢出→±32768。

    与 cann ops-transformer golden 的 ``trans_float_tensor_to_hifuint8`` 一致；
    默认 ``round``（TA）可逐码 round-trip，``hybrid`` 复刻 VF 的 TA/SSR 混合口径。
    """
    shape = x.shape
    x = x.reshape(-1).to(torch.float32)

    sign_mask = x < 0.0
    sign_int_value = torch.where(sign_mask, torch.full_like(x, HIF8_SIGN_MASK, dtype=torch.int32),
                                 torch.zeros_like(x, dtype=torch.int32))
    x_abs = torch.abs(x)

    mask_nan = torch.isnan(x)
    over_value = HIF8_OVERFLOW_SCALE * (2.0**HIF8_EXP_D4_MAX)
    mask_over = torch.isinf(x) | (x_abs >= over_value)
    mask_zero = x_abs == 0.0

    out = torch.zeros_like(x, dtype=torch.int32)
    # SAT：溢出饱和到最大有限值，NaN 归零
    out = torch.where(mask_over, torch.where(sign_mask, HIF8_NEG_MAX, HIF8_POS_MAX), out).to(torch.int32)
    out = torch.where(mask_nan, 0, out).to(torch.int32)
    out = torch.where(mask_zero, 0, out).to(torch.int32)

    mask_normal = ~(mask_over | mask_nan | mask_zero)
    if not bool(mask_normal.any()):
        return out.reshape(shape).to(torch.uint8)

    x_norm = x_abs[mask_normal]
    sign_norm = sign_int_value[mask_normal]

    # 精确取 floor(log2(x))：frexp 给出 x = m*2^e (m∈[0.5,1))，e-1 即二进制指数。
    # torch.log2 在 binade 边界（含 subnormal）会因超越函数舍入差一，导致 dot 档/
    # TA-SSR 舍入模式判错。
    exponent = (torch.frexp(x_norm).exponent - 1).to(torch.int32)
    if round_mode == "hybrid":
        cut_bit_is_ta = torch.abs(exponent) < HYBRID_ROUND_EXP_THRESHOLD
    elif round_mode == "storound":
        cut_bit_is_ta = torch.zeros_like(exponent, dtype=torch.bool)
    else:
        cut_bit_is_ta = torch.ones_like(exponent, dtype=torch.bool)

    fraction_int = (
        x_norm * (2.0**FP32_FRACTION_BITS) * torch.pow(2.0, -exponent.float()) - (2.0**FP32_FRACTION_BITS)
    ).to(torch.int32)

    dot, exp_bits, frac_bits = _classify(exponent)

    carry_ta, frac_ta = _round_ta(fraction_int, frac_bits, exponent)
    carry_ssr, frac_ssr = _round_ssr(fraction_int, frac_bits, exponent)
    carry = torch.where(cut_bit_is_ta, carry_ta, carry_ssr)
    frac_value = torch.where(cut_bit_is_ta, frac_ta, frac_ssr)

    exponent = torch.where(carry, exponent + 1, exponent)
    new_dot, new_exp_bits, new_frac_bits = _classify(exponent)
    dot = torch.where(carry, new_dot, dot)
    exp_bits = torch.where(carry, new_exp_bits, exp_bits)
    frac_bits = torch.where(carry, new_frac_bits, frac_bits)

    sign_value = torch.where(exponent < 0, 1, 0)
    dot_int = dot << HIF8_DOT_BIT_SHIFT
    abs_exp = torch.abs(exponent)
    abs_exp = abs_exp - torch.pow(torch.tensor(2, dtype=torch.int32, device=abs_exp.device),
                                  (exp_bits - 1).clamp(min=0).to(torch.int32))
    exp_int = abs_exp * torch.pow(torch.tensor(2, dtype=torch.int32, device=abs_exp.device),
                                  frac_bits.to(torch.int32))
    sig_exp = sign_value * torch.pow(torch.tensor(2, dtype=torch.int32, device=abs_exp.device),
                                     (exp_bits - 1 + frac_bits).clamp(min=0).to(torch.int32))

    dml = dot <= 0
    d0 = dot == 1
    code_dml = torch.where(exponent <= HIF8_EXP_ZERO_THRESHOLD, 0, sign_norm + exponent + HIF8_DML_EXP_OFFSET)
    code_d0 = sign_norm + dot_int + frac_value
    code_other = sign_norm + dot_int + sig_exp + exp_int + frac_value

    code = torch.where(dml, code_dml, torch.where(d0, code_d0, code_other))
    out[mask_normal] = code.to(torch.int32)
    return out.reshape(shape).to(torch.uint8)


def _build_decode_table():
    """按位布局解析 256 个 hif8 码对应的 float32 值。"""
    table = []
    for code in range(256):
        if code == HIF8_NAN:
            table.append(float("nan"))
            continue
        if code in (HIF8_POS_INF, HIF8_NEG_INF):
            table.append(float("inf") if code == HIF8_POS_INF else float("-inf"))
            continue
        if code in (HIF8_POS_MAX, HIF8_NEG_MAX):
            table.append(HIF8_MAX_FINITE_VALUE if code == HIF8_POS_MAX else -HIF8_MAX_FINITE_VALUE)
            continue
        sign = -1.0 if code & HIF8_SIGN_MASK else 1.0
        d = code & 0x7F
        if d == 0:
            value = 0.0
        elif d <= 7:
            value = 2.0 ** (d - HIF8_DML_EXP_OFFSET)
        elif d <= 15:
            value = 1.0 + (d - 8) / 8.0
        elif d <= 23:
            value = (1.0 + (d - 16) / 8.0) * 2.0
        elif d <= 31:
            value = (1.0 + (d - 24) / 8.0) * 0.5
        elif d <= 39:
            value = (1.0 + (d - 32) / 8.0) * 4.0
        elif d <= 47:
            value = (1.0 + (d - 40) / 8.0) * 8.0
        elif d <= 55:
            value = (1.0 + (d - 48) / 8.0) * 0.25
        elif d <= 63:
            value = (1.0 + (d - 56) / 8.0) * 0.125
        elif d <= 95:
            neg_exp = (d >> 4) & 1
            mag = (d >> 2) & 3
            frac = (d & 3) / 4.0
            e = -(4 + mag) if neg_exp else (4 + mag)
            value = (1.0 + frac) * (2.0 ** e)
        else:
            neg_exp = (d >> 4) & 1
            mag = (d >> 1) & 7
            frac = (d & 1) / 2.0
            e = -(8 + mag) if neg_exp else (8 + mag)
            value = (1.0 + frac) * (2.0 ** e)
        table.append(sign * value)
    return torch.tensor(table, dtype=torch.float32)


_HIF8_DECODE_TABLE = _build_decode_table()


def hif8_codes_to_float(codes: torch.Tensor) -> torch.Tensor:
    table = _HIF8_DECODE_TABLE.to(codes.device)
    values = table[codes.long()]
    # SAT：NaN 码（编码器不会产生，但防御）归零
    return torch.nan_to_num(values, nan=0.0)


def hif8_fake_quant(x: torch.Tensor, scale) -> torch.Tensor:
    """x / scale → hif8 网格 → * scale。"""
    scale_t = torch.as_tensor(scale, dtype=torch.float32, device=x.device)
    scaled = x.to(torch.float32) / scale_t
    codes = encode_hif8(scaled, round_mode=DEFAULT_ROUND_MODE)
    return hif8_codes_to_float(codes) * scale_t


@QFuncRegistry.register(dispatch_key=(QDType.HIFLOAT8, QScope.PER_TENSOR, True), api_name="calculate_qparam")
def calculate_hif8_qparam(
    min_val: torch.Tensor,
    max_val: torch.Tensor,
    q_dtype: QDType,
    q_scope: QScope,
    symmetric: bool,
    **kwargs,
) -> QParam:
    """静态 per-tensor：scale = amax / max_abs_ratio（默认 16）。

    hif8 的 3 位尾数窗口约 [0.125, 16)，把 amax 压到 16 才能让主分布落在高精度区间；
    amax/32768 这类 FP8 习惯会白丢 2~3 位尾数。

    注意 QFuncRegistry 要求注册函数的签名与 API 完全一致，max_abs_ratio 只能走 kwargs。
    """
    ratio = float(kwargs.get("max_abs_ratio", DEFAULT_MAX_ABS_RATIO))
    if not (0.0 < ratio <= HIF8_MAX_FINITE_VALUE):
        raise ValueError(f"max_abs_ratio must be in (0, {HIF8_MAX_FINITE_VALUE}], got {ratio}")
    amax = torch.maximum(min_val.abs(), max_val.abs()).to(torch.float32)
    scale = amax.clamp(min=_MIN_SCALE) / ratio
    offset = torch.zeros_like(scale)
    return QParam(
        scheme=QScheme(dtype=q_dtype, scope=q_scope, symmetric=symmetric),
        ext={"scale": scale, "offset": offset},
    )


@QFuncRegistry.register(
    dispatch_key=(QDType.FLOAT, QDType.HIFLOAT8, QScope.PER_TENSOR, True), api_name="quantize"
)
def hif8_quantize(tensor: QStorage, q_param: QParam) -> QStorage:
    scale = q_param.ext["scale"]
    if scale is None:
        raise ValueError("scale is None, please check the q_param")
    scaled = tensor.value.to(torch.float32) / torch.as_tensor(scale, dtype=torch.float32,
                                                              device=tensor.value.device)
    scaled = torch.nan_to_num(scaled, nan=0.0, posinf=HIF8_MAX_FINITE_VALUE, neginf=-HIF8_MAX_FINITE_VALUE)
    scaled = scaled.clamp(-HIF8_MAX_FINITE_VALUE, HIF8_MAX_FINITE_VALUE)
    codes = encode_hif8(scaled, round_mode=DEFAULT_ROUND_MODE)
    values = hif8_codes_to_float(codes)
    return tensor.same_like(values).to(QDType.HIFLOAT8)


@QFuncRegistry.register(
    dispatch_key=(QDType.HIFLOAT8, QDType.HIFLOAT8, QScope.PER_TENSOR, True), api_name="dequantize"
)
def hif8_dequantize(tensor: QStorage, q_param: QParam) -> QStorage:
    scale = q_param.ext["scale"]
    scale_t = torch.as_tensor(scale, dtype=torch.float32, device=tensor.value.device)
    values = tensor.value.to(torch.float32) * scale_t
    return tensor.same_like(values).to(QDType.FLOAT)
