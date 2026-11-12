#!/usr/bin/env python3
"""回归测试：mse_round 在 3D 融合专家权重上的 per-block 量化。

背景：IQuest-Q1 的 MoE 专家权重是 3D 融合张量 [E, 2I, H]（256 专家），
mxfp8/mxfp4 per_block 量化要沿最后一维分块。此前 MXWeightPerBlockMseRound
把量化后的 storage 还原成原形状，却用块形状的 scale 去反量化，广播失败：

    RuntimeError: The size of tensor a (1024) must match the size of tensor b (96)

修复方式与 MXWeightPerBlockMinmax 对齐：保留两份 storage
（block 形状供 forward 反量化 / 原形状供 deploy）。

本测试覆盖：
  1. 3D 权重能完成量化（不再抛广播异常）
  2. forward() 返回原形状
  3. get_q_storage() 返回原形状（deploy 用）
  4. get_q_param() 的 scale 是块形状
  5. 量化误差与 minmax 同量级（不因修复而劣化）
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../..")))

from msmodelslim.core.quantizer.base import QConfig  # noqa: E402
from msmodelslim.core.quantizer.impl.mse_round import MXWeightPerBlockMseRound  # noqa: E402
from msmodelslim.ir.qal import QDType, QScope, QStorage  # noqa: E402


def _make_quantizer(dtype, shape, weight=None):
    qconfig = QConfig(
        dtype=dtype,
        scope=QScope.PER_BLOCK,
        symmetric=True,
        method="mse_round",
    )
    q = MXWeightPerBlockMseRound(qconfig)
    w = weight if weight is not None else torch.randn(*shape, dtype=torch.float32)
    q.init_weight(QStorage(QDType.FLOAT, w))
    return q, w


@pytest.mark.parametrize("dtype", [QDType.MXFP8, QDType.MXFP4])
def test_mse_round_3d_fused_expert(dtype):
    """3D 融合专家权重 [E, 2I, H]：量化不崩、形状正确。"""
    E, I2, H = 4, 64, 128
    q, w = _make_quantizer(dtype, (E, I2, H))

    # forward：反量化结果必须是原始形状
    deq = q.forward()
    assert deq.shape == w.shape, f"forward 形状应为 {tuple(w.shape)}，实际 {tuple(deq.shape)}"

    # deploy 用的 storage：原始形状
    storage = q.get_q_storage()
    assert storage.value.shape == w.shape

    # scale 是块形状：块数在倒数第二维，末维是 observer 聚合后的 keepdim=1
    scale = q.get_q_param().ext["scale"]
    block = dtype.mx_finfo.block_size
    assert scale.shape[-2] == H // block, f"scale 块数维应为 {H // block}，实际 {scale.shape}"
    assert scale.shape[-1] == 1, f"scale 末维应为 keepdim=1，实际 {scale.shape}"

    # 量化误差应与权重量级同阶（不能是垃圾值）
    err = (deq - w).abs().max().item()
    assert err < 2.0, f"量化误差过大: {err}"


def test_mse_round_matches_minmax_accuracy():
    """修复不应让精度比 minmax 差太多（mse_round 的意义就是更优的舍入）。

    注意：两个 quantizer 必须喂**同一份**权重，否则 MSE 不可比。
    """
    from msmodelslim.core.quantizer.impl.minmax import MXWeightPerBlockMinmax

    shape = (4, 64, 128)
    torch.manual_seed(0)
    w = torch.randn(*shape, dtype=torch.float32)

    q_mse, _ = _make_quantizer(QDType.MXFP8, shape, weight=w)
    err_mse = (q_mse.forward() - w).pow(2).mean().item()

    qconfig = QConfig(dtype=QDType.MXFP8, scope=QScope.PER_BLOCK, symmetric=True, method="minmax")
    q_mm = MXWeightPerBlockMinmax(qconfig)
    q_mm.init_weight(QStorage(QDType.FLOAT, w.clone()))
    err_mm = (q_mm.forward() - w).pow(2).mean().item()

    assert err_mse <= err_mm * 1.5, f"mse_round MSE({err_mse:.6f}) 明显差于 minmax({err_mm:.6f})"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
