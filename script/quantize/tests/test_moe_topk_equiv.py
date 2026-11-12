"""top-k dispatch vs dense all-expert reference: must match numerically."""
import sys
from types import SimpleNamespace

import torch
import torch.nn.functional as F

sys.path.insert(0, "/mnt/sfs_turbo/work/liuhaiyang/modelslim_full")
from msmodelslim.model.iquest_q1.iquest_q1_model import IQuestQ1CalibMoE  # noqa: E402

torch.manual_seed(0)
cfg = SimpleNamespace(num_experts=16, num_experts_per_tok=4, hidden_size=128, intermediate_size=64)
moe = IQuestQ1CalibMoE(cfg)
E, I, H = cfg.num_experts, cfg.intermediate_size, cfg.hidden_size
fc3d = torch.randn(E, 2 * I, H, dtype=torch.bfloat16)
proj3d = torch.randn(E, H, I, dtype=torch.bfloat16)
moe.capture_fused_weights(fc3d, proj3d)
moe.eval()

x = torch.randn(2, 5, H, dtype=torch.bfloat16)
with torch.no_grad():
    got = moe(x)

# dense reference (the previous implementation)
def dense_ref(hidden_states):
    xs = hidden_states.view(-1, H)
    T = xs.shape[0]
    logits = F.linear(xs.to(torch.float32), moe.router.weight.to(torch.float32))
    probs = torch.softmax(logits, dim=-1)
    tw, te = torch.topk(probs, cfg.num_experts_per_tok, dim=-1)
    tw = tw / tw.sum(dim=-1, keepdim=True)
    gate3, up3, down3 = fc3d[:, :I, :], fc3d[:, I:, :], proj3d
    xt = xs.t().to(gate3.dtype)
    gate_out = torch.matmul(gate3, xt)
    up_out = torch.matmul(up3, xt)
    act = (F.silu(gate_out) * up_out).transpose(1, 2)
    expert_out = torch.matmul(act, down3.transpose(1, 2))
    tok = torch.arange(T)
    final = torch.zeros(T, H, dtype=torch.float32)
    for i in range(cfg.num_experts_per_tok):
        sel = expert_out[te[:, i], tok].float()
        final += sel * tw[:, i].to(torch.float32).unsqueeze(-1)
    return final.to(xs.dtype).view(hidden_states.shape)

with torch.no_grad():
    ref = dense_ref(x)

diff = (got.float() - ref.float()).abs()
rel = diff.max().item() / ref.float().abs().max().item()
print(f"shape={tuple(got.shape)} dtype={got.dtype} max_abs={diff.max().item():.3e} rel={rel:.3e}")
assert rel < 1e-2, "top-k dispatch diverges from dense reference"
print("EQUIV OK")
