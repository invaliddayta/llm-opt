"""PyTorch DFlash2 drafter that mirrors llama.cpp's inference graph (src/models/dflash.cpp).

Frozen base weights come from the published z-lab checkpoint; training adds LoRA
adapters on the big matmuls and fully trains the small DFlash2-specific tensors.
The target model's token embedding and LM head are the dequantized tensors of the
deployed GGUF, so training sees exactly what inference uses.
"""

import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint


class LoRALinear(nn.Module):
    def __init__(self, weight: torch.Tensor, rank: int, alpha: float):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)  # [out, in]
        self.rank = rank
        if rank > 0:
            self.lora_a = nn.Parameter(torch.empty(rank, weight.shape[1], dtype=torch.float32))
            self.lora_b = nn.Parameter(torch.zeros(weight.shape[0], rank, dtype=torch.float32))
            nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
            self.scale = alpha / rank

    def forward(self, x):
        y = F.linear(x, self.weight)
        if self.rank > 0:
            y = y + F.linear(F.linear(x, self.lora_a.to(x.dtype)), self.lora_b.to(x.dtype)) * self.scale
        return y

    @torch.no_grad()
    def merged_weight(self):
        w = self.weight.float()
        if self.rank > 0:
            w = w + (self.lora_b @ self.lora_a) * self.scale
        return w


def rms_norm(x, w, eps):
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xf * w.float()).to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, w, eps):
        super().__init__()
        self.weight = nn.Parameter(w.float())
        self.eps = eps

    def forward(self, x):
        return rms_norm(x, self.weight, self.eps)


def rope_cos_sin(pos, head_dim, theta, device):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    f = pos.float()[:, None] * inv[None, :]
    emb = torch.cat([f, f], dim=-1)
    return emb.cos(), emb.sin()


def apply_rope(x, cos, sin):
    # x [..., T, H, D]  NEOX rotate_half; cos/sin [T, D]
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    rot = torch.cat([-x2, x1], dim=-1)
    c = cos[:, None, :].to(x.dtype)
    s = sin[:, None, :].to(x.dtype)
    return x * c + rot * s


class GroupedConv(nn.Module):
    """DFlash2 two-tap dynamic depthwise conv inside each block (matches build_dflash2_conv)."""

    def __init__(self, base_kernel, proj_w, group_size, lora_rank, lora_alpha):
        super().__init__()
        self.base_kernel = nn.Parameter(base_kernel.float())  # [2 side, taps, C]
        self.kernel_projection = nn.Parameter(proj_w.float())  # [2*taps*G, C], fully trained (small)
        self.taps = base_kernel.shape[1]
        self.group_size = group_size
        self.n_groups = base_kernel.shape[2] // group_size

    def prepare(self, x, block):
        # x [N_blocks*block, C]
        coeff = F.linear(x, self.kernel_projection.to(x.dtype)).view(-1, 2, self.taps, self.n_groups)
        return self._conv(x, coeff[:, 0], 0, block), coeff[:, 1]

    def finish(self, x, coeff, block):
        return self._conv(x, coeff, 1, block)

    def _conv(self, x, dyn, side, block):
        T, C = x.shape
        nb = T // block
        xb = x.view(nb, block, self.n_groups, self.group_size)
        dyn = dyn.view(nb, block, self.taps, self.n_groups, 1)
        base = self.base_kernel[side].view(1, 1, self.taps, self.n_groups, self.group_size)
        coef = (base + dyn.float()).to(x.dtype)
        out = coef[:, :, 0] * xb
        for tap in range(1, min(self.taps, block)):
            shifted = F.pad(xb[:, : block - tap], (0, 0, 0, 0, tap, 0))
            out = out + coef[:, :, tap] * shifted
        return out.reshape(T, C)


class Layer(nn.Module):
    def __init__(self, sd, i, cfg, rank, alpha):
        super().__init__()
        p = f"layers.{i}."
        eps = cfg["rms_norm_eps"]
        g = lambda k: sd[p + k]
        self.input_layernorm = RMSNorm(g("input_layernorm.weight"), eps)
        self.post_attention_layernorm = RMSNorm(g("post_attention_layernorm.weight"), eps)
        self.q_proj = LoRALinear(g("self_attn.q_proj.weight"), rank, alpha)
        self.k_proj = LoRALinear(g("self_attn.k_proj.weight"), rank, alpha)
        self.v_proj = LoRALinear(g("self_attn.v_proj.weight"), rank, alpha)
        self.o_proj = LoRALinear(g("self_attn.o_proj.weight"), rank, alpha)
        self.q_norm = RMSNorm(g("self_attn.q_norm.weight"), eps)
        self.k_norm = RMSNorm(g("self_attn.k_norm.weight"), eps)
        self.gate_proj = LoRALinear(g("mlp.gate_proj.weight"), rank, alpha)
        self.up_proj = LoRALinear(g("mlp.up_proj.weight"), rank, alpha)
        self.down_proj = LoRALinear(g("mlp.down_proj.weight"), rank, alpha)
        gs = cfg["dflash_config"]["conv_group_size"]
        self.attention_conv = GroupedConv(g("attention_conv.base_kernel"), g("attention_conv.kernel_projection.weight"), gs, rank, alpha)
        self.mlp_conv = GroupedConv(g("mlp_conv.base_kernel"), g("mlp_conv.kernel_projection.weight"), gs, rank, alpha)
        self.nh = cfg["num_attention_heads"]
        self.nkv = cfg["num_key_value_heads"]
        self.hd = cfg["head_dim"]


class DFlash2(nn.Module):
    def __init__(self, ckpt_dir, lm_head_w, rank=128, alpha=128.0, dtype=torch.bfloat16):
        super().__init__()
        ckpt_dir = Path(ckpt_dir)
        cfg = json.loads((ckpt_dir / "config.json").read_text())
        self.cfg = cfg
        dc = cfg["dflash_config"]
        self.block = dc["block_size"]
        self.mask_id = dc["mask_token_id"]
        self.top_k = dc["selector_top_k"]
        self.window = cfg["sliding_window"]
        self.theta = cfg["rope_parameters"]["rope_theta"]
        self.hd = cfg["head_dim"]
        sd = {k: v.to(dtype) for k, v in load_file(str(ckpt_dir / "model.safetensors")).items()}
        eps = cfg["rms_norm_eps"]
        self.fc = LoRALinear(sd["fc.weight"], rank, alpha)
        self.hidden_norm = RMSNorm(sd["hidden_norm.weight"], eps)
        self.norm = RMSNorm(sd["norm.weight"], eps)
        self.layers = nn.ModuleList([Layer(sd, i, cfg, rank, alpha) for i in range(cfg["num_hidden_layers"])])
        self.sel_prev = nn.Parameter(sd["candidate_selector.predecessor_codebook"].float())
        self.sel_next = nn.Parameter(sd["candidate_selector.successor_codebook"].float())
        self.sel_hidden = nn.Parameter(sd["candidate_selector.hidden_projection.weight"].float())
        self.lm_head = nn.Parameter(lm_head_w.to(dtype), requires_grad=False)  # [V, C]
        self.eps = eps

    # ---- context: target features -> per-layer K/V -------------------------------------
    def context_kv(self, feats, pos):
        """feats [L, n_feat] -> list of (k [L, nkv, hd] roped, v [L, nkv, hd])."""
        g = self.hidden_norm(self.fc(feats))
        cos, sin = rope_cos_sin(pos, self.hd, self.theta, feats.device)
        out = []
        for ly in self.layers:
            k = ly.k_proj(g).view(-1, ly.nkv, ly.hd)
            v = ly.v_proj(g).view(-1, ly.nkv, ly.hd)
            k = apply_rope(ly.k_norm(k), cos, sin)
            out.append((k, v))
        return out

    # ---- block forward ------------------------------------------------------------------
    def _layer(self, ly, x, kc, vc, cos, sin, bias, B):
        h = ly.input_layernorm(x)
        h, attn_dyn = ly.attention_conv.prepare(h, B)
        q = ly.q_proj(h).view(-1, ly.nh, ly.hd)
        k = ly.k_proj(h).view(-1, ly.nkv, ly.hd)
        v = ly.v_proj(h).view(-1, ly.nkv, ly.hd)
        q = apply_rope(ly.q_norm(q), cos, sin)
        k = apply_rope(ly.k_norm(k), cos, sin)
        rep = ly.nh // ly.nkv
        K = torch.cat([kc, k], 0)
        V = torch.cat([vc, v], 0)
        pad = bias.shape[-1] - K.shape[0]
        if pad:
            K = F.pad(K, (0, 0, 0, 0, 0, pad))
            V = F.pad(V, (0, 0, 0, 0, 0, pad))
        K = K.transpose(0, 1).repeat_interleave(rep, 0)[None]  # [1, nh, S, hd]
        V = V.transpose(0, 1).repeat_interleave(rep, 0)[None]
        Q = q.transpose(0, 1)[None]
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
            o = F.scaled_dot_product_attention(Q, K, V, attn_mask=bias.to(Q.dtype)[None, None], scale=1.0 / math.sqrt(ly.hd))
        o = o[0].transpose(0, 1).reshape(-1, ly.nh * ly.hd)
        o = ly.o_proj(o)
        o = ly.attention_conv.finish(o, attn_dyn, B)
        x = x + o
        h = ly.post_attention_layernorm(x)
        h, mlp_dyn = ly.mlp_conv.prepare(h, B)
        h = ly.down_proj(F.silu(ly.gate_proj(h)) * ly.up_proj(h))
        h = ly.mlp_conv.finish(h, mlp_dyn, B)
        return x + h

    def blocks_forward(self, ctx_kv, block_emb, anchors, attn_mask, B, ckpt=True):
        """block_emb [A*B, C]; anchors [A]; attn_mask [A*B, L + A*B] bool (True = attend); B = runtime block length."""
        pos = (anchors[:, None] + torch.arange(B, device=anchors.device)[None, :]).reshape(-1)
        cos, sin = rope_cos_sin(pos, self.hd, self.theta, block_emb.device)
        S = attn_mask.shape[1]
        S_pad = (S + 15) // 16 * 16
        bias = torch.full((attn_mask.shape[0], S_pad), float("-inf"), device=attn_mask.device, dtype=torch.float32)
        bias[:, :S].masked_fill_(attn_mask, 0.0)
        x = block_emb
        for ly, (kc, vc) in zip(self.layers, ctx_kv):
            if ckpt and torch.is_grad_enabled():
                x = checkpoint(self._layer, ly, x, kc, vc, cos, sin, bias, B, use_reentrant=False)
            else:
                x = self._layer(ly, x, kc, vc, cos, sin, bias, B)
        return self.norm(x)  # [A*B, C]

    def selector_scores(self, hidden, cand_ids, unary, pred_ids):
        """hidden [N, C]; cand_ids/unary [N, K]; pred_ids [N] -> scores [N, K]."""
        gate = F.linear(hidden.float(), self.sel_hidden)  # [N, R]
        ctx = self.sel_prev[pred_ids] * gate
        succ = self.sel_next[cand_ids]  # [N, K, R]
        return unary.float() + torch.einsum("nr,nkr->nk", ctx, succ)

    # ---- export --------------------------------------------------------------------------
    @torch.no_grad()
    def export_state_dict(self):
        sd = {}
        sd["fc.weight"] = self.fc.merged_weight()
        sd["hidden_norm.weight"] = self.hidden_norm.weight
        sd["norm.weight"] = self.norm.weight
        for i, ly in enumerate(self.layers):
            p = f"layers.{i}."
            sd[p + "input_layernorm.weight"] = ly.input_layernorm.weight
            sd[p + "post_attention_layernorm.weight"] = ly.post_attention_layernorm.weight
            for n in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                sd[p + f"self_attn.{n}.weight"] = getattr(ly, n).merged_weight()
            sd[p + "self_attn.q_norm.weight"] = ly.q_norm.weight
            sd[p + "self_attn.k_norm.weight"] = ly.k_norm.weight
            for n in ["gate_proj", "up_proj", "down_proj"]:
                sd[p + f"mlp.{n}.weight"] = getattr(ly, n).merged_weight()
            for n in ["attention_conv", "mlp_conv"]:
                c = getattr(ly, n)
                sd[p + f"{n}.base_kernel"] = c.base_kernel
                sd[p + f"{n}.kernel_projection.weight"] = c.kernel_projection
        sd["candidate_selector.predecessor_codebook"] = self.sel_prev
        sd["candidate_selector.successor_codebook"] = self.sel_next
        sd["candidate_selector.hidden_projection.weight"] = self.sel_hidden
        return {k: v.detach().to(torch.bfloat16).contiguous().cpu() for k, v in sd.items()}


def build_attn_mask(L, anchors, block, window, device):
    """Bool mask [A*B, L + A*B]: block j position i sees ctx keys k < a_j with (a_j+i) - k < window,
    and all B keys of its own block (non-causal)."""
    A = anchors.shape[0]
    qpos = (anchors[:, None] + torch.arange(block, device=device)[None, :]).reshape(-1)  # [A*B]
    qanchor = anchors.repeat_interleave(block)
    kpos = torch.arange(L, device=device)
    ctx = (kpos[None, :] < qanchor[:, None]) & ((qpos[:, None] - kpos[None, :]) < window)
    qblk = torch.arange(A, device=device).repeat_interleave(block)
    own = qblk[:, None] == qblk[None, :]
    return torch.cat([ctx, own], dim=1)
