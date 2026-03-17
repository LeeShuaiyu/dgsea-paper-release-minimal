# dgsea_torch.py — GPU-accelerated (PyTorch) differentiable GSEA
# Project: L1000 dGSEA
# Requirements: torch >= 1.10
#
# Features
# - 接受 numpy / list / torch，自动在函数入口转换为 torch.Tensor
# - dES / dNES (Nyström + Window 变体 "nyswin" 推荐)
# - Classical GSEA (ES/NES) in torch（便于 κ 校准与一致性）
# - 置换并行 + 分块计算；避免 G×G 全矩阵（大 G 时使用 Nyström/Window）
# - 统一别名（无 _torch 后缀），与 CPU 端保持一致的 3 返回值签名
#
# 注意：
# - 大规模时避免 "orig"（O(G^2)）；首选 "nyswin"
# - 训练期用 dES（无置换），显著性用 dNES（置换）

from typing import Optional, Tuple
import math
import torch

__all__ = [
    "recommended_hparams",
    "set_device",
    "get_device",
    "shared_permutations_torch",
    "classical_gsea_es_torch",
    "classical_gsea_nes_with_perms_torch",
    "dgsea_des_components_torch",
    "dgsea_des_torch",
    "dgsea_dnes_with_perms_torch",
    # unified alias（无 _torch 后缀）
    "shared_permutations",
    "classical_gsea_es",
    "classical_gsea_nes_with_perms",
    "dgsea_des_components",
    "dgsea_des",
    "dgsea_dnes_with_perms",
]

# ---------------------- device / dtype helpers ----------------------

_DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
_DTYPE  = torch.float32

def set_device(device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> None:
    """Set default device and dtype used by this module."""
    global _DEVICE, _DTYPE
    if device is not None:
        _DEVICE = device
    if dtype is not None:
        _DTYPE = dtype

def get_device() -> Tuple[torch.device, torch.dtype]:
    return _DEVICE, _DTYPE

def _to(x, device=None, dtype=None) -> torch.Tensor:
    """Convert numpy/list/torch to torch.Tensor on target device/dtype."""
    dev = _DEVICE if device is None else device
    dt  = _DTYPE  if dtype  is None else dtype
    if isinstance(x, torch.Tensor):
        return x.to(device=dev, dtype=dt)
    return torch.as_tensor(x, device=dev, dtype=dt)

def _to_long(x, device=None) -> torch.Tensor:
    dev = _DEVICE if device is None else device
    if isinstance(x, torch.Tensor):
        return x.to(device=dev, dtype=torch.long)
    return torch.as_tensor(x, device=dev, dtype=torch.long)

# ---------------------- utilities ----------------------

def recommended_hparams() -> dict:
    return dict(
        P=1.0,
        TAU_RANK=1.2,
        TAU_PREFIX=1.0,
        TAU_ABS=0.6,
        K_CLASSICAL=300,
        K_DIFF=300,
        TRIM_FRAC=0.15,
        SHRINK_LAMBDA=0.10,
        SPLIT_RATIO=0.5,
        CALIBRATE_KAPPA=True,
        M_BATCH=32,
        G_DEFAULT=400,
        SET_SIZE=30,
        NYSTROM_M=512,
        WINDOW_FRAC=0.10,
        WINDOW_MARGIN=50
    )

def shared_permutations_torch(G: int, K: int, seed: int = 123, device: Optional[torch.device] = None) -> torch.Tensor:
    """Deterministically generate K permutations of range(G) as a [K, G] int64 tensor."""
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    perms = torch.stack([torch.randperm(G, generator=g) for _ in range(K)], dim=0)  # on CPU
    return perms.to(device if device is not None else _DEVICE, dtype=torch.long)

def _sign_specific_mean_abs_torch(v: torch.Tensor, sign: int, trim: float = 0.1, shrink_lambda: float = 0.0) -> torch.Tensor:
    """Robust mean of |v| conditional on sign(v) = sign (1 for positive, -1 for negative)."""
    if sign >= 0:
        pool = v[v > 0].abs()
    else:
        pool = v[v < 0].abs()
    if pool.numel() == 0:
        return torch.tensor(1.0, device=v.device, dtype=v.dtype)
    t = max(0.0, min(0.49, float(trim)))
    if pool.numel() == 1:
        tm = pool.mean(); wz = tm
    else:
        q = torch.tensor([t, 1.0 - t], device=pool.device, dtype=pool.dtype)
        lo, hi = torch.quantile(pool, q)
        mask = (pool >= lo) & (pool <= hi)
        tm = pool[mask].mean() if mask.any() else pool.mean()
        pool_w = pool.clone()
        pool_w[pool_w < lo] = lo
        pool_w[pool_w > hi] = hi
        wz = pool_w.mean()
    mu = (1.0 - shrink_lambda) * tm + shrink_lambda * wz
    return torch.clamp(mu, min=1e-12)

# ---------------------- classical GSEA (torch) ----------------------

def classical_gsea_es_torch(s, g, p: float = 1.0):
    """
    Classical GSEA ES.
    Inputs:
      s: [B, G] or [G] (numpy/list/torch)
      g: [G] 0/1 (numpy/list/torch)
    Returns:
      ES: [B], C: [B, G], order: [B, G]
    """
    s = _to(s, dtype=_DTYPE)           # [B,G] or [G]
    g = _to(g, dtype=_DTYPE)           # [G]
    if s.dim() == 1:
        s = s.unsqueeze(0)
    if g.dim() != 1:
        g = g.view(-1)
    B, G = s.shape
    g = g.view(1, G).expand(B, G)

    order = torch.argsort(s, dim=1, descending=True)
    s_ord = torch.gather(s, 1, order)
    g_ord = torch.gather(g, 1, order)

    w_hit  = g_ord * torch.pow(s_ord.abs(), p)
    w_miss = 1.0 - g_ord
    denom_hit  = torch.clamp(w_hit.sum(dim=1, keepdim=True),  min=1e-12)
    denom_miss = torch.clamp(w_miss.sum(dim=1, keepdim=True), min=1e-12)

    pref_hit  = torch.cumsum(w_hit,  dim=1) / denom_hit
    pref_miss = torch.cumsum(w_miss, dim=1) / denom_miss
    C = pref_hit - pref_miss

    max_pos, _ = C.max(dim=1)
    min_neg, _ = C.min(dim=1)
    ES = torch.where(max_pos.abs() >= min_neg.abs(), max_pos, min_neg)
    return ES, C, order

def classical_gsea_nes_with_perms_torch(
    s, g, p: float, perms, trim: float = 0.1, shrink_lambda: float = 0.0, split_ratio: float = 0.0
):
    """
    NES = ES / E_null[|ES| | sign], p-value is one-sided by observed sign.
    s: [B,G]/[G]; g: [G]; perms: [K,G] int64 (numpy/torch)
    Returns: NES [B], ES_obs [B], pval [B]
    """
    s = _to(s, dtype=_DTYPE)
    g = _to(g, dtype=_DTYPE)
    perms = _to_long(perms)
    if s.dim() == 1:
        s = s.unsqueeze(0)
    if g.dim() != 1:
        g = g.view(-1)
    B, G = s.shape
    ES_obs, _, _ = classical_gsea_es_torch(s, g, p)  # [B]
    sign = torch.where(ES_obs >= 0, torch.tensor(1, device=s.device), torch.tensor(-1, device=s.device))
    K = int(perms.shape[0])

    chunk = min(K, 128)
    es_null_list = []
    for start in range(0, K, chunk):
        end = min(K, start + chunk)
        idx = perms[start:end].to(device=s.device)          # [kc, G]
        kc  = end - start
        idx_b = idx.unsqueeze(0).expand(B, kc, G)           # [B, kc, G]
        s_b   = s.unsqueeze(1).expand(B, kc, G)             # [B, kc, G]
        s_perm = torch.gather(s_b, 2, idx_b).reshape(B * kc, G)
        es, _, _ = classical_gsea_es_torch(s_perm, g, p)    # [B*kc]
        es_null_list.append(es.view(B, kc))
    es_null = torch.cat(es_null_list, dim=1)                # [B, K]

    denom_list = []; pval_list = []
    if split_ratio > 0.0:
        K1 = max(1, int(K * split_ratio)); K2 = K - K1
    for b in range(B):
        sgn = int(sign[b].item())
        if split_ratio > 0.0:
            vals1 = es_null[b, :K1]; vals2 = es_null[b, K1:]
            denom_b = _sign_specific_mean_abs_torch(vals1, sgn, trim, shrink_lambda)
            comp = (vals2 >= ES_obs[b]) if (sgn > 0) else (vals2 <= ES_obs[b])
            p_b = (comp.sum().float() + 1.0) / (vals2.numel() + 1.0)
        else:
            denom_b = _sign_specific_mean_abs_torch(es_null[b], sgn, trim, shrink_lambda)
            comp = (es_null[b] >= ES_obs[b]) if (sgn > 0) else (es_null[b] <= ES_obs[b])
            p_b = (comp.sum().float() + 1.0) / (es_null.shape[1] + 1.0)
        denom_list.append(denom_b); pval_list.append(p_b)
    denom = torch.stack(denom_list).to(s.device)
    pval  = torch.stack(pval_list).to(s.device)
    NES   = ES_obs / torch.clamp(denom, min=1e-12)
    return NES, ES_obs, pval

# ---------------------- dGSEA (torch) ----------------------

def _soft_rank_full(s: torch.Tensor, tau_rank: float) -> torch.Tensor:
    """Full soft-rank (O(G^2)) — small G only."""
    if s.dim() == 1: s = s.unsqueeze(0)
    # diff[b,i,j] = (s[b,j] - s[b,i]) / tau
    diff = (s.unsqueeze(1) - s.unsqueeze(2)) / float(tau_rank)  # [B,G,G]
    H = torch.sigmoid(diff)
    H = H - torch.diag_embed(torch.diagonal(H, dim1=1, dim2=2))
    r = 1.0 + H.sum(dim=2)
    return r

def _soft_rank_nystrom(s: torch.Tensor, tau_rank: float, m: int = 512, quantile_anchors: bool = True) -> torch.Tensor:
    """
    Nyström soft-rank approximation:
      r ≈ 1 + (G/m) * sum_{a in anchors} sigmoid((a - s)/tau)
    - Anchors来自 s，但 **detach**（不经由 quantile/sort 反传）
    - 兼容不同 PyTorch 版本的 quantile 形状；失败时回落 sort+gather
    """
    if s.dim() == 1:
        s = s.unsqueeze(0)
    B, G = s.shape
    m = min(int(m), G)

    if quantile_anchors:
        qs = torch.linspace(0.0, 1.0, m + 2, device=s.device, dtype=s.dtype)[1:-1]  # [m]
        s_det = s.detach()
        try:
            A = torch.quantile(s_det, qs, dim=1)  # 期望 [B, m]
            if A.shape != (B, m):                 # 兼容老版本 [m, B]
                A = A.transpose(0, 1).contiguous()
        except Exception:
            s_sorted, _ = s_det.sort(dim=1)                          # [B, G]
            idx = torch.clamp((qs * (G - 1)).round().long(), 0, G-1) # [m]
            A = s_sorted.gather(1, idx.unsqueeze(0).expand(B, -1))   # [B, m]
    else:
        idx = torch.randint(0, G, (B, m), device=s.device)
        A = torch.gather(s, 1, idx).detach()                         # [B, m]

    H = torch.sigmoid((A.unsqueeze(1) - s.unsqueeze(2)) / float(tau_rank))  # [B,G,m]
    r = 1.0 + (G / float(m)) * H.sum(dim=2)
    return r

def _choose_window(G: int, r: torch.Tensor, frac: float = 0.1, margin: int = 50) -> int:
    T0 = max(int(frac * G), 1); T0 = min(T0, G // 2)
    with torch.no_grad():
        t_est = int(torch.median(r).item())
    if (t_est <= T0) or (t_est >= G - T0 + 1):
        return T0
    T1 = max(T0, min(max(abs(t_est - 1) + margin, abs(G - t_est) + margin), G // 2))
    return T1

def _running_curve_from_r_torch(s: torch.Tensor, g: torch.Tensor, p: float,
                                r: torch.Tensor, tau_prefix: float, t_grid: torch.Tensor) -> torch.Tensor:
    """Compute soft running curve C_soft(t) on given t_grid."""
    if s.dim() == 1: s = s.unsqueeze(0)
    if g.dim() != 1: g = g.view(-1)
    B, G = s.shape
    g = g.to(device=s.device, dtype=s.dtype).view(1, G).expand(B, G)

    w_hit  = g * torch.pow(s.abs(), p)
    w_miss = (1.0 - g)
    denom_hit  = torch.clamp(w_hit.sum(dim=1, keepdim=True),  min=1e-12)
    denom_miss = torch.clamp(w_miss.sum(dim=1, keepdim=True), min=1e-12)

    H = torch.sigmoid((t_grid.view(1, 1, -1) + 0.5 - r.unsqueeze(-1)) / float(tau_prefix))  # [B,G,T]
    pref_hit  = (w_hit.unsqueeze(-1)  * H).sum(dim=1) / denom_hit
    pref_miss = (w_miss.unsqueeze(-1) * H).sum(dim=1) / denom_miss
    C_soft = pref_hit - pref_miss
    return C_soft

def _signed_softmax_abs(C_soft: torch.Tensor, tau_abs: float) -> torch.Tensor:
    tau_abs = max(float(tau_abs), 1e-12)
    W = torch.softmax(C_soft.abs() / tau_abs, dim=-1)
    dES = (W * C_soft).sum(dim=-1)
    return dES

def dgsea_des_components_torch(
    s, g, p: float = 1.0, tau_rank: float = 1.0, tau_prefix: float = 1.5, tau_abs: float = 0.6,
    variant: str = "nyswin", m: int = 512, frac: float = 0.1, margin: int = 50
):
    """
    Compute dES with intermediate components.
    Inputs accept numpy/list/torch. Returns torch tensors.
    Returns: (dES: [B], C_soft: [B,T], r: [B,G], t_grid: [T])
    """
    s = _to(s, dtype=_DTYPE); g = _to(g, dtype=_DTYPE)
    if s.dim() == 1: s = s.unsqueeze(0)
    B, G = s.shape
    device, dtype = s.device, s.dtype

    if variant == "orig":
        r = _soft_rank_full(s, tau_rank)
        t_grid = torch.arange(1, G + 1, device=device, dtype=dtype)
    elif variant == "nystrom":
        r = _soft_rank_nystrom(s, tau_rank, m=m, quantile_anchors=True)
        t_grid = torch.arange(1, G + 1, device=device, dtype=dtype)
    elif variant == "nyswin":
        r = _soft_rank_nystrom(s, tau_rank, m=m, quantile_anchors=True)
        T = _choose_window(G, r.reshape(-1), frac=frac, margin=margin)
        t_grid = torch.cat([
            torch.arange(1, T + 1, device=device, dtype=dtype),
            torch.arange(G - T + 1, G + 1, device=device, dtype=dtype)
        ], dim=0)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    C_soft = _running_curve_from_r_torch(s, g, p, r, tau_prefix, t_grid)
    dES = _signed_softmax_abs(C_soft, tau_abs)
    return dES, C_soft, r, t_grid

def dgsea_des_torch(
    s, g, p: float = 1.0, tau_rank: float = 1.0, tau_prefix: float = 1.5, tau_abs: float = 0.6,
    variant: str = "nyswin", m: int = 512, frac: float = 0.1, margin: int = 50
):
    dES, _, _, _ = dgsea_des_components_torch(s, g, p, tau_rank, tau_prefix, tau_abs, variant, m, frac, margin)
    return dES

def _kappa_calibration_torch(
    s: torch.Tensor, g: torch.Tensor, p: float,
    tau_rank: float, tau_prefix: float, tau_abs: float,
    perms: torch.Tensor, trim: float, shrink_lambda: float,
    variant: str, m: int, frac: float, margin: int, chunk_size: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:
    if s.dim() == 1: s = s.unsqueeze(0)
    if g.dim() != 1: g = g.view(-1)
    B, G = s.shape
    device = s.device
    K = int(perms.shape[0])
    chunk = min(K, chunk_size)

    des_null_list = []; es_null_list  = []
    for start in range(0, K, chunk):
        end = min(K, start + chunk)
        idx = perms[start:end].to(device=device)           # [kc, G]
        kc  = end - start
        idx_b = idx.unsqueeze(0).expand(B, kc, G)          # [B, kc, G]
        s_b   = s.unsqueeze(1).expand(B, kc, G)            # [B, kc, G]
        s_perm = torch.gather(s_b, 2, idx_b).reshape(B * kc, G)

        dES = dgsea_des_torch(s_perm, g, p, tau_rank, tau_prefix, tau_abs, variant, m, frac, margin)  # [B*kc]
        des_null_list.append(dES.view(B, kc))
        es, _, _ = classical_gsea_es_torch(s_perm, g, p)
        es_null_list.append(es.view(B, kc))

    des_null = torch.cat(des_null_list, dim=1)  # [B, K]
    es_null  = torch.cat(es_null_list,  dim=1)  # [B, K]

    mu_pos_d = torch.stack([_sign_specific_mean_abs_torch(des_null[b], +1, trim, shrink_lambda) for b in range(B)])
    mu_neg_d = torch.stack([_sign_specific_mean_abs_torch(des_null[b], -1, trim, shrink_lambda) for b in range(B)])
    mu_pos_c = torch.stack([_sign_specific_mean_abs_torch(es_null[b],  +1, trim, shrink_lambda) for b in range(B)])
    mu_neg_c = torch.stack([_sign_specific_mean_abs_torch(es_null[b],  -1, trim, shrink_lambda) for b in range(B)])

    k_pos = mu_pos_c / torch.clamp(mu_pos_d, min=1e-12)
    k_neg = mu_neg_c / torch.clamp(mu_neg_d, min=1e-12)
    return k_pos, k_neg

def dgsea_dnes_with_perms_torch(
    s, g, p: float, tau_rank: float, tau_prefix: float, tau_abs: float, perms,
    trim: float = 0.1, shrink_lambda: float = 0.0, split_ratio: float = 0.0, calibrate_kappa: bool = False,
    variant: str = "nyswin", m: int = 512, frac: float = 0.1, margin: int = 50, chunk_size: int = 128
):
    """dNES with permutation-based null. Returns (dNES [B], dES_obs [B], pval [B])."""
    s = _to(s, dtype=_DTYPE); g = _to(g, dtype=_DTYPE); perms = _to_long(perms)
    if s.dim() == 1: s = s.unsqueeze(0)
    if g.dim() != 1: g = g.view(-1)
    B, G = s.shape; device = s.device; K = int(perms.shape[0])

    dES_obs = dgsea_des_torch(s, g, p, tau_rank, tau_prefix, tau_abs, variant, m, frac, margin)  # [B]
    sign = torch.where(dES_obs >= 0, torch.tensor(1, device=device), torch.tensor(-1, device=device))

    chunk = min(K, chunk_size)
    des_null_list = []
    for start in range(0, K, chunk):
        end = min(K, start + chunk)
        idx = perms[start:end].to(device=device)            # [kc, G]
        kc  = end - start
        idx_b = idx.unsqueeze(0).expand(B, kc, G)           # [B, kc, G]
        s_b   = s.unsqueeze(1).expand(B, kc, G)             # [B, kc, G]
        s_perm = torch.gather(s_b, 2, idx_b).reshape(B * kc, G)
        dES = dgsea_des_torch(s_perm, g, p, tau_rank, tau_prefix, tau_abs, variant, m, frac, margin)  # [B*kc]
        des_null_list.append(dES.view(B, kc))
    des_null = torch.cat(des_null_list, dim=1)              # [B, K]

    denom_list = []; pval_list  = []
    if split_ratio > 0.0:
        K1 = max(1, int(K * split_ratio)); K2 = K - K1
    for b in range(B):
        sgn = int(sign[b].item())
        if split_ratio > 0.0:
            vals1 = des_null[b, :K1]; vals2 = des_null[b, K1:]
            denom_b = _sign_specific_mean_abs_torch(vals1, sgn, trim, shrink_lambda)
            if calibrate_kappa:
                k_pos, k_neg = _kappa_calibration_torch(
                    s[b:b+1], g, p, tau_rank, tau_prefix, tau_abs,
                    perms[:K1], trim, shrink_lambda, variant, m, frac, margin, chunk_size
                )
                denom_b = denom_b * (k_pos if sgn > 0 else k_neg)
            comp = (vals2 >= dES_obs[b]) if (sgn > 0) else (vals2 <= dES_obs[b])
            p_b = (comp.sum().float() + 1.0) / (vals2.numel() + 1.0)
        else:
            denom_b = _sign_specific_mean_abs_torch(des_null[b], sgn, trim, shrink_lambda)
            if calibrate_kappa:
                k_pos, k_neg = _kappa_calibration_torch(
                    s[b:b+1], g, p, tau_rank, tau_prefix, tau_abs,
                    perms, trim, shrink_lambda, variant, m, frac, margin, chunk_size
                )
                denom_b = denom_b * (k_pos if sgn > 0 else k_neg)
            comp = (des_null[b] >= dES_obs[b]) if (sgn > 0) else (des_null[b] <= dES_obs[b])
            p_b = (comp.sum().float() + 1.0) / (des_null.shape[1] + 1.0)
        denom_list.append(denom_b); pval_list.append(p_b)

    denom = torch.stack(denom_list).to(device)
    pval  = torch.stack(pval_list).to(device)
    dNES  = dES_obs / torch.clamp(denom, min=1e-12)
    return dNES, dES_obs, pval

# ---------------------- Unified alias API (no "_torch" suffix) ----------------------

def shared_permutations(G: int, K: int, seed: int = 123, device: Optional[torch.device] = None):
    return shared_permutations_torch(G, K, seed=seed, device=device)

def classical_gsea_es(s, g, p=1.0):
    return classical_gsea_es_torch(s, g, p)

def classical_gsea_nes_with_perms(s, g, p, perms, trim=0.1, shrink_lambda=0.0, split_ratio=0.0):
    return classical_gsea_nes_with_perms_torch(s, g, p, perms, trim, shrink_lambda, split_ratio)

def dgsea_des_components(s, g, p=1.0, tau_rank=1.0, tau_prefix=1.5, tau_abs=0.6,
                         variant: str = "nyswin", m: int = 512, frac: float = 0.1, margin: int = 50):
    return dgsea_des_components_torch(s, g, p, tau_rank, tau_prefix, tau_abs, variant, m, frac, margin)

def dgsea_des(s, g, p=1.0, tau_rank=1.0, tau_prefix=1.5, tau_abs=0.6,
              variant: str = "nyswin", m: int = 512, frac: float = 0.1, margin: int = 50):
    dES, C_soft, r, _ = dgsea_des_components_torch(s, g, p, tau_rank, tau_prefix, tau_abs, variant, m, frac, margin)
    return dES, C_soft, r

def dgsea_dnes_with_perms(s, g, p, tau_rank, tau_prefix, tau_abs, perms,
                          trim=0.1, shrink_lambda=0.0, split_ratio=0.0, calibrate_kappa: bool=False,
                          variant: str = "nyswin", m: int = 512, frac: float = 0.1, margin: int = 50, chunk_size: int = 128):
    return dgsea_dnes_with_perms_torch(s, g, p, tau_rank, tau_prefix, tau_abs, perms, trim, shrink_lambda,
                                       split_ratio, calibrate_kappa, variant, m, frac, margin, chunk_size)

# ---------------------- (optional) quick self-test ----------------------

if __name__ == "__main__":
    torch.manual_seed(20250901)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_device(device, torch.float32)

    G = 400
    import numpy as np
    rng = np.random.default_rng(1)
    s = rng.normal(0,1,G)  # numpy → 自动转换
    g = np.zeros(G, int); g[:30] = 1

    hp = recommended_hparams()
    perms = shared_permutations(G, 64, seed=7, device=device)

    NES, ES, p = classical_gsea_nes_with_perms(s, g, p=hp["P"], perms=perms)
    print("Classical NES/ES/p:", float(NES), float(ES), float(p))

    dES, Csoft, r = dgsea_des(s, g, p=hp["P"], tau_rank=hp["TAU_RANK"], tau_prefix=hp["TAU_PREFIX"], tau_abs=hp["TAU_ABS"])
    dNES, dES_obs, dp = dgsea_dnes_with_perms(s, g, p=hp["P"], tau_rank=hp["TAU_RANK"], tau_prefix=hp["TAU_PREFIX"], tau_abs=hp["TAU_ABS"],
                                              perms=perms, trim=hp["TRIM_FRAC"], shrink_lambda=hp["SHRINK_LAMBDA"],
                                              split_ratio=hp["SPLIT_RATIO"], calibrate_kappa=hp["CALIBRATE_KAPPA"])
    print("dGSEA dNES/dES/p:", float(dNES), float(dES_obs), float(dp))
