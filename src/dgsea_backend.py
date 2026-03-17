# dgsea_backend.py — Unified API for dGSEA (CPU/GPU backends)
# 用法：
#   import dgsea_backend as DG
#   DG.set_backend('gpu')  # 'auto'|'gpu'|'cpu'
#   HP = DG.recommended_hparams()
#   perms = DG.shared_permutations(G, K, seed=7)
#   NES, ES, p = DG.classical_gsea_nes_with_perms(s, g, p=1.0, perms=perms)
#   dES, Csoft, r = DG.dgsea_des(s, g, ...)
#   dNES, dES_obs, dp = DG.dgsea_dnes_with_perms(s, g, ...)

from typing import Optional
import math, inspect

# try GPU backend
_HAS_TORCH = False
_HAS_CUDA  = False
try:
    import torch
    _HAS_TORCH = True
    _HAS_CUDA = torch.cuda.is_available()
except Exception:
    pass

# try CPU backend
try:
    import dgsea_core as CPU
    _HAS_CPU = True
except Exception:
    _HAS_CPU = False

# try GPU module (with unified alias api)
try:
    import dgsea_torch as GPU
    _HAS_GPU_MODULE = True
except Exception:
    _HAS_GPU_MODULE = False

# Prefer torch backend whenever available (works on both CUDA and CPU).
_BACKEND = 'gpu' if (_HAS_TORCH and _HAS_GPU_MODULE) else 'cpu'

def set_backend(backend: str = 'auto', device: Optional[str] = None, dtype: Optional[str] = None):
    """Select backend: 'auto' | 'gpu' | 'cpu'."""
    global _BACKEND
    if backend == 'auto':
        _BACKEND = 'gpu' if (_HAS_TORCH and _HAS_GPU_MODULE) else 'cpu'
    elif backend in ('gpu','cpu'):
        _BACKEND = backend
    else:
        raise ValueError("backend must be 'auto'|'gpu'|'cpu'")
    if _BACKEND == 'gpu':
        dev = torch.device(device) if (device is not None) else torch.device('cuda' if _HAS_CUDA else 'cpu')
        dt  = getattr(torch, dtype) if (dtype is not None and hasattr(torch, dtype)) else torch.float32
        GPU.set_device(dev, dt)

def backend() -> str:
    return _BACKEND

def recommended_hparams() -> dict:
    if _BACKEND == 'gpu':
        return GPU.recommended_hparams()
    else:
        return CPU.recommended_hparams() if (_HAS_CPU and hasattr(CPU,'recommended_hparams')) else {
            "P":1.0, "TAU_RANK":1.2, "TAU_PREFIX":1.0, "TAU_ABS":0.6,
            "K_CLASSICAL":300, "K_DIFF":300, "TRIM_FRAC":0.15, "SHRINK_LAMBDA":0.10,
            "SPLIT_RATIO":0.5, "CALIBRATE_KAPPA":True, "M_BATCH":32, "G_DEFAULT":400,
            "SET_SIZE":30, "NYSTROM_M":512, "WINDOW_FRAC":0.10, "WINDOW_MARGIN":50
        }

def shared_permutations(G: int, K: int, seed: int = 123):
    if _BACKEND == 'gpu':
        fn = getattr(GPU, 'shared_permutations', None) or getattr(GPU, 'shared_permutations_torch', None)
        if fn is None:
            raise AttributeError("dgsea_torch missing shared_permutations[_torch]")
        return fn(G, K, seed)
    else:
        if not _HAS_CPU:
            import numpy as np
            rng = np.random.default_rng(int(seed))
            return np.stack([rng.permutation(G) for _ in range(K)], axis=0)
        fn = getattr(CPU, 'shared_permutations', None)
        if fn is None:
            import numpy as np
            rng = np.random.default_rng(int(seed))
            return np.stack([rng.permutation(G) for _ in range(K)], axis=0)
        import numpy as np
        sig = inspect.signature(fn)
        params = sig.parameters
        if 'rng' in params:
            return fn(G, K, np.random.default_rng(int(seed)))
        elif 'seed' in params:
            return fn(G, K, int(seed))
        else:
            return fn(G, K)

def classical_gsea_es(s, g, p=1.0):
    if _BACKEND == 'gpu':
        return GPU.classical_gsea_es(s, g, p)
    else:
        return CPU.classical_gsea_es(s, g, p)

def classical_gsea_nes_with_perms(s, g, p, perms, trim=0.1, shrink_lambda=0.0, split_ratio=0.0):
    if _BACKEND == 'gpu':
        return GPU.classical_gsea_nes_with_perms(s, g, p, perms, trim, shrink_lambda, split_ratio)
    else:
        fn = getattr(CPU, 'classical_gsea_nes_with_perms', None)
        if fn is None:
            raise RuntimeError("CPU backend lacks classical_gsea_nes_with_perms; please update dgsea_core.")
        ret = fn(s, g, p, perms=perms, trim=trim, shrink_lambda=shrink_lambda, split_ratio=split_ratio)
        # unify to (NES, ES, pval)
        if isinstance(ret, tuple):
            if len(ret) == 3:
                return ret
            elif len(ret) == 4:
                NES, ES, mu, sd = ret
                if sd <= 1e-12:
                    pval = 1.0
                else:
                    z = (ES - mu) / sd
                    ppos = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
                    pval = 1.0 - ppos if ES >= 0 else ppos
                return NES, ES, pval
        return ret, float('nan'), float('nan')

def dgsea_des(s, g, p=1.0, tau_rank=1.0, tau_prefix=1.5, tau_abs=0.6, variant='nyswin', m=512, frac=0.1, margin=50):
    if _BACKEND == 'gpu':
        return GPU.dgsea_des(s, g, p, tau_rank, tau_prefix, tau_abs, variant, m, frac, margin)
    else:
        ret = CPU.dgsea_des(s, g, p, tau_rank, tau_prefix, tau_abs, variant=variant, m=m, frac=frac, margin=margin) \
              if hasattr(CPU.dgsea_des, '__code__') and ('variant' in CPU.dgsea_des.__code__.co_varnames) else \
              CPU.dgsea_des(s, g, p, tau_rank, tau_prefix, tau_abs)
        if isinstance(ret, tuple) and len(ret) == 3:
            return ret
        else:
            return ret, None, None

def dgsea_des_components(s, g, p=1.0, tau_rank=1.0, tau_prefix=1.5, tau_abs=0.6, variant='nyswin', m=512, frac=0.1, margin=50):
    if _BACKEND == 'gpu':
        return GPU.dgsea_des_components(s, g, p, tau_rank, tau_prefix, tau_abs, variant, m, frac, margin)
    else:
        if hasattr(CPU, 'dgsea_des_components'):
            return CPU.dgsea_des_components(s, g, p, tau_rank, tau_prefix, tau_abs, variant=variant, m=m, frac=frac, margin=margin)
        dES, C_soft, r = dgsea_des(s, g, p, tau_rank, tau_prefix, tau_abs, variant=variant, m=m, frac=frac, margin=margin)
        import numpy as np
        G = len(g)
        t_grid = np.arange(1, G+1, dtype=float)
        return dES, C_soft, r, t_grid

def dgsea_dnes_with_perms(s, g, p, tau_rank, tau_prefix, tau_abs, perms,
                          trim=0.1, shrink_lambda=0.0, split_ratio=0.0, calibrate_kappa=False,
                          variant='nyswin', m=512, frac=0.1, margin=50, chunk_size=128):
    if _BACKEND == 'gpu':
        return GPU.dgsea_dnes_with_perms(s, g, p, tau_rank, tau_prefix, tau_abs, perms,
                                         trim, shrink_lambda, split_ratio, calibrate_kappa,
                                         variant, m, frac, margin, chunk_size)
    else:
        fn = getattr(CPU, 'dgsea_dnes_with_perms', None)
        if fn is None:
            raise RuntimeError("CPU backend lacks dgsea_dnes_with_perms; please update dgsea_core.")
        try:
            ret = fn(s, g, p, tau_rank, tau_prefix, tau_abs, perms=perms,
                     trim=trim, shrink_lambda=shrink_lambda, split_ratio=split_ratio,
                     calibrate_kappa=calibrate_kappa, variant=variant, m=m, frac=frac, margin=margin)
        except TypeError:
            ret = fn(s, g, p, tau_rank, tau_prefix, tau_abs, perms=perms,
                     trim=trim, shrink_lambda=shrink_lambda, split_ratio=split_ratio,
                     calibrate_kappa=calibrate_kappa)
        if isinstance(ret, tuple):
            if len(ret) == 3:
                return ret
            elif len(ret) == 4:
                dNES, dES, mu, sd = ret
                if sd <= 1e-12:
                    pval = 1.0
                else:
                    z = (dES - mu) / sd
                    ppos = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
                    pval = 1.0 - ppos if dES >= 0 else ppos
                return dNES, dES, pval
        return ret, float('nan'), float('nan')
