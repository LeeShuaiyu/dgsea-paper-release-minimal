
# dgsea_core.py — Differentiable GSEA core (NumPy)
# Author: Your Team
# License: MIT
import math, numpy as np

# ---------- Utilities ----------

def logsumexp(x, axis=None, keepdims=False):
    x = np.asarray(x, dtype=float)
    m = np.max(x, axis=axis, keepdims=True)
    y = np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True)) + m
    if axis is None:
        y = np.squeeze(y)
        return float(y)
    if not keepdims:
        y = np.squeeze(y, axis=axis)
    return y

def sigmoid(z):
    z = np.asarray(z, dtype=float)
    z = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))

def _rankdata_average_ties(x):
    x = np.asarray(x)
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    n = x.size
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and np.isclose(xs[j+1], xs[i], rtol=1e-12, atol=1e-12):
            j += 1
        avg = 0.5 * (i + j) + 1.0  # 1-based
        ranks[order[i:j+1]] = avg
        i = j + 1
    return ranks

def spearman_corr(x, y):
    x = np.asarray(x); y = np.asarray(y)
    rx = _rankdata_average_ties(x)
    ry = _rankdata_average_ties(y)
    rx -= rx.mean(); ry -= ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    if denom == 0:
        return 0.0
    return float((rx @ ry) / denom)

def trimmed_mean(a, trim=0.1):
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return 0.0
    t = max(0.0, min(0.49, float(trim)))
    lo, hi = np.quantile(a, [t, 1.0 - t])
    mask = (a >= lo) & (a <= hi)
    return float(a[mask].mean()) if mask.any() else float(a.mean())

def winsorize(a, trim=0.1):
    a = np.asarray(a, dtype=float).copy()
    t = max(0.0, min(0.49, float(trim)))
    lo, hi = np.quantile(a, [t, 1.0 - t])
    a[a < lo] = lo; a[a > hi] = hi
    return a

def bh_fdr(pvals, alpha=0.05):
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranks = np.arange(1, n+1)
    crit = alpha * ranks / n
    passed = p[order] <= crit
    k = np.max(np.where(passed)[0]) + 1 if np.any(passed) else 0
    thresh = crit[k-1] if k > 0 else 0.0
    if k == 0:
        return float(thresh), np.array([], dtype=int)
    return float(thresh), order[:k]

def shared_permutations(G, K, rng):
    perms = np.empty((K, G), dtype=int)
    for k in range(K):
        perms[k] = rng.permutation(G)
    return perms

# ---------- Classical GSEA ----------

def classical_gsea_es(s, g, p=1.0):
    s = np.asarray(s, dtype=float)
    g = np.asarray(g, dtype=int)
    order = np.argsort(-s)
    s_ord = s[order]; g_ord = g[order]
    w_hit  = g_ord * np.power(np.abs(s_ord), p)
    w_miss = 1 - g_ord
    denom_hit  = w_hit.sum()  or 1.0
    denom_miss = w_miss.sum() or 1.0
    pref_hit  = np.cumsum(w_hit)  / denom_hit
    pref_miss = np.cumsum(w_miss) / denom_miss
    C = pref_hit - pref_miss
    max_pos = C.max(); min_neg = C.min()
    ES = max_pos if abs(max_pos) >= abs(min_neg) else min_neg
    return float(ES), C, order

def _sign_specific_mean_abs(null_vals, sign, trim=0.1, shrink_lambda=0.0):
    v = np.asarray(null_vals, dtype=float)
    if sign >= 0:
        pool = np.abs(v[v > 0])
    else:
        pool = np.abs(v[v < 0])
    if pool.size == 0:
        return 1.0
    tm = trimmed_mean(pool, trim=trim)
    wz = winsorize(pool, trim=trim).mean()
    mu = (1.0 - shrink_lambda) * tm + shrink_lambda * wz
    return float(mu) if mu > 1e-12 else 1.0

def classical_gsea_nes_with_perms(s, g, p, perms, trim=0.1, shrink_lambda=0.0, split_ratio=0.0):
    ES, _, _ = classical_gsea_es(s, g, p)
    K = perms.shape[0]
    if split_ratio > 0.0:
        K1 = max(1, int(K * split_ratio)); K2 = K - K1
        vals1 = np.empty(K1); vals2 = np.empty(K2)
        for k in range(K1):
            vals1[k], _, _ = classical_gsea_es(s[perms[k]], g, p)
        for k in range(K2):
            vals2[k], _, _ = classical_gsea_es(s[perms[K1+k]], g, p)
        denom = _sign_specific_mean_abs(vals1, np.sign(ES), trim=trim, shrink_lambda=shrink_lambda)
        if ES >= 0: pval = (np.sum(vals2 >= ES) + 1.0) / (K2 + 1.0)
        else:       pval = (np.sum(vals2 <= ES) + 1.0) / (K2 + 1.0)
    else:
        null_vals = np.empty(K)
        for k in range(K):
            null_vals[k], _, _ = classical_gsea_es(s[perms[k]], g, p)
        denom = _sign_specific_mean_abs(null_vals, np.sign(ES), trim=trim, shrink_lambda=shrink_lambda)
        if ES >= 0: pval = (np.sum(null_vals >= ES) + 1.0) / (K + 1.0)
        else:       pval = (np.sum(null_vals <= ES) + 1.0) / (K + 1.0)
    NES = ES / denom
    return float(NES), float(ES), float(pval)

# ---------- dGSEA core ----------

def soft_rank_descending(s, tau=1.0):
    s = np.asarray(s, dtype=float).reshape(-1,1)
    diff = (s.T - s) / float(tau)
    H = sigmoid(diff)
    np.fill_diagonal(H, 0.0)
    r = 1.0 + H.sum(axis=1)
    return r

def soft_rank_descending_nystrom(s, tau=1.0, m=512, rng=None, quantile_anchors=True):
    s = np.asarray(s, dtype=float).reshape(-1,1)
    G = s.shape[0]
    if rng is None: rng = np.random.default_rng(123)
    m = min(int(m), G)
    if quantile_anchors:
        qs = np.linspace(0.0, 1.0, m+2)[1:-1]
        A = np.quantile(s.ravel(), qs, method="linear").reshape(m,1)
    else:
        idx = rng.choice(G, size=m, replace=False)
        A = s[idx, :]
    H = 1.0 / (1.0 + np.exp(-((A.T - s) / float(tau))))  # (G,m)
    r = 1.0 + (G / float(m)) * H.sum(axis=1)
    return r

def _running_curve_from_r(s, g, p, r, tau_prefix, t_grid=None):
    s = np.asarray(s, dtype=float); g = np.asarray(g, dtype=int)
    G = s.shape[0]
    if t_grid is None:
        t_grid = np.arange(1, G+1, dtype=float)
    w_hit  = g * np.power(np.abs(s), p)
    w_miss = (1 - g).astype(float)
    denom_hit  = w_hit.sum()  or 1.0
    denom_miss = w_miss.sum() or 1.0
    H = sigmoid((t_grid[None,:] + 0.5 - r[:,None]) / float(tau_prefix))  # (G,T)
    pref_hit  = (w_hit[:,None]  * H).sum(axis=0) / denom_hit
    pref_miss = (w_miss[:,None] * H).sum(axis=0) / denom_miss
    C_soft = pref_hit - pref_miss
    return C_soft, t_grid

def signed_softmax_abs(values, tau_abs):
    values = np.asarray(values, dtype=float)
    tau_abs = max(float(tau_abs), 1e-12)
    w = np.exp(np.abs(values) / tau_abs)
    w = w / (w.sum() + 1e-12)
    return float(np.sum(w * values))

def _adaptive_tau_abs(C_soft, c=1.0):
    med = np.median(C_soft)
    mad = np.median(np.abs(C_soft - med)) + 1e-9
    return float(c * mad)

def _choose_window(G, r, frac=0.1, margin=50):
    T0 = max(int(frac*G), 1); T0 = min(T0, G//2)
    t_est = int(np.median(r))  # coarse proxy of peak location
    if (t_est <= T0) or (t_est >= G - T0 + 1):
        return T0
    T1 = max(T0, min(max(abs(t_est-1)+margin, abs(G-t_est)+margin), G//2))
    return T1

def dgsea_des(s, g, p=1.0, tau_rank=1.0, tau_prefix=1.5, tau_abs=0.6):
    r = soft_rank_descending(s, tau=tau_rank)
    C_soft, _ = _running_curve_from_r(s, g, p, r, tau_prefix)
    if tau_abs <= 0:
        tau_abs = _adaptive_tau_abs(C_soft, c=1.0)
    dES = signed_softmax_abs(C_soft, tau_abs)
    return float(dES), C_soft, r

def dgsea_des_nystrom(s, g, p=1.0, tau_rank=1.0, tau_prefix=1.5, tau_abs=0.6, m=512, rng=None):
    r = soft_rank_descending_nystrom(s, tau=tau_rank, m=m, rng=rng, quantile_anchors=True)
    C_soft, _ = _running_curve_from_r(s, g, p, r, tau_prefix)
    if tau_abs <= 0:
        tau_abs = _adaptive_tau_abs(C_soft, c=1.0)
    dES = signed_softmax_abs(C_soft, tau_abs)
    return float(dES), C_soft, r

def dgsea_des_windowed(s, g, p=1.0, tau_rank=1.0, tau_prefix=1.5, tau_abs=0.6, frac=0.1, margin=50):
    r = soft_rank_descending(s, tau=tau_rank)
    G = len(s)
    T = _choose_window(G, r, frac, margin)
    t_grid = np.concatenate([np.arange(1, T+1, dtype=float), np.arange(G-T+1, G+1, dtype=float)], axis=0)
    C_soft, _ = _running_curve_from_r(s, g, p, r, tau_prefix, t_grid=t_grid)
    if tau_abs <= 0:
        tau_abs = _adaptive_tau_abs(C_soft, c=1.0)
    dES = signed_softmax_abs(C_soft, tau_abs)
    return float(dES), C_soft, r

def dgsea_des_nystrom_windowed(s, g, p=1.0, tau_rank=1.0, tau_prefix=1.5, tau_abs=0.6,
                               m=512, frac=0.1, margin=50, rng=None):
    r = soft_rank_descending_nystrom(s, tau=tau_rank, m=m, rng=rng, quantile_anchors=True)
    G = len(s)
    T = _choose_window(G, r, frac, margin)
    t_grid = np.concatenate([np.arange(1, T+1, dtype=float), np.arange(G-T+1, G+1, dtype=float)], axis=0)
    C_soft, _ = _running_curve_from_r(s, g, p, r, tau_prefix, t_grid=t_grid)
    if tau_abs <= 0:
        tau_abs = _adaptive_tau_abs(C_soft, c=1.0)
    dES = signed_softmax_abs(C_soft, tau_abs)
    return float(dES), C_soft, r

# ---------- dNES with shared perms, robust mean, optional calibration ----------

def _dES_stat_variant(x, g, p, tau_rank, tau_prefix, tau_abs, variant, m, frac, margin, rng):
    if variant == "orig":
        return dgsea_des(x, g, p, tau_rank, tau_prefix, tau_abs)[0]
    elif variant == "nystrom":
        return dgsea_des_nystrom(x, g, p, tau_rank, tau_prefix, tau_abs, m, rng)[0]
    elif variant == "window":
        return dgsea_des_windowed(x, g, p, tau_rank, tau_prefix, tau_abs, frac, margin)[0]
    elif variant == "nyswin":
        return dgsea_des_nystrom_windowed(x, g, p, tau_rank, tau_prefix, tau_abs, m, frac, margin, rng)[0]
    else:
        raise ValueError("unknown variant")

def _kappa_calibration(s, g, p, tau_rank, tau_prefix, tau_abs, perms, trim, shrink_lambda, variant, m, frac, margin, rng):
    K = perms.shape[0]
    des0, es0 = [], []
    for k in range(K):
        s0 = s[perms[k]]
        des0.append(_dES_stat_variant(s0, g, p, tau_rank, tau_prefix, tau_abs, variant, m, frac, margin, rng))
        es0.append(classical_gsea_es(s0, g, p)[0])
    des0 = np.asarray(des0); es0 = np.asarray(es0)
    mu_pos_d = _sign_specific_mean_abs(des0, +1, trim, shrink_lambda)
    mu_neg_d = _sign_specific_mean_abs(des0, -1, trim, shrink_lambda)
    mu_pos_c = _sign_specific_mean_abs(es0,  +1, trim, shrink_lambda)
    mu_neg_c = _sign_specific_mean_abs(es0,  -1, trim, shrink_lambda)
    k_pos = mu_pos_c / max(mu_pos_d, 1e-12)
    k_neg = mu_neg_c / max(mu_neg_d, 1e-12)
    return float(k_pos), float(k_neg)

def dgsea_dnes_with_perms(s, g, p, tau_rank, tau_prefix, tau_abs, perms,
                          trim=0.1, shrink_lambda=0.0, split_ratio=0.0,
                          calibrate_kappa=False, variant="orig", m=512, frac=0.1, margin=50, rng=None):
    if rng is None: rng = np.random.default_rng(456)
    dES = _dES_stat_variant(s, g, p, tau_rank, tau_prefix, tau_abs, variant, m, frac, margin, rng)
    sign = +1 if dES >= 0 else -1
    K = perms.shape[0]
    if split_ratio > 0.0:
        K1 = max(1, int(K * split_ratio)); K2 = K - K1
        vals1 = np.empty(K1); vals2 = np.empty(K2)
        for k in range(K1):
            vals1[k] = _dES_stat_variant(s[perms[k]], g, p, tau_rank, tau_prefix, tau_abs, variant, m, frac, margin, rng)
        for k in range(K2):
            vals2[k] = _dES_stat_variant(s[perms[K1+k]], g, p, tau_rank, tau_prefix, tau_abs, variant, m, frac, margin, rng)
        denom = _sign_specific_mean_abs(vals1, sign, trim, shrink_lambda)
        if calibrate_kappa:
            k_pos, k_neg = _kappa_calibration(s, g, p, tau_rank, tau_prefix, tau_abs, perms[:K1], trim, shrink_lambda, variant, m, frac, margin, rng)
            denom *= (k_pos if sign > 0 else k_neg)
        pval = (np.sum(vals2 >= dES) + 1.0) / (K2 + 1.0) if sign > 0 else (np.sum(vals2 <= dES) + 1.0) / (K2 + 1.0)
    else:
        null_vals = np.empty(K)
        for k in range(K):
            null_vals[k] = _dES_stat_variant(s[perms[k]], g, p, tau_rank, tau_prefix, tau_abs, variant, m, frac, margin, rng)
        denom = _sign_specific_mean_abs(null_vals, sign, trim, shrink_lambda)
        if calibrate_kappa:
            k_pos, k_neg = _kappa_calibration(s, g, p, tau_rank, tau_prefix, tau_abs, perms, trim, shrink_lambda, variant, m, frac, margin, rng)
            denom *= (k_pos if sign > 0 else k_neg)
        pval = (np.sum(null_vals >= dES) + 1.0) / (K + 1.0) if sign > 0 else (np.sum(null_vals <= dES) + 1.0) / (K + 1.0)
    dNES = dES / (denom if denom > 1e-12 else 1.0)
    return float(dNES), float(dES), float(pval)

# ---------- Helpers for synthetic experiments ----------

def make_one_set(G=400, set_size=30, up_mean=0.6, up_sd=0.2, rng=None):
    if rng is None: rng = np.random.default_rng(7)
    s = rng.normal(0, 1, size=G)
    g = np.zeros(G, dtype=int)
    idx = rng.choice(G, size=set_size, replace=False)
    g[idx] = 1
    s[idx] += rng.normal(up_mean, up_sd, size=set_size)
    return s, g

def speed_benchmark_variants(G, set_size, K, hp, rng=None, repeats=3):
    if rng is None: rng = np.random.default_rng(123)
    import time
    def bench_variant(variant):
        # shared s,g and perms per repeat to reduce variance
        t = []
        for _ in range(repeats):
            s, g = make_one_set(G, set_size, rng=rng)
            perms = shared_permutations(G, K, rng)
            t0 = time.perf_counter()
            _ = dgsea_dnes_with_perms(s, g, p=hp["P"],
                                      tau_rank=hp["TAU_RANK"], tau_prefix=hp["TAU_PREFIX"], tau_abs=hp["TAU_ABS"],
                                      perms=perms, trim=hp["TRIM_FRAC"], shrink_lambda=hp["SHRINK_LAMBDA"],
                                      split_ratio=hp["SPLIT_RATIO"], calibrate_kappa=hp["CALIBRATE_KAPPA"],
                                      variant=variant, m=hp["NYSTROM_M"], frac=hp["WINDOW_FRAC"], margin=hp["WINDOW_MARGIN"], rng=rng)
            t1 = time.perf_counter()
            t.append(t1 - t0)
        return float(np.median(t))
    times = {}
    for v in ["orig","nystrom","window","nyswin"]:
        times[v] = bench_variant(v)
    return times

# A tiny convenience: recommended defaults derived from our experiments
def recommended_hparams():
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
