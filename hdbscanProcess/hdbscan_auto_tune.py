# -*- coding: utf-8 -*-
"""
HDBSCAN auto analysis + auto parameter selection for segment embeddings.

✅ Refactor per your request:
- Stage timing prints
- HDBSCAN trials progress + ETA
- ✅ Parameter search uses SUBSAMPLE (fast)
- ✅ After best params found, run ONE FULL-DATA HDBSCAN to output final labels/probs
- ✅ Avoid DBCV heavy/unstable computation by default (can re-enable)
- ✅ Reduce memory explosion risks: core_dist_n_jobs=1, prediction_data=False during search
- ✅ You can set SEARCH_SAMPLE large for "quality first"

Usage:
  1) conda/pip install numpy scipy scikit-learn hdbscan torch
  2) python hdbscan_auto_tune.py

Input supports:
  - .pt (torch.save dict or tensor): expects 'user_emb' key OR a tensor directly
  - .npy (numpy array)
  - .npz (expects 'emb' or first array)
"""

import os
import json
import time
import random
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, List

import numpy as np


# -------------------- You only edit these --------------------
EMB_PATH = r"lightgcn_best.pt"    # segment(user) embedding path (pt/npy/npz)
OUT_DIR  = r"hdbscan_out"         # output folder
RANDOM_SEED = 2026

# Preprocess
L2_NORMALIZE = True
PCA_ENABLE = True
PCA_MAX_DIM = 50
PCA_MIN_DIM = 20
PCA_DIM_CANDIDATES = [20, 30, 40, 50]

# kNN stats
KNN_K_LIST = [5, 10, 15, 20]
KNN_SAMPLE = 30000                # sample size for kNN stats; None => all

# Hopkins test (clustering tendency)
HOPKINS_SAMPLE = 8000

# ---------------- HDBSCAN Search Strategy ----------------
# ✅ Search on subset (speed) then final fit on full data (quality)
SEARCH_SAMPLE = 50000             # <-- 你说“采样大一点”：建议 50k~80k；90k 全量会非常慢
SEARCH_WITH_FULL_DATA = False     # 不建议 True（会回到 8 小时级）

# During search we turn off prediction_data to reduce memory overhead
SEARCH_PREDICTION_DATA = False

# Final full fit: if you want membership probabilities, keep True
FINAL_PREDICTION_DATA = True

# HDBSCAN parameter grid
MIN_CLUSTER_SIZE_GRID = [10, 15, 20, 30, 50]  # 你可以加到 120/200（更保守）
MIN_SAMPLES_GRID      = [5, 10, 15]
SELECTION_METHODS     = ["leaf", "eom"]            # leaf => more small clusters

# Soft preferences
TARGET_NOISE_RANGE = (0.40, 0.95)   # 你这里强共识簇 + 大噪声是合理的
TARGET_CLUSTER_RANGE = (30, 8000)
TOPK_CLUSTER_REPORT = 30

# If you want faster search:
MAX_TRIALS = 24  # set e.g. 20/30; None = full grid

# Validity index (DBCV) is heavy and may overflow/slow on Windows -> OFF by default
USE_DBCV = False

# HDBSCAN parallelism: on Windows + joblib, too many processes often explodes memory
CORE_DIST_N_JOBS = 1
# ------------------------------------------------------------


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _fmt_hms(sec: float) -> str:
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def load_embeddings(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()

    if ext == ".npy":
        x = np.load(path)
        if not isinstance(x, np.ndarray):
            raise ValueError("Invalid .npy content.")
        return x

    if ext == ".npz":
        z = np.load(path)
        if "emb" in z:
            return z["emb"]
        for k in z.files:
            return z[k]
        raise ValueError("Empty .npz.")

    if ext == ".pt":
        # torch is optional unless .pt is used
        import torch
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy()
        if isinstance(obj, dict):
            if "user_emb" in obj:
                return obj["user_emb"].detach().cpu().numpy()
            if "emb" in obj:
                t = obj["emb"]
                if hasattr(t, "detach"):
                    return t.detach().cpu().numpy()
                return np.asarray(t)
        raise ValueError(
            f"Unrecognized .pt format: keys={list(obj.keys()) if isinstance(obj, dict) else type(obj)}"
        )

    raise ValueError(f"Unsupported embedding format: {ext}")


def basic_stats(x: np.ndarray) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["shape"] = list(x.shape)
    out["dtype"] = str(x.dtype)
    out["nan_count"] = int(np.isnan(x).sum())
    out["inf_count"] = int(np.isinf(x).sum())
    norms = np.linalg.norm(x, axis=1)
    out["norm"] = {
        "min": float(np.min(norms)),
        "mean": float(np.mean(norms)),
        "p10": float(np.quantile(norms, 0.10)),
        "p50": float(np.quantile(norms, 0.50)),
        "p90": float(np.quantile(norms, 0.90)),
        "max": float(np.max(norms)),
    }
    return out


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def pca_reduce(x: np.ndarray, dim: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    from sklearn.decomposition import PCA
    pca = PCA(n_components=dim, random_state=RANDOM_SEED)
    xr = pca.fit_transform(x)
    info = {
        "dim": dim,
        "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
        "explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_[:min(dim, 10)]],
    }
    return xr, info


def choose_pca_dim(x: np.ndarray) -> Tuple[int, Dict[str, Any]]:
    """
    Heuristic:
      - try candidate dims
      - prefer smallest dim achieving variance >= 0.85
      - else pick best variance among candidates
    """
    best = None
    records = []
    for d in PCA_DIM_CANDIDATES:
        d = max(PCA_MIN_DIM, min(PCA_MAX_DIM, int(d)))
        _, info = pca_reduce(x, d)
        records.append(info)
        s = info["explained_variance_ratio_sum"]
        if best is None or s > best[1]:
            best = (d, s)

    ok = [r for r in records if r["explained_variance_ratio_sum"] >= 0.85]
    if ok:
        chosen = min(ok, key=lambda r: r["dim"])
        return int(chosen["dim"]), {"candidates": records, "chosen": chosen}

    d_best, _ = best
    chosen = [r for r in records if r["dim"] == d_best][0]
    return int(chosen["dim"]), {"candidates": records, "chosen": chosen}


def knn_distance_stats(x: np.ndarray, k_list: List[int], sample_n: Optional[int]) -> Dict[str, Any]:
    from sklearn.neighbors import NearestNeighbors

    n = x.shape[0]
    if sample_n is not None and sample_n < n:
        idx = np.random.choice(n, size=sample_n, replace=False)
        xs = x[idx]
    else:
        xs = x

    out = {"sample_n": int(xs.shape[0]), "k_stats": {}}

    max_k = max(k_list)
    nn = NearestNeighbors(n_neighbors=max_k + 1, metric="cosine", n_jobs=-1)
    nn.fit(xs)

    t0 = time.time()
    dists, _ = nn.kneighbors(xs, return_distance=True)
    t1 = time.time()

    for k in k_list:
        kth = dists[:, k]  # k-th neighbor (since 0 is self)
        out["k_stats"][str(k)] = {
            "min": float(np.min(kth)),
            "mean": float(np.mean(kth)),
            "p10": float(np.quantile(kth, 0.10)),
            "p50": float(np.quantile(kth, 0.50)),
            "p90": float(np.quantile(kth, 0.90)),
            "max": float(np.max(kth)),
        }

    out["kneighbors_elapsed_sec"] = float(t1 - t0)
    return out


def hopkins_statistic(x: np.ndarray, sample_n: int) -> float:
    """
    Hopkins statistic:
      ~0.5: random
      closer to 1: clusterable
      closer to 0: regularly spaced
    Uses Euclidean in current space (OK after PCA + normalize).
    """
    from sklearn.neighbors import NearestNeighbors

    n, d = x.shape
    m = min(sample_n, n)
    idx = np.random.choice(n, size=m, replace=False)
    X_m = x[idx]

    mins = np.min(x, axis=0)
    maxs = np.max(x, axis=0)
    U = np.random.uniform(low=mins, high=maxs, size=(m, d))

    nn = NearestNeighbors(n_neighbors=2, metric="euclidean", n_jobs=-1)
    nn.fit(x)

    dist_x, _ = nn.kneighbors(X_m, return_distance=True)
    w = dist_x[:, 1]

    dist_u, _ = nn.kneighbors(U, return_distance=True)
    u = dist_u[:, 0]

    H = float(np.sum(u) / (np.sum(u) + np.sum(w) + 1e-12))
    return H


@dataclass
class ClusterQuality:
    n_clusters: int
    noise_ratio: float
    mean_prob: float
    mean_persistence: float
    compactness: float
    separation: float
    dbcv: Optional[float]
    score: float


def compute_compactness_separation(
    x: np.ndarray, labels: np.ndarray, sample_per_cluster: int = 200
) -> Tuple[float, float]:
    """
    Compactness: avg cosine sim to centroid within clusters (higher better).
    Separation: avg cosine distance between cluster centroids (higher better).
    Requires x to be L2-normalized.
    """
    mask = labels >= 0
    if mask.sum() == 0:
        return 0.0, 0.0

    x_in = x[mask]
    y_in = labels[mask]
    clusters = np.unique(y_in)

    centroids = []
    weights = []
    compact_vals = []

    for c in clusters:
        idx = np.where(y_in == c)[0]
        weights.append(len(idx))
        if len(idx) > sample_per_cluster:
            idx = np.random.choice(idx, size=sample_per_cluster, replace=False)
        xc = x_in[idx]
        mu = np.mean(xc, axis=0)
        mu = mu / (np.linalg.norm(mu) + 1e-12)
        centroids.append(mu)
        sims = (xc @ mu)
        compact_vals.append(float(np.mean(sims)))

    compactness = float(np.average(compact_vals, weights=weights))

    C = np.vstack(centroids)
    sim_mat = C @ C.T
    if C.shape[0] <= 1:
        separation = 0.0
    else:
        triu = sim_mat[np.triu_indices(C.shape[0], k=1)]
        separation = float(np.mean(1.0 - triu))
    return compactness, separation


def evaluate_hdbscan(x: np.ndarray, clusterer) -> ClusterQuality:
    labels = clusterer.labels_
    probs = getattr(clusterer, "probabilities_", None)

    n = len(labels)
    noise = int(np.sum(labels < 0))
    noise_ratio = noise / max(n, 1)

    if probs is None:
        mean_prob = 0.0
    else:
        m = labels >= 0
        mean_prob = float(np.mean(probs[m])) if m.sum() > 0 else 0.0

    pers = getattr(clusterer, "cluster_persistence_", None)
    if pers is None or len(pers) == 0:
        mean_persistence = 0.0
    else:
        mean_persistence = float(np.mean(pers))

    compactness, separation = compute_compactness_separation(x, labels)

    dbcv = None
    if USE_DBCV:
        try:
            from hdbscan.validity import validity_index
            # may overflow / be slow -> use only if you insist
            dbcv = float(validity_index(x, labels))
        except Exception:
            dbcv = None

    n_clusters = int(len(set(labels)) - (1 if -1 in labels else 0))

    noise_lo, noise_hi = TARGET_NOISE_RANGE
    cl_lo, cl_hi = TARGET_CLUSTER_RANGE

    noise_pen = 0.0
    if noise_ratio < noise_lo:
        noise_pen += (noise_lo - noise_ratio) * 2.0
    if noise_ratio > noise_hi:
        noise_pen += (noise_ratio - noise_hi) * 1.5

    cl_pen = 0.0
    if n_clusters < cl_lo:
        cl_pen += (cl_lo - n_clusters) / max(cl_lo, 1)
    if n_clusters > cl_hi:
        cl_pen += (n_clusters - cl_hi) / max(cl_hi, 1)

    # ✅ Score: do NOT rely on DBCV by default
    term_dbcv = (dbcv if dbcv is not None else 0.0)
    score = (
        (0.8 * term_dbcv) +
        (1.4 * mean_persistence) +
        (0.9 * mean_prob) +
        (1.3 * compactness) +
        (0.7 * separation) -
        (2.2 * noise_pen) -
        (1.6 * cl_pen)
    )

    return ClusterQuality(
        n_clusters=n_clusters,
        noise_ratio=float(noise_ratio),
        mean_prob=float(mean_prob),
        mean_persistence=float(mean_persistence),
        compactness=float(compactness),
        separation=float(separation),
        dbcv=dbcv,
        score=float(score),
    )


def _subsample_for_search(x: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Returns (x_search, idx) where idx maps x_search back to x.
    If SEARCH_WITH_FULL_DATA is True, returns full x.
    """
    n = x.shape[0]
    if SEARCH_WITH_FULL_DATA:
        return x, None

    m = int(SEARCH_SAMPLE) if SEARCH_SAMPLE is not None else n
    m = min(m, n)
    if m >= n:
        return x, None

    idx = np.random.choice(n, size=m, replace=False)
    return x[idx], idx


def run_hdbscan_search(x_full: np.ndarray) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Search params on x_search (subsample), return:
      - search_result dict
      - best_params dict {mcs, ms, method}
    """
    import hdbscan

    x_search, idx = _subsample_for_search(x_full)
    print(f"[HDBSCAN] search data: {x_search.shape[0]:,} / full {x_full.shape[0]:,}")

    trials = []
    grid = [(mcs, ms, sm) for mcs in MIN_CLUSTER_SIZE_GRID for ms in MIN_SAMPLES_GRID for sm in SELECTION_METHODS]

    if MAX_TRIALS is not None and MAX_TRIALS < len(grid):
        grid = random.sample(grid, MAX_TRIALS)

    total = len(grid)
    print(f"[HDBSCAN] total trials: {total} (MAX_TRIALS={MAX_TRIALS})")

    best = None
    best_params = None
    search_start = time.time()

    for t, (mcs, ms, sm) in enumerate(grid, 1):
        trial_start = time.time()

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=int(mcs),
            min_samples=int(ms),
            metric="euclidean",              # works well after normalize+PCA
            cluster_selection_method=sm,
            prediction_data=SEARCH_PREDICTION_DATA,
            core_dist_n_jobs=int(CORE_DIST_N_JOBS),
        )

        fit_t0 = time.time()
        _ = clusterer.fit_predict(x_search)
        fit_elapsed = time.time() - fit_t0

        q = evaluate_hdbscan(x_search, clusterer)

        rec = {
            "trial": t,
            "min_cluster_size": int(mcs),
            "min_samples": int(ms),
            "selection_method": sm,
            "elapsed_sec": float(time.time() - trial_start),
            "fit_elapsed_sec": float(fit_elapsed),
            "quality": {
                "n_clusters": q.n_clusters,
                "noise_ratio": q.noise_ratio,
                "mean_prob": q.mean_prob,
                "mean_persistence": q.mean_persistence,
                "compactness": q.compactness,
                "separation": q.separation,
                "dbcv": q.dbcv,
                "score": q.score,
            }
        }
        trials.append(rec)

        if best is None or q.score > best["quality"]["score"]:
            best = rec
            best_params = {
                "min_cluster_size": int(mcs),
                "min_samples": int(ms),
                "selection_method": sm,
            }

        spent = time.time() - search_start
        avg = spent / t
        eta = avg * (total - t)
        progress = 100.0 * t / total

        print(
            f"[TRIAL {t:03d}/{total} {progress:5.1f}%] "
            f"mcs={mcs:<3d} ms={ms:<3d} sel={sm:<4s} "
            f"clusters={q.n_clusters:<5d} noise={q.noise_ratio:.3f} "
            f"p={q.mean_prob:.3f} pers={q.mean_persistence:.3f} "
            f"dbcv={q.dbcv if q.dbcv is not None else 'NA'} score={q.score:.4f} "
            f"fit={fit_elapsed:.1f}s  avg={avg:.1f}s  ETA={_fmt_hms(eta)}",
            flush=True
        )

    result = {
        "search_on_subsample": (idx is not None),
        "search_sample_n": int(x_search.shape[0]),
        "full_n": int(x_full.shape[0]),
        "use_dbcv": bool(USE_DBCV),
        "search_prediction_data": bool(SEARCH_PREDICTION_DATA),
        "core_dist_n_jobs": int(CORE_DIST_N_JOBS),
        "best": best,
        "best_params": best_params,
        "trials": trials,
    }
    return result, best_params


def fit_final_hdbscan(x_full: np.ndarray, best_params: Dict[str, Any]):
    """
    Run one full-data HDBSCAN with best_params.
    """
    import hdbscan

    print("\n[HDBSCAN-FINAL] fitting on FULL data with best params ...")
    print("[HDBSCAN-FINAL] params:", best_params)

    final = hdbscan.HDBSCAN(
        min_cluster_size=int(best_params["min_cluster_size"]),
        min_samples=int(best_params["min_samples"]),
        metric="euclidean",
        cluster_selection_method=best_params["selection_method"],
        prediction_data=FINAL_PREDICTION_DATA,
        core_dist_n_jobs=int(CORE_DIST_N_JOBS),
    )
    t0 = time.time()
    final.fit(x_full)
    print(f"[HDBSCAN-FINAL] done. elapsed={time.time()-t0:.1f}s")
    return final


def summarize_clusters(clusterer, out_path: str, topk: int = 30):
    labels = clusterer.labels_
    probs = getattr(clusterer, "probabilities_", None)
    pers = getattr(clusterer, "cluster_persistence_", None)

    n = len(labels)
    noise = int(np.sum(labels < 0))
    clusters = sorted([c for c in np.unique(labels) if c >= 0])

    sizes = {int(c): int(np.sum(labels == c)) for c in clusters}
    top = sorted(sizes.items(), key=lambda kv: kv[1], reverse=True)[:topk]

    pers_map = {}
    if pers is not None and len(pers) > 0:
        for i, pv in enumerate(pers):
            pers_map[int(i)] = float(pv)

    out = {
        "n_points": int(n),
        "noise_points": int(noise),
        "noise_ratio": float(noise / max(n, 1)),
        "n_clusters": int(len(clusters)),
        "top_clusters_by_size": [
            {
                "cluster_id": int(cid),
                "size": int(sz),
                "persistence": pers_map.get(int(cid), None),
                "mean_prob": float(np.mean(probs[labels == cid])) if probs is not None else None,
            }
            for cid, sz in top
        ],
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


def main():
    set_seed(RANDOM_SEED)
    ensure_dir(OUT_DIR)

    # ----------------- Load -----------------
    print("Loading embeddings:", EMB_PATH)
    t_load = time.time()
    x = load_embeddings(EMB_PATH)
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"Embeddings must be 2D, got shape={x.shape}")
    print(f"[LOAD] done. N={x.shape[0]:,} D={x.shape[1]} elapsed={time.time()-t_load:.1f}s")

    stat0 = basic_stats(x)
    print("[BASIC]", stat0)
    if stat0["nan_count"] > 0 or stat0["inf_count"] > 0:
        raise ValueError("Embeddings contain NaN/Inf. Fix training/export first.")

    # ----------------- Preprocess -----------------
    if L2_NORMALIZE:
        t_norm = time.time()
        x = l2_normalize(x)
        print(f"[NORM] L2 normalize done. elapsed={time.time()-t_norm:.1f}s")

    pca_info = None
    x_work = x
    if PCA_ENABLE:
        t_pca = time.time()
        dim, pca_info = choose_pca_dim(x_work)
        print(f"[PCA] choosing dim={dim}, detail={pca_info['chosen']}")
        x_work, pca_fit_info = pca_reduce(x_work, dim)
        x_work = l2_normalize(x_work)
        print(f"[PCA] done. elapsed={time.time()-t_pca:.1f}s")

        with open(os.path.join(OUT_DIR, "pca_info.json"), "w", encoding="utf-8") as f:
            json.dump(pca_info, f, ensure_ascii=False, indent=2)
        with open(os.path.join(OUT_DIR, "pca_fit_info.json"), "w", encoding="utf-8") as f:
            json.dump(pca_fit_info, f, ensure_ascii=False, indent=2)

    # ----------------- kNN stats -----------------
    print("[kNN] computing distance stats ...")
    t_knn = time.time()
    knn_stats = knn_distance_stats(x_work, KNN_K_LIST, KNN_SAMPLE)
    with open(os.path.join(OUT_DIR, "knn_stats.json"), "w", encoding="utf-8") as f:
        json.dump(knn_stats, f, ensure_ascii=False, indent=2)
    print(f"[kNN] done. elapsed={time.time()-t_knn:.1f}s, kneighbors={knn_stats.get('kneighbors_elapsed_sec', None)}s")
    print("[kNN]", knn_stats["k_stats"])

    # ----------------- Hopkins -----------------
    print("[HOPKINS] computing ...")
    t_h = time.time()
    H = hopkins_statistic(x_work, HOPKINS_SAMPLE)
    with open(os.path.join(OUT_DIR, "hopkins.json"), "w", encoding="utf-8") as f:
        json.dump({"hopkins": H, "sample": HOPKINS_SAMPLE}, f, ensure_ascii=False, indent=2)
    print(f"[HOPKINS] done. elapsed={time.time()-t_h:.1f}s, H={H:.4f} (0.5≈random, closer to 1 => more clusterable)")

    # ----------------- HDBSCAN search (subsample) -----------------
    print("[HDBSCAN] searching best params ...")
    t_search = time.time()
    search_result, best_params = run_hdbscan_search(x_work)
    print(f"[HDBSCAN] search done. elapsed={time.time()-t_search:.1f}s")

    with open(os.path.join(OUT_DIR, "hdbscan_search.json"), "w", encoding="utf-8") as f:
        json.dump(search_result, f, ensure_ascii=False, indent=2)

    print("\n[BEST PARAMS]")
    print(json.dumps(best_params, ensure_ascii=False, indent=2))

    # ----------------- Final full fit (quality) -----------------
    t_final = time.time()
    final_clusterer = fit_final_hdbscan(x_work, best_params)
    print(f"[HDBSCAN-FINAL] total elapsed={time.time()-t_final:.1f}s")

    # ----------------- Save artifacts -----------------
    labels = final_clusterer.labels_.astype(np.int32)
    probs = getattr(final_clusterer, "probabilities_", None)

    np.save(os.path.join(OUT_DIR, "labels.npy"), labels)
    if probs is not None:
        np.save(os.path.join(OUT_DIR, "probs.npy"), probs.astype(np.float32))

    # Save reduced embedding used for clustering
    np.save(os.path.join(OUT_DIR, "emb_used.npy"), x_work.astype(np.float32))

    summarize_clusters(final_clusterer, os.path.join(OUT_DIR, "cluster_summary.json"), TOPK_CLUSTER_REPORT)

    # Overall report
    report = {
        "emb_path": EMB_PATH,
        "out_dir": OUT_DIR,
        "basic_stats": stat0,
        "preprocess": {
            "l2_normalize": L2_NORMALIZE,
            "pca_enable": PCA_ENABLE,
            "pca_info": pca_info,
        },
        "knn_stats": knn_stats,
        "hopkins": H,
        "hdbscan": {
            "search_result": "hdbscan_search.json",
            "best_params": best_params,
            "search_sample": SEARCH_SAMPLE,
            "search_prediction_data": SEARCH_PREDICTION_DATA,
            "final_prediction_data": FINAL_PREDICTION_DATA,
            "core_dist_n_jobs": CORE_DIST_N_JOBS,
            "use_dbcv": USE_DBCV,
        },
        "artifacts": {
            "labels": "labels.npy",
            "probs": "probs.npy" if probs is not None else None,
            "emb_used": "emb_used.npy",
            "cluster_summary": "cluster_summary.json",
        }
    }
    with open(os.path.join(OUT_DIR, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n[DONE]")
    print("Saved to:", OUT_DIR)
    print(" - report.json")
    print(" - labels.npy", (" / probs.npy" if probs is not None else ""))
    print(" - emb_used.npy")
    print(" - cluster_summary.json")
    print(" - hdbscan_search.json")


if __name__ == "__main__":
    main()
