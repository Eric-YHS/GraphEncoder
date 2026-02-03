# utils/covmat.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Dict
import numpy as np

@dataclass
class CovMatResult:
    cov_r: float
    mat_r: float
    cov_p: float
    mat_p: float

def _kabsch_align(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """
    Align P onto Q using Kabsch (rotation only). Both [n,3], already centered.
    Returns rotated P.
    """
    H = P.T @ Q
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    # reflection fix
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    return P @ R

def rmsd_kabsch(P: np.ndarray, Q: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """
    RMSD(P,Q) after Kabsch alignment. P,Q: [n,3], same n.
    mask: [n] bool to select atoms (optional).
    """
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        P = P[mask]
        Q = Q[mask]

    n = P.shape[0]
    if n == 0:
        return float("nan")
    if n == 1:
        d = P[0] - Q[0]
        return float(np.sqrt(np.sum(d * d)))

    Pc = P - P.mean(axis=0, keepdims=True)
    Qc = Q - Q.mean(axis=0, keepdims=True)
    P_aligned = _kabsch_align(Pc, Qc)
    return float(np.sqrt(np.mean(np.sum((P_aligned - Qc) ** 2, axis=1))))

def pairwise_rmsd_matrix(
    preds: Sequence[np.ndarray],
    refs: Sequence[np.ndarray],
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    preds: list of [n,3]
    refs:  list of [n,3]
    returns: [K,M] RMSD matrix
    """
    K, M = len(preds), len(refs)
    out = np.zeros((K, M), dtype=np.float64)
    for i in range(K):
        for j in range(M):
            out[i, j] = rmsd_kabsch(preds[i], refs[j], mask=mask)
    return out

def covmat_from_rmsd(rmsd: np.ndarray, threshold: float) -> CovMatResult:
    """
    rmsd: [K,M] where K=len(preds), M=len(refs)
    COV-R/MAT-R: reference-side coverage/matching
    COV-P/MAT-P: prediction-side coverage/matching
    Definitions align with common conformer-gen literature. :contentReference[oaicite:3]{index=3}
    """
    rmsd = np.asarray(rmsd, dtype=np.float64)
    K, M = rmsd.shape

    # For each reference, best matching prediction
    min_over_pred = np.min(rmsd, axis=0)  # [M]
    cov_r = float(np.mean(min_over_pred <= threshold)) if M > 0 else float("nan")
    mat_r = float(np.mean(min_over_pred)) if M > 0 else float("nan")

    # For each prediction, best matching reference
    min_over_ref = np.min(rmsd, axis=1)  # [K]
    cov_p = float(np.mean(min_over_ref <= threshold)) if K > 0 else float("nan")
    mat_p = float(np.mean(min_over_ref)) if K > 0 else float("nan")

    return CovMatResult(cov_r=cov_r, mat_r=mat_r, cov_p=cov_p, mat_p=mat_p)

def aggregate_covmat(per_mol: Sequence[CovMatResult]) -> Dict[str, float]:
    """
    Aggregate across molecules (mean + median), GeoDiff/related papers often report both.
    """
    if len(per_mol) == 0:
        return {
            "COV-R_mean": float("nan"), "MAT-R_mean": float("nan"),
            "COV-P_mean": float("nan"), "MAT-P_mean": float("nan"),
            "COV-R_median": float("nan"), "MAT-R_median": float("nan"),
            "COV-P_median": float("nan"), "MAT-P_median": float("nan"),
        }

    cov_r = np.array([r.cov_r for r in per_mol], dtype=float)
    mat_r = np.array([r.mat_r for r in per_mol], dtype=float)
    cov_p = np.array([r.cov_p for r in per_mol], dtype=float)
    mat_p = np.array([r.mat_p for r in per_mol], dtype=float)

    return {
        "COV-R_mean": float(np.nanmean(cov_r)),
        "MAT-R_mean": float(np.nanmean(mat_r)),
        "COV-P_mean": float(np.nanmean(cov_p)),
        "MAT-P_mean": float(np.nanmean(mat_p)),
        "COV-R_median": float(np.nanmedian(cov_r)),
        "MAT-R_median": float(np.nanmedian(mat_r)),
        "COV-P_median": float(np.nanmedian(cov_p)),
        "MAT-P_median": float(np.nanmedian(mat_p)),
    }
