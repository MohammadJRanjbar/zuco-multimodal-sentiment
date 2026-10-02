"""Do English and Persian reading EEG pick out the same directions of a multilingual text space?

For each language, a ridge map from (within-sentence centered) word vectors to
the top-k EEG components defines an "EEG subspace" (the column space of the
coefficients). Two subspaces are compared by their overlap, the mean squared
cosine of their principal angles (0 = orthogonal, 1 = identical). A null comes
from subspaces fitted to EEG targets shuffled across words. The comparison is
repeated after projecting out each language's word-feature directions
(length, frequency, position), to see whether any shared part goes beyond them.
"""

import numpy as np
import torch

from ..brainshaping.data import grouped_splits
from ..brainshaping.encoding import ALPHAS, ridge_predictor


def orth(matrix):
    q, r = np.linalg.qr(matrix)
    keep = np.abs(np.diag(r)) > 1e-10 * max(np.abs(np.diag(r)).max(), 1e-300)
    return q[:, keep]


def overlap(U, V):
    """Mean squared cosine of the principal angles between span(U) and span(V)."""
    U, V = orth(U), orth(V)
    k = min(U.shape[1], V.shape[1])
    return float((np.linalg.norm(U.T @ V) ** 2) / k) if k else float("nan")


def remove_subspace(U, P):
    """Orthonormal basis of U with the span of P projected out."""
    P = orth(P)
    return orth(U - P @ (P.T @ U))


def ridge_coefficients(X, Y, groups, n_inner=4, alphas=ALPHAS, device="cpu", seed=0):
    """Coefficients [features, targets] of a ridge map on standardized X, penalty by sentence-grouped CV."""
    t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float64, device=device)  # noqa: E731
    Xt, Yt, alphas_t = t(X), t(Y - Y.mean(axis=0)), t(alphas)
    sse = torch.zeros(len(alphas), dtype=torch.float64, device=device)
    for a, b in grouped_splits(groups, n_inner, seed):
        sse += ((ridge_predictor(Xt[a], Xt[b])(Yt[a], alphas_t) - Yt[b][None]) ** 2).sum(dim=(1, 2))
    alpha = float(alphas[int(sse.argmin())])
    Xc = Xt - Xt.mean(dim=0)
    gram = Xc.T @ Xc + alpha * torch.eye(Xc.shape[1], dtype=torch.float64, device=device)
    return torch.linalg.solve(gram, Xc.T @ Yt).cpu().numpy(), alpha
