"""Version A: amplify the EEG-predictive directions of frozen word vectors.

1. Standardize word vectors with training-word statistics.
2. Ridge-regress word EEG targets on the vectors (training words only); the
   column space of the coefficients is the "EEG subspace" U.
3. Shape: x' = x + beta * U U' x (beta = 0 is the plain text baseline).
4. Sentence feature = mean of shaped word vectors (equal to shaping the mean,
   since the map is linear); logistic regression for sentiment.
beta and C are chosen on validation sentences only.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import f1_score, r2_score

from .data import grouped_splits

ALPHAS = np.logspace(-1, 7, 17)
BETAS = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)
CS = (0.01, 0.1, 1.0, 10.0)


def standardizer(X):
    mean, std = X.mean(axis=0), X.std(axis=0)
    std[std < 1e-8] = 1.0
    return mean, std


def fit_subspace(Xs_train, Y_train, alphas=ALPHAS, groups=None):
    """Ridge from word vectors to EEG targets; U spans the coefficient columns.

    With ``groups`` (sentence id per row) the penalty is chosen by
    cross-validation over whole sentences. Without, RidgeCV uses
    leave-one-word-out, where words of the same sentence (shared EEG offset,
    similar contextual vectors) inform each other and too little
    regularization wins.
    """
    cv = None if groups is None else grouped_splits(np.asarray(groups), 5, seed=0)
    ridge = RidgeCV(alphas=alphas, cv=cv).fit(Xs_train, Y_train)
    coef = np.atleast_2d(ridge.coef_).T  # [dim, k]
    U, _ = np.linalg.qr(coef)
    return U, ridge


def encoding_r2(ridge, Xs, Y):
    """How well the word vectors predict word EEG (variance-weighted R^2)."""
    return float(r2_score(Y, ridge.predict(Xs), multioutput="variance_weighted"))


def sentence_features(word_vectors, mean, std):
    return np.stack([((v - mean) / std).mean(axis=0) for v in word_vectors])


def shape(S, U, beta):
    if U is None or beta == 0:
        return S
    return S + beta * (S @ U) @ U.T


def select_and_predict(S_train, y_train, S_val, y_val, S_test, U, betas=BETAS, cs=CS):
    """Pick (beta, C) by validation macro-F1; return test probabilities and the choice."""
    best = None
    grid = []
    for beta in (betas if U is not None else (0.0,)):
        train, val = shape(S_train, U, beta), shape(S_val, U, beta)
        for c in cs:
            model = LogisticRegression(C=c, max_iter=5000, class_weight="balanced").fit(train, y_train)
            score = f1_score(y_val, model.predict(val), average="macro")
            grid.append({"beta": beta, "C": c, "val_macro_f1": float(score)})
            if best is None or score > best[0] + 1e-9:
                best = (score, beta, c, model)
    _, beta, c, model = best
    probs = model.predict_proba(shape(S_test, U, beta))
    full = np.zeros((len(S_test), 3))
    full[:, model.classes_] = probs
    return full, {"beta": beta, "C": c, "val_macro_f1": float(best[0]), "grid": grid, "model": model}


def save_model(path, U, mean, std, choice):
    """Everything needed to apply a fitted arm to new word vectors."""
    model = choice["model"]
    np.savez(path, U=np.zeros((len(mean), 0)) if U is None else U, mean=mean, std=std,
             beta=choice["beta"], C=choice["C"], coef=model.coef_, intercept=model.intercept_,
             classes=model.classes_)


def alignment_with_sentiment(U, S, y):
    """Share of the sentiment classifier's weight norm that lies in the EEG subspace."""
    if U is None:
        return None
    model = LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced").fit(S, y)
    W = model.coef_.T
    inside = np.linalg.norm(U.T @ W) ** 2
    return float(inside / max(np.linalg.norm(W) ** 2, 1e-12)), float(U.shape[1] / U.shape[0])
