"""Stage 3: per-layer linear probes, their metrics, and the reference baselines.

For each encoder layer:
    regression probe:      standardise features, ridge regression on the (transformed) label
    classification probe:  standardise features, 4-bin logistic regression
Regularisation strength is chosen by grouped K-fold cross-validation on the TRAIN set only
(folds never split a room or source), then the probe is refit on all of train and scored on test.

Output, under data/probes/<property>/<condition>/<probe_hash>/:
    results.json            metrics per layer + chance + hand-crafted baseline
    predictions.npz         per-layer predictions for every clip (train ones are in-sample)
    models.pkl              the fitted (regressor, classifier) per layer
    plots/scatter_layer_XX.png   predicted vs true, and residuals
"""

from __future__ import annotations

import json
import pickle

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from audio_probes.config import probe_dir, save_resolved_config
from audio_probes.data import load_clip, load_meta
from audio_probes.publish import publish
from audio_probes.split import get_split


# ---- labels -------------------------------------------------------------------------

def transform_label(y: np.ndarray, kind: str) -> np.ndarray:
    """linear: y.  log: ln(y). Probes are fit (and R^2 measured) on this scale."""
    return np.log(y) if kind == "log" else np.asarray(y, dtype=float)


def inverse_label(z: np.ndarray, kind: str) -> np.ndarray:
    return np.exp(z) if kind == "log" else z


def to_bins(y: np.ndarray, edges: list[float]) -> np.ndarray:
    """Bin index 0..len(edges): bin b holds edges[b-1] <= y < edges[b]."""
    return np.digitize(y, edges)


# ---- metrics ------------------------------------------------------------------------

def r2_score(z_true: np.ndarray, z_pred: np.ndarray) -> float:
    """R^2 = 1 - sum (z - z_hat)^2 / sum (z - mean(z))^2.  1 = perfect, 0 = no better than
    predicting the mean, negative = worse than the mean."""
    ss_res = np.sum((z_true - z_pred) ** 2)
    ss_tot = np.sum((z_true - z_true.mean()) ** 2)
    return float(1 - ss_res / ss_tot)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rho: Pearson correlation of the ranks. 1 = same ordering, 0 = unrelated."""
    if np.std(b) == 0:
        return 0.0
    return float(spearmanr(a, b).statistic)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error in label units: mean |y - y_hat|."""
    return float(np.mean(np.abs(y_true - y_pred)))


def regression_metrics(y_true: np.ndarray, z_pred: np.ndarray, kind: str) -> dict:
    z_true = transform_label(y_true, kind)
    return {"r2": r2_score(z_true, z_pred), "spearman": spearman(y_true, z_pred),
            "mae": mae(y_true, inverse_label(z_pred, kind))}


def accuracy(b_true: np.ndarray, b_pred: np.ndarray) -> float:
    return float(np.mean(b_true == b_pred))


# ---- model fitting ------------------------------------------------------------------

def make_regressor(probe_cfg: dict, alpha: float):
    """Standardise each feature to mean 0 / std 1, then ridge:
    w = argmin ||z - Xw||^2 + alpha ||w||^2  =  (X^T X + alpha I)^-1 X^T z."""
    if probe_cfg["type"] == "mlp":
        return make_pipeline(StandardScaler(), MLPRegressor((probe_cfg["mlp_hidden"],), alpha=probe_cfg["mlp_alpha"],
                                                            early_stopping=True, max_iter=500, random_state=0))
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def make_classifier(probe_cfg: dict, C: float):
    """Standardise, then multinomial logistic regression:
    p(bin=k | x) = softmax(W x + b)_k, fit by minimising cross-entropy + ||W||^2 / (2C)."""
    if probe_cfg["type"] == "mlp":
        return make_pipeline(StandardScaler(), MLPClassifier((probe_cfg["mlp_hidden"],), alpha=probe_cfg["mlp_alpha"],
                                                             early_stopping=True, max_iter=500, random_state=0))
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))


def grouped_cv_pick(make_model, grid, X, y, groups, folds, score_fn) -> float:
    """Return the grid value with the best mean validation score over grouped K-fold CV."""
    if len(grid) == 1:
        return grid[0]
    cv = GroupKFold(n_splits=min(folds, len(np.unique(groups))))
    means = []
    for value in grid:
        scores = []
        for tr, va in cv.split(X, y, groups):
            model = make_model(value).fit(X[tr], y[tr])
            scores.append(score_fn(y[va], model.predict(X[va])))
        means.append(np.mean(scores))
    return grid[int(np.argmax(means))]


def fit_probe_pair(X: np.ndarray, y: np.ndarray, train: np.ndarray, groups: np.ndarray, cfg: dict) -> dict:
    """Fit the regression and classification probes on train rows; predict every row."""
    pc = cfg["probe"]
    kind, edges = pc["label_transform"], pc["bin_edges"]
    z, bins = transform_label(y, kind), to_bins(y, edges)
    Xtr, gtr = X[train], groups[train]

    alpha_grid = [pc["alpha"]] if pc["alpha"] is not None else pc["alphas"]
    alpha = grouped_cv_pick(lambda a: make_regressor(pc, a), alpha_grid, Xtr, z[train], gtr, pc["cv_folds"], r2_score)
    reg = make_regressor(pc, alpha).fit(Xtr, z[train])

    C_grid = [pc["logistic_C"]] if pc["logistic_C"] is not None else pc["logistic_Cs"]
    C = grouped_cv_pick(lambda c: make_classifier(pc, c), C_grid, Xtr, bins[train], gtr, pc["cv_folds"], accuracy)
    clf = make_classifier(pc, C).fit(Xtr, bins[train])
    return {"z_pred": reg.predict(X), "bin_pred": clf.predict(X), "alpha": alpha, "C": C, "models": (reg, clf)}


def score(y, fit, test, cfg) -> dict:
    kind, edges = cfg["probe"]["label_transform"], cfg["probe"]["bin_edges"]
    out = regression_metrics(y[test], fit["z_pred"][test], kind)
    out["acc4"] = accuracy(to_bins(y[test], edges), fit["bin_pred"][test])
    out["alpha"], out["C"] = fit["alpha"], fit["C"]
    return out


def chance_scores(y, train, test, cfg) -> dict:
    """Predict the train mean for regression and the most common train bin for classification."""
    kind, edges = cfg["probe"]["label_transform"], cfg["probe"]["bin_edges"]
    z_mean = transform_label(y[train], kind).mean()
    out = regression_metrics(y[test], np.full(len(test), z_mean), kind)
    majority = np.bincount(to_bins(y[train], edges), minlength=len(edges) + 1).argmax()
    out["acc4"] = accuracy(to_bins(y[test], edges), np.full(len(test), majority))
    return out


# ---- stage --------------------------------------------------------------------------

def run(cfg: dict, condition: str) -> None:
    from audio_probes.embed import load_embeddings

    out = probe_dir(cfg, condition)
    if (out / "DONE").exists():
        print(f"[probe] {cfg['property']}/{condition}: cached at {out}")
        publish(cfg, condition, out / "plots", "probes")
        return
    layers, ids = load_embeddings(cfg, condition)
    records = load_meta(cfg)
    assert ids == [r["id"] for r in records], "embedding rows do not match clip metadata"
    split = get_split(cfg, records)
    train, test, groups = np.array(split["train"]), np.array(split["test"]), np.array(split["groups"])
    y = np.array([r["label"] for r in records], dtype=float)
    layer_ids = list(range(len(layers))) if cfg["encoder"]["layers"] == "all" else list(cfg["encoder"]["layers"])

    per_layer, z_preds, bin_preds, models = {}, [], [], {}
    for layer_id, X in zip(layer_ids, layers):
        fit = fit_probe_pair(X, y, train, groups, cfg)
        per_layer[f"layer_{layer_id:02d}"] = score(y, fit, test, cfg)
        z_preds.append(fit["z_pred"])
        bin_preds.append(fit["bin_pred"])
        models[int(layer_id)] = fit["models"]
        m = per_layer[f"layer_{layer_id:02d}"]
        print(f"[probe] {cfg['property']}/{condition} layer {layer_id:2d}: R2={m['r2']:.3f} "
              f"rho={m['spearman']:.3f} MAE={m['mae']:.3g} acc4={m['acc4']:.3f} (alpha={m['alpha']:g}, C={m['C']:g})")

    baseline = handcrafted_baseline(cfg, records, condition, y, train, test, groups)
    results = {
        "property": cfg["property"], "condition": condition, "label_transform": cfg["probe"]["label_transform"],
        "bin_edges": cfg["probe"]["bin_edges"], "n_train": int(len(train)), "n_test": int(len(test)),
        "n_groups": split["n_components"], "layers": per_layer,
        "chance": chance_scores(y, train, test, cfg), "baseline": baseline,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    is_test = np.zeros(len(y), bool)
    is_test[test] = True
    np.savez(out / "predictions.npz", ids=np.array(ids), y=y, z_pred=np.stack(z_preds),
             bin_pred=np.stack(bin_preds), is_test=is_test, layer_ids=np.array(layer_ids))
    with open(out / "models.pkl", "wb") as f:  # fitted probes, reused by `predict`
        pickle.dump(models, f)
    write_scatter_plots(cfg, y, np.stack(z_preds), test, layer_ids, per_layer, out / "plots")
    save_resolved_config({**cfg, "condition": condition}, out / "config.yaml")
    (out / "DONE").write_text("ok\n")
    publish(cfg, condition, out / "plots", "probes")


def handcrafted_baseline(cfg, records, condition, y, train, test, groups) -> dict:
    """Fit the same probes on a few hand-made features (e.g. RMS level, direct-to-reverb ratio)."""
    from audio_probes.properties import get_property

    prop = get_property(cfg)
    X = np.stack([prop.baseline_features(load_clip(cfg, r, condition)) for r in records])
    fit = fit_probe_pair(X, y, train, groups, cfg)
    out = score(y, fit, test, cfg)
    out["features"] = prop.baseline_description()
    return out


def write_scatter_plots(cfg, y, z_preds, test, layer_ids, per_layer, plot_dir) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    kind = cfg["probe"]["label_transform"]
    z_true = transform_label(y[test], kind)
    axis_name = f"{'log ' if kind == 'log' else ''}{cfg['property']}"
    for layer_id, z_pred in zip(layer_ids, z_preds):
        zp = z_pred[test]
        m = per_layer[f"layer_{layer_id:02d}"]
        fig, (a, b) = plt.subplots(1, 2, figsize=(9, 4))
        a.scatter(z_true, zp, s=6, alpha=0.6)
        lo, hi = z_true.min(), z_true.max()
        a.plot([lo, hi], [lo, hi], "k--", lw=1)
        a.set(xlabel=f"true {axis_name}", ylabel=f"predicted {axis_name}",
              title=f"layer {layer_id}: R2={m['r2']:.3f}, rho={m['spearman']:.3f}")
        b.scatter(z_true, zp - z_true, s=6, alpha=0.6)
        b.axhline(0, color="k", lw=1)
        b.set(xlabel=f"true {axis_name}", ylabel="residual (pred - true)", title="residuals (test clips)")
        fig.tight_layout()
        fig.savefig(plot_dir / f"scatter_layer_{layer_id:02d}.png", dpi=100)
        plt.close(fig)
