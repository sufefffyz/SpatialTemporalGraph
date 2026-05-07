from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def f1_from_counts(tp: float, fp: float, fn: float) -> float:
    denom = 2 * tp + fp + fn
    if denom == 0:
        return float("nan")
    return float(2 * tp / denom)


def average_precision_from_hist(pos_hist: np.ndarray, neg_hist: np.ndarray) -> float:
    total_pos = float(np.sum(pos_hist))
    if total_pos == 0:
        return float("nan")
    tp = 0.0
    fp = 0.0
    prev_recall = 0.0
    ap = 0.0
    for p, n in zip(pos_hist[::-1], neg_hist[::-1]):
        tp += float(p)
        fp += float(n)
        if tp == 0:
            continue
        recall = tp / total_pos
        precision = tp / (tp + fp)
        ap += (recall - prev_recall) * precision
        prev_recall = recall
    return float(ap)


def update_binary_hist(
    scores: np.ndarray,
    labels: np.ndarray,
    bin_edges: np.ndarray,
    pos_hist: np.ndarray,
    neg_hist: np.ndarray,
) -> None:
    valid = np.isfinite(scores) & np.isfinite(labels)
    if not np.any(valid):
        return
    scores = np.asarray(scores[valid], dtype=np.float32)
    labels = np.asarray(labels[valid], dtype=bool)
    idx = np.searchsorted(bin_edges, scores, side="right") - 1
    idx = np.clip(idx, 0, len(pos_hist) - 1)
    pos_hist += np.bincount(idx[labels], minlength=len(pos_hist))
    neg_hist += np.bincount(idx[~labels], minlength=len(neg_hist))


def road_positive_quantile_thresholds(
    train_targets: np.ndarray,
    q: float = 0.90,
    eps: float = 1e-5,
    min_positive_support: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    train_targets = np.asarray(train_targets, dtype=np.float32)
    thresholds = np.full(train_targets.shape[1], np.nan, dtype=np.float32)
    support = np.zeros(train_targets.shape[1], dtype=np.int64)
    for node_idx in range(train_targets.shape[1]):
        values = train_targets[:, node_idx]
        values = values[np.isfinite(values) & (values > eps)]
        support[node_idx] = len(values)
        if len(values) >= min_positive_support:
            thresholds[node_idx] = float(np.quantile(values, q))
    return thresholds, support


@dataclass
class MetricConfig:
    eps: float = 1e-5
    high_q: float = 0.90
    min_positive_support: int = 24
    hist_bins: int = 512


class ZeroAwareMetricAccumulator:
    def __init__(self, train_targets: np.ndarray, config: MetricConfig | None = None):
        self.config = config or MetricConfig()
        train_targets = np.asarray(train_targets, dtype=np.float32)
        positive_train = train_targets[np.isfinite(train_targets) & (train_targets > self.config.eps)]
        score_cap = float(np.quantile(positive_train, 0.999)) if positive_train.size else 1.0
        self.occurrence_edges = np.linspace(0.0, max(score_cap, 1.0), self.config.hist_bins + 1)
        self.high_edges = np.linspace(0.0, 3.0, self.config.hist_bins + 1)
        self.high_thresholds, self.high_support = road_positive_quantile_thresholds(
            train_targets,
            q=self.config.high_q,
            eps=self.config.eps,
            min_positive_support=self.config.min_positive_support,
        )
        self.reset()

    def reset(self) -> None:
        self.n = 0.0
        self.abs_sum = 0.0
        self.sq_sum = 0.0
        self.target_abs_sum = 0.0

        self.pos_n = 0.0
        self.pos_abs_sum = 0.0
        self.pos_sq_sum = 0.0
        self.pos_target_abs_sum = 0.0
        self.pos_log_abs_sum = 0.0

        self.occ_tp = 0.0
        self.occ_fp = 0.0
        self.occ_fn = 0.0
        self.occ_tn = 0.0
        self.occ_pos_hist = np.zeros(self.config.hist_bins, dtype=np.float64)
        self.occ_neg_hist = np.zeros(self.config.hist_bins, dtype=np.float64)

        self.high_tp = 0.0
        self.high_fp = 0.0
        self.high_fn = 0.0
        self.high_tn = 0.0
        self.high_valid_n = 0.0
        self.high_pos_hist = np.zeros(self.config.hist_bins, dtype=np.float64)
        self.high_neg_hist = np.zeros(self.config.hist_bins, dtype=np.float64)

    def update(self, prediction: np.ndarray, target: np.ndarray) -> None:
        pred = np.asarray(prediction, dtype=np.float32)
        tgt = np.asarray(target, dtype=np.float32)
        if pred.ndim == 4:
            pred = pred[..., 0]
        if tgt.ndim == 4:
            tgt = tgt[..., 0]

        valid = np.isfinite(pred) & np.isfinite(tgt)
        if not np.any(valid):
            return

        err = pred - tgt
        self.n += float(np.sum(valid))
        self.abs_sum += float(np.sum(np.abs(err[valid])))
        self.sq_sum += float(np.sum(err[valid] ** 2))
        self.target_abs_sum += float(np.sum(np.abs(tgt[valid])))

        pos = valid & (tgt > self.config.eps)
        if np.any(pos):
            self.pos_n += float(np.sum(pos))
            self.pos_abs_sum += float(np.sum(np.abs(err[pos])))
            self.pos_sq_sum += float(np.sum(err[pos] ** 2))
            self.pos_target_abs_sum += float(np.sum(np.abs(tgt[pos])))
            self.pos_log_abs_sum += float(
                np.sum(np.abs(np.log1p(np.maximum(pred[pos], 0.0)) - np.log1p(tgt[pos])))
            )

        occ_label = valid & (tgt > self.config.eps)
        occ_pred = valid & (pred > self.config.eps)
        self.occ_tp += float(np.sum(occ_label & occ_pred))
        self.occ_fp += float(np.sum(~occ_label & occ_pred & valid))
        self.occ_fn += float(np.sum(occ_label & ~occ_pred))
        self.occ_tn += float(np.sum(~occ_label & ~occ_pred & valid))
        update_binary_hist(
            np.maximum(pred[valid], 0.0),
            occ_label[valid],
            self.occurrence_edges,
            self.occ_pos_hist,
            self.occ_neg_hist,
        )

        thresholds = self.high_thresholds.reshape(1, 1, -1)
        high_valid = valid & np.isfinite(thresholds)
        if np.any(high_valid):
            high_label = high_valid & (tgt > thresholds)
            high_pred = high_valid & (pred > thresholds)
            self.high_tp += float(np.sum(high_label & high_pred))
            self.high_fp += float(np.sum(~high_label & high_pred & high_valid))
            self.high_fn += float(np.sum(high_label & ~high_pred))
            self.high_tn += float(np.sum(~high_label & ~high_pred & high_valid))
            self.high_valid_n += float(np.sum(high_valid))
            normalized_score = np.divide(
                np.maximum(pred, 0.0),
                thresholds,
                out=np.zeros_like(pred, dtype=np.float32),
                where=np.isfinite(thresholds) & (thresholds > 0),
            )
            update_binary_hist(
                normalized_score[high_valid],
                high_label[high_valid],
                self.high_edges,
                self.high_pos_hist,
                self.high_neg_hist,
            )

    def summary(self) -> dict[str, float]:
        occ_precision = safe_div(self.occ_tp, self.occ_tp + self.occ_fp)
        occ_recall = safe_div(self.occ_tp, self.occ_tp + self.occ_fn)
        high_precision = safe_div(self.high_tp, self.high_tp + self.high_fp)
        high_recall = safe_div(self.high_tp, self.high_tp + self.high_fn)
        return {
            "MAE": safe_div(self.abs_sum, self.n),
            "RMSE": float(np.sqrt(safe_div(self.sq_sum, self.n))) if self.n else float("nan"),
            "WAPE": safe_div(self.abs_sum, self.target_abs_sum),
            "MAE_pos": safe_div(self.pos_abs_sum, self.pos_n),
            "RMSE_pos": float(np.sqrt(safe_div(self.pos_sq_sum, self.pos_n))) if self.pos_n else float("nan"),
            "WAPE_pos": safe_div(self.pos_abs_sum, self.pos_target_abs_sum),
            "log1p_MAE_pos": safe_div(self.pos_log_abs_sum, self.pos_n),
            "occurrence_precision": occ_precision,
            "occurrence_recall": occ_recall,
            "occurrence_f1": f1_from_counts(self.occ_tp, self.occ_fp, self.occ_fn),
            "occurrence_AP_approx": average_precision_from_hist(self.occ_pos_hist, self.occ_neg_hist),
            "occurrence_positive_rate": safe_div(self.occ_tp + self.occ_fn, self.n),
            "high_q": self.config.high_q,
            "high_valid_road_ratio": float(np.mean(np.isfinite(self.high_thresholds))),
            "high_precision": high_precision,
            "high_recall": high_recall,
            "high_f1": f1_from_counts(self.high_tp, self.high_fp, self.high_fn),
            "high_AP_approx": average_precision_from_hist(self.high_pos_hist, self.high_neg_hist),
            "high_positive_rate": safe_div(self.high_tp + self.high_fn, self.high_valid_n),
        }
