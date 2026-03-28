import re

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR


def make_human_readable(out, d):
    out = pd.DataFrame(out, columns=d.columns, index=d.columns)
    out = pd.concat([pd.concat([out], keys=["Cause"], axis=1)], keys=["Effect"])
    return out


def summary_transform(pred, opt):
    if opt == "max":
        prediction = pred.max(axis=2)
    elif opt == "mean":
        prediction = pred.mean(axis=2)
    return prediction


def _safe_corr(x, y):
    if len(x) == 0 or len(y) == 0:
        return 0.0
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    corr = np.corrcoef(x, y)[0, 1]
    if np.isnan(corr):
        return 0.0
    return float(corr)


def _prepare_dense_frame(d: pd.DataFrame):
    prepared = d.copy().replace([np.inf, -np.inf], np.nan)
    prepared = prepared.astype(float)
    prepared = prepared.interpolate(limit_direction="both")
    prepared = prepared.ffill().bfill()
    prepared = prepared.fillna(0.0)
    return prepared


def _finalize_pairwise_matrix(pred, d, human_readable=False):
    pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(pred, 0.0)
    if human_readable:
        return make_human_readable(pred, d)
    return pred


def _finalize_lagged_matrix(pred, d, cfg, human_readable=False):
    pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    for lag_idx in range(pred.shape[2]):
        np.fill_diagonal(pred[:, :, lag_idx], 0.0)
    out = summary_transform(pred, getattr(cfg, "map_to_summary_graph", "max"))
    if human_readable:
        return make_human_readable(out, d)
    return out


def _maybe_log_var_issue(cfg, message):
    if getattr(cfg, "var_verbose_failures", False):
        print(message)


def _select_full_rank_columns(frame: pd.DataFrame, min_std: float):
    selected_columns = []
    for column in frame.columns:
        candidate_columns = selected_columns + [column]
        candidate_values = frame[candidate_columns].to_numpy(dtype=float)
        candidate_values = candidate_values - candidate_values.mean(axis=0, keepdims=True)
        if np.linalg.matrix_rank(candidate_values) > len(selected_columns):
            selected_columns.append(column)

    selected_frame = frame[selected_columns]
    if len(selected_columns) < 2:
        return selected_frame

    std = selected_frame.std(axis=0)
    return selected_frame.loc[:, std > min_std]


def _prepare_var_frame(d: pd.DataFrame, cfg):
    min_std = float(getattr(cfg, "var_min_std", 1e-12))
    prepared = d.copy().replace([np.inf, -np.inf], np.nan)
    prepared = prepared.astype(float)
    prepared = prepared.interpolate(limit_direction="both")
    prepared = prepared.ffill().bfill()
    prepared = prepared.dropna(axis=1, how="any")

    if prepared.shape[1] < 2:
        return None, []

    std = prepared.std(axis=0)
    prepared = prepared.loc[:, std > min_std]
    if prepared.shape[1] < 2:
        return None, []

    prepared = _select_full_rank_columns(prepared, min_std=min_std)
    if prepared.shape[1] < 2:
        return None, []

    if len(prepared) <= int(cfg.max_lag) + 1:
        return None, []

    inferred_freq = prepared.index.inferred_freq
    if inferred_freq is not None:
        prepared.index = pd.DatetimeIndex(prepared.index.values, freq=inferred_freq)
    else:
        prepared.index = pd.DatetimeIndex(prepared.index.values)

    valid_positions = [d.columns.get_loc(column) for column in prepared.columns]
    return prepared, valid_positions


def _directional_cross_correlation(values, max_lag, absolute_values=True):
    n_vars = values.shape[1]
    pred = np.zeros((n_vars, n_vars, max_lag))

    for effect_idx in range(n_vars):
        effect_series = values[:, effect_idx]
        for cause_idx in range(n_vars):
            if cause_idx == effect_idx:
                continue
            cause_series = values[:, cause_idx]
            for lag in range(1, max_lag + 1):
                corr = _safe_corr(cause_series[:-lag], effect_series[lag:])
                pred[effect_idx, cause_idx, lag - 1] = abs(corr) if absolute_values else max(corr, 0.0)
    return pred


def _pairwise_winner_matrix(score_matrix):
    pairwise = np.zeros_like(score_matrix)
    n_vars = score_matrix.shape[0]
    for cause_idx in range(n_vars):
        for effect_idx in range(cause_idx + 1, n_vars):
            cause_to_effect = score_matrix[effect_idx, cause_idx]
            effect_to_cause = score_matrix[cause_idx, effect_idx]
            if cause_to_effect > effect_to_cause:
                pairwise[effect_idx, cause_idx] = cause_to_effect
            elif effect_to_cause > cause_to_effect:
                pairwise[cause_idx, effect_idx] = effect_to_cause
    return pairwise


def _apply_parent_selection(pred, means, corr_support, selection):
    selection = str(selection).lower()
    if selection in {"all", "none"}:
        return pred

    filtered = np.zeros_like(pred)
    n_vars = pred.shape[0]
    for cause_idx in range(n_vars):
        children = [effect_idx for effect_idx in np.where(pred[:, cause_idx] > 0)[0] if effect_idx != cause_idx]
        if not children:
            continue

        if selection == "next":
            larger_children = [effect_idx for effect_idx in children if means[effect_idx] > means[cause_idx]]
            if larger_children:
                winner = min(larger_children, key=lambda effect_idx: (means[effect_idx] - means[cause_idx], -pred[effect_idx, cause_idx]))
            else:
                winner = max(children, key=lambda effect_idx: pred[effect_idx, cause_idx])
        elif selection == "biggest":
            winner = max(children, key=lambda effect_idx: (means[effect_idx], pred[effect_idx, cause_idx]))
        elif selection in {"corr", "crosscorr", "c"}:
            winner = max(children, key=lambda effect_idx: (corr_support[effect_idx, cause_idx], pred[effect_idx, cause_idx]))
        else:
            raise ValueError(
                f"Unknown naive child selection '{selection}'. "
                "Available options: all, next, biggest, corr"
            )

        filtered[winner, cause_idx] = pred[winner, cause_idx]
    return filtered


def _reverse_physical_scores(means, score_mode="difference"):
    n_vars = len(means)
    pred = np.zeros((n_vars, n_vars))
    for cause_idx in range(n_vars):
        for effect_idx in range(n_vars):
            if cause_idx == effect_idx:
                continue
            if means[effect_idx] > means[cause_idx]:
                if score_mode == "difference":
                    score = means[effect_idx] - means[cause_idx]
                elif score_mode == "ratio":
                    denom = max(abs(means[cause_idx]), 1e-12)
                    score = means[effect_idx] / denom
                else:
                    raise ValueError(
                        f"Unknown reverse physical score mode '{score_mode}'. "
                        "Available options: difference, ratio"
                    )
                pred[effect_idx, cause_idx] = score
    return pred


def _parse_dynotears_node(node_name, column_names):
    for column_name in sorted(column_names, key=len, reverse=True):
        match = re.fullmatch(rf"{re.escape(str(column_name))}_lag(\d+)", str(node_name))
        if match:
            return str(column_name), int(match.group(1))
    return None, None


def _cc_pairwise_pred(frame, cfg):
    max_lag = int(cfg.max_lag)
    lagged_scores = _directional_cross_correlation(
        frame.to_numpy(dtype=float),
        max_lag=max_lag,
        absolute_values=getattr(cfg, "cc_absolute_values", True),
    )
    support = summary_transform(lagged_scores, getattr(cfg, "map_to_summary_graph", "max"))
    pairwise = _pairwise_winner_matrix(support)
    return pairwise, support


def _rp_pairwise_pred(frame, cfg):
    means = frame.mean(axis=0).to_numpy(dtype=float)
    pairwise = _reverse_physical_scores(
        means,
        score_mode=getattr(cfg, "rp_score_mode", "difference"),
    )
    return pairwise, means


def _combo_pairwise_pred(cc_pred, rp_pred, combo_mode="max"):
    if combo_mode == "max":
        return np.maximum(cc_pred, rp_pred)
    if combo_mode == "sum":
        return cc_pred + rp_pred
    raise ValueError(
        f"Unknown combo union mode '{combo_mode}'. Available options: max, sum"
    )


def var_baseline(d, cfg, human_readable=False):
    """
    Simple Granger based strategy that selects based on absolute parameter values.
    """
    n_vars = d.values.shape[-1]
    pred = np.zeros((n_vars, n_vars, cfg.max_lag))
    d_fit, valid_positions = _prepare_var_frame(d, cfg)
    if d_fit is None:
        _maybe_log_var_issue(cfg, "VAR skipped sample due to invalid/degenerate columns")
        return _finalize_lagged_matrix(pred, d, cfg, human_readable=human_readable)

    try:
        res = VAR(d_fit).fit(cfg.max_lag)
        pred_fit = res.params[1:]

        if cfg.var_absolute_values:
            pred_fit = np.abs(pred_fit)

        pred_fit = np.stack(
            [
                pred_fit.values[:, x].reshape(cfg.max_lag, len(valid_positions)).T
                for x in range(pred_fit.shape[1])
            ]
        )
        for local_effect, global_effect in enumerate(valid_positions):
            for local_cause, global_cause in enumerate(valid_positions):
                pred[global_effect, global_cause, :] = pred_fit[local_effect, local_cause, :]
    except Exception:
        _maybe_log_var_issue(cfg, "VAR fitting failed after sanitization")

    return _finalize_lagged_matrix(pred, d, cfg, human_readable=human_readable)


def corr_baseline(d, cfg, human_readable=False):
    values = d.values.astype(float)
    pred = np.corrcoef(values, rowvar=False)
    pred = np.nan_to_num(pred, nan=0.0)

    if getattr(cfg, "corr_absolute_values", True):
        pred = np.abs(pred)

    if human_readable:
        pred = make_human_readable(pred, d)
    return pred


def lagcorr_baseline(d, cfg, human_readable=False):
    values = d.values.astype(float)
    n_vars = values.shape[1]
    max_lag = int(cfg.max_lag)
    pred = np.zeros((n_vars, n_vars, max_lag))
    for effect_idx in range(n_vars):
        effect_series = values[:, effect_idx]
        for cause_idx in range(n_vars):
            cause_series = values[:, cause_idx]
            for lag in range(1, max_lag + 1):
                pred[effect_idx, cause_idx, lag - 1] = _safe_corr(
                    cause_series[:-lag],
                    effect_series[lag:],
                )

    if getattr(cfg, "lagcorr_absolute_values", True):
        pred = np.abs(pred)

    out = summary_transform(pred, cfg.map_to_summary_graph)
    if human_readable:
        out = make_human_readable(out, d)
    return out


def cc_baseline(d, cfg, human_readable=False):
    frame = _prepare_dense_frame(d)
    means = frame.mean(axis=0).to_numpy(dtype=float)
    pairwise, support = _cc_pairwise_pred(frame, cfg)
    pairwise = _apply_parent_selection(
        pairwise,
        means=means,
        corr_support=support,
        selection=getattr(cfg, "naive_child_selection", "all"),
    )
    return _finalize_pairwise_matrix(pairwise, d, human_readable=human_readable)


def rp_baseline(d, cfg, human_readable=False):
    frame = _prepare_dense_frame(d)
    pairwise, means = _rp_pairwise_pred(frame, cfg)
    _, support = _cc_pairwise_pred(frame, cfg)
    pairwise = _apply_parent_selection(
        pairwise,
        means=means,
        corr_support=support,
        selection=getattr(cfg, "naive_child_selection", "all"),
    )
    return _finalize_pairwise_matrix(pairwise, d, human_readable=human_readable)


def combo_baseline(d, cfg, human_readable=False):
    frame = _prepare_dense_frame(d)
    cc_pairwise, support = _cc_pairwise_pred(frame, cfg)
    rp_pairwise, means = _rp_pairwise_pred(frame, cfg)
    pairwise = _combo_pairwise_pred(
        cc_pairwise,
        rp_pairwise,
        combo_mode=getattr(cfg, "combo_union_mode", "max"),
    )
    pairwise = _apply_parent_selection(
        pairwise,
        means=means,
        corr_support=support,
        selection=getattr(cfg, "naive_child_selection", "all"),
    )
    return _finalize_pairwise_matrix(pairwise, d, human_readable=human_readable)


def pcmci_baseline(d, cfg, human_readable=False):
    try:
        from tigramite.data_processing import DataFrame as TigramiteDataFrame
        from tigramite.independence_tests.parcorr import ParCorr
        from tigramite.pcmci import PCMCI
    except ImportError as exc:
        raise ImportError(
            "PCMCI requires the optional dependency 'tigramite'. "
            "Install it manually, for example: pip install tigramite"
        ) from exc

    frame = _prepare_dense_frame(d)
    values = frame.to_numpy(dtype=float)
    max_lag = int(cfg.max_lag)
    pred = np.zeros((values.shape[1], values.shape[1], max_lag))

    if values.shape[1] < 2 or values.shape[0] <= max_lag + 1:
        return _finalize_lagged_matrix(pred, d, cfg, human_readable=human_readable)

    tigramite_frame = TigramiteDataFrame(values=values, var_names=list(frame.columns))
    pcmci = PCMCI(
        dataframe=tigramite_frame,
        cond_ind_test=ParCorr(significance="analytic"),
        verbosity=int(getattr(cfg, "pcmci_verbosity", 0)),
    )
    results = pcmci.run_pcmci(
        tau_min=1,
        tau_max=max_lag,
        pc_alpha=getattr(cfg, "pcmci_pc_alpha", 0.05),
    )

    val_matrix = np.asarray(results["val_matrix"], dtype=float)
    if getattr(cfg, "pcmci_absolute_values", True):
        val_matrix = np.abs(val_matrix)
    pred = np.transpose(val_matrix[:, :, 1 : max_lag + 1], (1, 0, 2))

    if getattr(cfg, "pcmci_filter_nonsignificant", True):
        alpha_level = float(getattr(cfg, "pcmci_alpha_level", 0.05))
        p_matrix = np.asarray(results["p_matrix"], dtype=float)
        significant = (p_matrix[:, :, 1 : max_lag + 1] <= alpha_level).astype(float)
        pred *= np.transpose(significant, (1, 0, 2))

    return _finalize_lagged_matrix(pred, d, cfg, human_readable=human_readable)


def varlingam_baseline(d, cfg, human_readable=False):
    try:
        from lingam import VARLiNGAM
    except ImportError as exc:
        raise ImportError(
            "VARLiNGAM requires the optional dependency 'lingam'. "
            "Install it manually, for example: pip install lingam"
        ) from exc

    frame = _prepare_dense_frame(d)
    values = frame.to_numpy(dtype=float)
    max_lag = int(cfg.max_lag)
    pred = np.zeros((values.shape[1], values.shape[1], max_lag))

    if values.shape[1] < 2 or values.shape[0] <= max_lag + 1:
        return _finalize_lagged_matrix(pred, d, cfg, human_readable=human_readable)

    model = VARLiNGAM(
        lags=max_lag,
        criterion=getattr(cfg, "varlingam_criterion", None),
        prune=bool(getattr(cfg, "varlingam_prune", True)),
        random_state=getattr(cfg, "varlingam_random_state", None),
    )
    model.fit(values)

    adjacency_matrices = np.asarray(model.adjacency_matrices_, dtype=float)
    if adjacency_matrices.ndim != 3:
        return _finalize_lagged_matrix(pred, d, cfg, human_readable=human_readable)

    if adjacency_matrices.shape[0] >= max_lag + 1:
        lagged = adjacency_matrices[1 : max_lag + 1]
    else:
        lagged = adjacency_matrices[:max_lag]

    if lagged.size == 0:
        return _finalize_lagged_matrix(pred, d, cfg, human_readable=human_readable)

    if getattr(cfg, "varlingam_absolute_values", True):
        lagged = np.abs(lagged)
    pred[:, :, : lagged.shape[0]] = np.transpose(lagged, (1, 2, 0))
    return _finalize_lagged_matrix(pred, d, cfg, human_readable=human_readable)


def dynotears_baseline(d, cfg, human_readable=False):
    try:
        from causalnex.structure.dynotears import from_pandas_dynamic
    except ImportError as exc:
        raise ImportError(
            "DynoTears requires the optional dependency 'causalnex' with dynamic structure support. "
            "Install it manually before running this baseline."
        ) from exc

    frame = _prepare_dense_frame(d)
    max_lag = int(cfg.max_lag)
    pred = np.zeros((frame.shape[1], frame.shape[1], max_lag))

    if frame.shape[1] < 2 or frame.shape[0] <= max_lag + 1:
        return _finalize_lagged_matrix(pred, d, cfg, human_readable=human_readable)

    structure_model = from_pandas_dynamic(
        frame,
        p=max_lag,
        lambda_w=float(getattr(cfg, "dynotears_lambda_w", 0.1)),
        lambda_a=float(getattr(cfg, "dynotears_lambda_a", 0.1)),
        max_iter=int(getattr(cfg, "dynotears_max_iter", 100)),
        h_tol=float(getattr(cfg, "dynotears_h_tol", 1e-8)),
        w_threshold=float(getattr(cfg, "dynotears_w_threshold", 0.0)),
    )

    name_to_index = {str(column_name): idx for idx, column_name in enumerate(frame.columns)}
    for source_node, target_node, weight in structure_model.edges.data("weight", default=0.0):
        source_name, source_lag = _parse_dynotears_node(source_node, frame.columns)
        target_name, target_lag = _parse_dynotears_node(target_node, frame.columns)
        if source_name is None or target_name is None:
            continue
        if target_lag != 0 or source_lag <= 0 or source_lag > max_lag:
            continue

        value = abs(weight) if getattr(cfg, "dynotears_absolute_values", True) else float(weight)
        pred[name_to_index[target_name], name_to_index[source_name], source_lag - 1] = value

    return _finalize_lagged_matrix(pred, d, cfg, human_readable=human_readable)


BASELINE_METHODS = {
    "var": var_baseline,
    "corr": corr_baseline,
    "lagcorr": lagcorr_baseline,
    "cc": cc_baseline,
    "rp": rp_baseline,
    "combo": combo_baseline,
    "pcmci": pcmci_baseline,
    "varlingam": varlingam_baseline,
    "dynotears": dynotears_baseline,
}


def get_baseline_method(name):
    if name not in BASELINE_METHODS:
        available = ", ".join(sorted(BASELINE_METHODS))
        raise ValueError(f"Unknown baseline '{name}'. Available baselines: {available}")
    return BASELINE_METHODS[name]
