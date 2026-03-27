import pandas as pd
import numpy as np
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
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    corr = np.corrcoef(x, y)[0, 1]
    if np.isnan(corr):
        return 0.0
    return float(corr)


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


def var_baseline(d,cfg, human_readable=False):
    """
    Simple Granger based strategy that selects based on absolute parameter values.
    """
    n_vars = d.values.shape[-1]
    pred = np.zeros((n_vars, n_vars, cfg.max_lag))
    d_fit, valid_positions = _prepare_var_frame(d, cfg)
    if d_fit is None:
        _maybe_log_var_issue(cfg, "VAR skipped sample due to invalid/degenerate columns")
        out = summary_transform(pred, cfg.map_to_summary_graph)
        if human_readable:
            out = make_human_readable(out, d)
        return out
    
    # For sime constant ts this sometimes fails so we predict 0 if no model can be estimated.
    try:
        # fit var with appropriate max lags
        res = VAR(d_fit).fit(cfg.max_lag)
        # convert to bool and throw away intersection
        # !In the context of rivers, negative correlation do not really make sense.
        # I guess trying both is fair
        pred_fit = res.params[1:]
        
        if cfg.var_absolute_values:
            pred_fit = np.abs(pred_fit)

        # reformat to original caused causing lag:
        # :) einsum needed i guess
        pred_fit = np.stack(
            [
                pred_fit.values[:, x].reshape(cfg.max_lag, len(valid_positions)).T
                for x in range(pred_fit.shape[1])
            ]
        )
        for local_effect, global_effect in enumerate(valid_positions):
            for local_cause, global_cause in enumerate(valid_positions):
                pred[global_effect, global_cause, :] = pred_fit[local_effect, local_cause, :]
    except:
        _maybe_log_var_issue(cfg, "VAR fitting failed after sanitization")
    out = summary_transform(pred, cfg.map_to_summary_graph)
    

    if human_readable:
        out = make_human_readable(out, d)
    return out


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


BASELINE_METHODS = {
    "var": var_baseline,
    "corr": corr_baseline,
    "lagcorr": lagcorr_baseline,
}


def get_baseline_method(name):
    if name not in BASELINE_METHODS:
        available = ", ".join(sorted(BASELINE_METHODS))
        raise ValueError(f"Unknown baseline '{name}'. Available baselines: {available}")
    return BASELINE_METHODS[name]
