import numpy as np
import pandas as pd
import sys
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from sklearn.metrics import (
    precision_recall_curve,
    roc_auc_score,
    accuracy_score,
)

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is optional
    tqdm = None

# precision recall warning.
np.seterr(divide="ignore", invalid="ignore")


def _progress(iterable, **kwargs):
    if tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs)


def _chunk_sequence(items, chunk_size):
    for start_idx in range(0, len(items), chunk_size):
        yield start_idx // chunk_size, items[start_idx : start_idx + chunk_size]


def _resolve_mp_context():
    preferred_method = "fork" if sys.platform.startswith("linux") else "spawn"
    return mp.get_context(preferred_method)





def remove_diagonal(T):
    # Takes in 3 dim tensor and removes diagonal of 2/3 dim.
    out = []
    for x in T:
        out.append(x[~np.eye(x.shape[0], dtype=bool)].reshape(x.shape[0], -1))
    return np.stack(out)


def max_accuracy(labs, preds):
    # ACCURACY MAX
    preds = preds.astype(float)
    if preds.min() == preds.max():
        a = []
    else:
        a = list(
            np.arange(
                preds.min(),
                preds.max(),
                (preds.max() - preds.min()) / 100,
            )
        )  # 100 steps
    possible_thresholds = [0] + a + [preds.max() + 1e-6]
    acc = [accuracy_score(labs, preds > thresh) for thresh in possible_thresholds]
    acc_thresh = possible_thresholds[np.argmax(acc)]
    acc_score = np.nanmax(acc)
    return acc_thresh, acc_score


def f1_max(labs, preds):
    # F1 MAX
    precision, recall, thresholds = precision_recall_curve(labs, preds)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1_scores = 2 * recall * precision / (recall + precision)
    f1_thresh = thresholds[np.argmax(f1_scores)]
    f1_score = np.nanmax(f1_scores)
    return f1_thresh, f1_score


def _score_single_sample(sample_pair):
    labs_flat, preds_flat = sample_pair
    if np.unique(labs_flat).size == 1:
        return None

    auroc = roc_auc_score(y_true=labs_flat, y_score=preds_flat)
    f1_thresh, f1_score = f1_max(labs_flat, preds_flat)
    acc_thresh, acc_score = max_accuracy(labs_flat, preds_flat)
    return {
        "auroc": auroc,
        "f1_thresh": f1_thresh,
        "f1_score": f1_score,
        "acc_thresh": acc_thresh,
        "acc_score": acc_score,
    }


def _score_chunk(sample_pairs):
    return [_score_single_sample(sample_pair) for sample_pair in sample_pairs]


def _collect_individual_scores(sample_pairs, n_jobs=1, chunk_size=512):
    if n_jobs <= 1 or len(sample_pairs) <= 1:
        results = []
        for sample_pair in _progress(sample_pairs, total=len(sample_pairs), desc="Scoring samples"):
            results.append(_score_single_sample(sample_pair))
        return results

    chunks = list(_chunk_sequence(sample_pairs, max(1, chunk_size)))
    max_workers = min(n_jobs, len(chunks))
    chunk_results = {}

    def collect_results(executor):
        future_to_chunk = {
            executor.submit(_score_chunk, chunk): chunk_index
            for chunk_index, chunk in chunks
        }
        progress_bar = tqdm(total=len(sample_pairs), desc="Scoring samples") if tqdm else None
        for future in as_completed(future_to_chunk):
            chunk_index = future_to_chunk[future]
            chunk_output = future.result()
            chunk_results[chunk_index] = chunk_output
            if progress_bar is not None:
                progress_bar.update(len(chunk_output))
        if progress_bar is not None:
            progress_bar.close()

    try:
        with ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=_resolve_mp_context(),
        ) as executor:
            collect_results(executor)
    except (PermissionError, OSError) as exc:
        print(f"Process scorers unavailable ({exc}); falling back to thread scorers")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            collect_results(executor)

    ordered_results = []
    for chunk_index in sorted(chunk_results):
        ordered_results.extend(chunk_results[chunk_index])
    return ordered_results


def _score_from_prepared(sample_pairs, labs_flat, preds_flat, name="Result", n_jobs=1, chunk_size=512):
    print("Scoring...")
    sample_results = _collect_individual_scores(
        sample_pairs,
        n_jobs=n_jobs,
        chunk_size=chunk_size,
    )
    valid_sample_results = [result for result in sample_results if result is not None]

    if valid_sample_results:
        f1_max_ind = np.mean([result["f1_score"] for result in valid_sample_results])
        f1_thresh_ind = np.mean([result["f1_thresh"] for result in valid_sample_results])
        accuracy_ind = np.mean([result["acc_score"] for result in valid_sample_results])
        accuracy_ind_thresh = np.mean([result["acc_thresh"] for result in valid_sample_results])
        auroc_ind = np.mean([result["auroc"] for result in valid_sample_results])
    else:
        f1_max_ind = np.nan
        f1_thresh_ind = np.nan
        accuracy_ind = np.nan
        accuracy_ind_thresh = np.nan
        auroc_ind = np.nan

    # AUROC
    auroc = roc_auc_score(labs_flat, preds_flat)
    # F1 MAX

    f1_thresh, f1_score = f1_max(labs_flat, preds_flat)
    # ACCURACY MAX
    acc_thresh, acc_score = max_accuracy(labs_flat, preds_flat)

    null_model_auroc = roc_auc_score(labs_flat, np.zeros(preds_flat.shape))

    _, null_model_f1 = f1_max(labs_flat, np.zeros(preds_flat.shape))

    _, null_model_acc = max_accuracy(labs_flat, np.zeros(preds_flat.shape))

    out = pd.DataFrame(
        [
            acc_thresh,
            acc_score,
            accuracy_ind_thresh,
            accuracy_ind,
            null_model_acc,
            f1_thresh,
            f1_score,
            f1_thresh_ind,
            f1_max_ind,
            null_model_f1,
            auroc,
            auroc_ind,
            null_model_auroc,
        ],
        # columns=[cfg.method.name + "_" + cfg.label_path.split("/")[-2]],
        columns=[name],
        index=[
            "Max Acc thresh",
            "Max Acc",
            "Max individual Acc thresh",
            "Max individual Acc",
            "Null Acc",
            "Max F1 thresh",
            "Max F1",
            "Max individual F1 thresh",
            "Max individual F1",
            "Null F1",
            "AUROC",
            "Individual AUROC",
            "Null AUROC",
        ],
    )
    out.index.name = "Metric"
    return out


def score_preprocessed(preds, labs, name="Result", n_jobs=1, chunk_size=512):
    """
    Calculates metrics assuming ``preds`` and ``labs`` are already aligned and
    preprocessed to the final scoring shape.
    """
    preds = np.asarray(preds)
    labs = np.asarray(labs)

    if preds.ndim == 2:
        preds = np.expand_dims(preds, 0)
        labs = np.expand_dims(labs, 0)

    sample_pairs = [
        (labs[x].flatten(), preds[x].flatten())
        for x in range(len(labs))
    ]
    return _score_from_prepared(
        sample_pairs,
        labs.flatten(),
        preds.flatten(),
        name=name,
        n_jobs=n_jobs,
        chunk_size=chunk_size,
    )


def score(preds, labs, remove_autoregressive=True, name="Result", n_jobs=1, chunk_size=512):
    """
    Calculates a number of metrics given preds and labs.
    Takes in either a 2dim or a 3dim tensor (batch of summary graphs)
    name is used for later column naming.
    """

    if isinstance(preds, list):
        preds = np.array(preds)
    if isinstance(labs, list):
        labs = np.array(labs)
    if isinstance(preds, pd.DataFrame):
        preds = preds.values
    if isinstance(labs, pd.DataFrame):
        labs = labs.values
    if preds.ndim == 2:
        preds = np.expand_dims(preds, 0)
        labs = np.expand_dims(labs, 0)

    if remove_autoregressive:
        labs = remove_diagonal(labs)
        preds = remove_diagonal(preds)
    else:
        labs = np.array(labs)
        preds = np.array(preds)

    return score_preprocessed(
        preds,
        labs,
        name=name,
        n_jobs=n_jobs,
        chunk_size=chunk_size,
    )
