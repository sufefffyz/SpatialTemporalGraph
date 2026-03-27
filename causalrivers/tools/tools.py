import numpy as np
import pandas as pd
import pickle
import os
import datetime
from pathlib import Path
from omegaconf import OmegaConf

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is optional
    tqdm = None


def _progress(iterable, **kwargs):
    if tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs)


def remove_trailing_nans(sample_prep):
    """
    Removes samples that were not removed by interpolate.
    """
    check_trailing_nans = np.where(sample_prep.isnull().values.any(axis=1) == 0)[0]
    if not len(check_trailing_nans) == 0:  # A ts is completely 0:
        sample_prep = sample_prep[
            check_trailing_nans.min() : check_trailing_nans.max() + 1
        ]
        
    if len(sample_prep) == 0: 
        # random case that everything is empty. This can happen when selecting a window.
        print("EMPTY SAMPLE DETECTED")
    return sample_prep

# Subsample only all subgraphs that contain only saxony and thuringia nodes:
def filter_samples_based_on_properties(ds, G, selection=["T", "S"], prop="origin"):
    sub_ds = []
    for d in ds:
        origin_check = set([G.nodes[x][prop] for x in d.nodes])
        if np.all([x in selection for x in origin_check]):
            sub_ds.append(d)
    return sub_ds


def graph_to_label_tensor(G_sample, human_readable=False):
    nodes = sorted(G_sample.nodes)
    labels = np.zeros((len(nodes), len(nodes)))

    for n, x in enumerate(nodes):
        for m, y in enumerate(nodes):
            if (x, y) in G_sample.edges:
                labels[m, n] = 1
    if human_readable:
        labels = pd.DataFrame(labels, columns=nodes, index=nodes)
        labels = pd.concat(
            [pd.concat([labels], keys=["Cause"], axis=1)], keys=["Effect"]
        )
        return labels
    else:
        return labels


def _build_datetime_index(unix_timestamps, index_col="datetime"):
    dt_index = pd.to_datetime(np.asarray(unix_timestamps), unit="s")
    if index_col:
        dt_index.name = index_col
    return dt_index


class _InMemoryTimeseriesSource:
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame

    def get_frame(self, node_ids):
        return self.frame[[str(x) for x in node_ids]].copy()


class _DirectoryTimeseriesSource:
    def __init__(self, data_path, index_col="datetime"):
        data_path = Path(data_path)
        targets_path = data_path / "targets.npy"
        timestamps_path = data_path / "unix_timestamps.npy"
        node_ids_path = data_path / "node_ids.npy"

        if not targets_path.exists() or not timestamps_path.exists():
            raise FileNotFoundError(
                f"{data_path} must contain targets.npy and unix_timestamps.npy to be used as a time series source."
            )

        self.targets = np.load(targets_path, mmap_mode="r")
        self.index = _build_datetime_index(
            np.load(timestamps_path, mmap_mode="r"),
            index_col=index_col,
        )
        if node_ids_path.exists():
            available_ids = np.load(node_ids_path, allow_pickle=True)
            self.id_to_position = {
                int(node_id): idx for idx, node_id in enumerate(available_ids.tolist())
            }
        else:
            self.id_to_position = None

    def get_frame(self, node_ids):
        if self.id_to_position is None:
            column_positions = [int(x) for x in node_ids]
        else:
            column_positions = [self.id_to_position[int(x)] for x in node_ids]

        return pd.DataFrame(
            np.asarray(self.targets[:, column_positions]),
            columns=[str(x) for x in node_ids],
            index=self.index,
        )


class _NpzTimeseriesSource:
    def __init__(self, data_path, index_col="datetime"):
        self.data = np.load(data_path, allow_pickle=True)
        if "targets" not in self.data or "unix_timestamps" not in self.data:
            raise KeyError(
                f"{data_path} must contain 'targets' and 'unix_timestamps' to be used as a time series source."
            )
        self.targets = self.data["targets"]
        self.index = _build_datetime_index(self.data["unix_timestamps"], index_col=index_col)

    def get_frame(self, node_ids):
        column_ids = [int(x) for x in node_ids]
        return pd.DataFrame(
            self.targets[:, column_ids],
            columns=[str(x) for x in column_ids],
            index=self.index,
        )


def _load_timeseries_from_csv(data_path, node_ids, index_col="datetime"):
    usecols = ([index_col] if index_col else []) + [str(x) for x in node_ids]
    return pd.read_csv(
        data_path,
        index_col=index_col if index_col else None,
        usecols=usecols,
    )


def _load_timeseries_from_npz(data_path, node_ids, index_col="datetime"):
    """
    Convenience loader for npz datasets.
    Note: np.load on compressed npz materializes the full targets array in memory.
    For large traffic datasets prefer the directory format produced by
    prepare_urban_traffic_for_causalrivers.py, which stores targets as mmap-able .npy.
    """
    data = np.load(data_path, allow_pickle=True)
    if "targets" not in data or "unix_timestamps" not in data:
        raise KeyError(
            f"{data_path} must contain 'targets' and 'unix_timestamps' to be used as a time series source."
        )

    column_ids = [int(x) for x in node_ids]
    frame = pd.DataFrame(
        data["targets"][:, column_ids],
        columns=[str(x) for x in column_ids],
        index=_build_datetime_index(data["unix_timestamps"], index_col=index_col),
    )
    return frame


def _load_timeseries_from_directory(data_path, node_ids, index_col="datetime"):
    return _DirectoryTimeseriesSource(data_path, index_col=index_col).get_frame(node_ids)


def load_timeseries_source(data_path, index_col="datetime"):
    source_path = Path(data_path)

    if source_path.is_dir():
        return _DirectoryTimeseriesSource(source_path, index_col=index_col)

    match source_path.suffix.lower():
        case ".csv":
            frame = pd.read_csv(
                source_path,
                index_col=index_col if index_col else None,
            )
            return _InMemoryTimeseriesSource(frame)
        case ".npz":
            return _NpzTimeseriesSource(source_path, index_col=index_col)
        case _:
            raise ValueError(
                f"Unsupported data_path format: {data_path}. Expected CSV, NPZ, or a directory "
                "containing targets.npy and unix_timestamps.npy."
            )


def load_timeseries_table(data_path, node_ids, index_col="datetime"):
    return load_timeseries_source(data_path, index_col=index_col).get_frame(node_ids)


def load_sample(which, p="resources/rivers_ts_east_germany.csv"):
    return load_timeseries_table(p, list(which.nodes), index_col="datetime")



def preprocess_data(
    data,
    resolution="2H",
    interpolate=True,
    subset_year=False,
    subset_month=False,
    subsample=1,
    normalize=False,
    remove_trailing_nans_early=False,
    fill_remaining_nans_with_zero=False,
):

    sample_data = data.copy()  # dont change the original data

    # Remove trailing nans (so start and end of the ts to improve data quality.
    # WARNING: This can make the TS arbitrarily short).
    if remove_trailing_nans_early:
        sample_data = remove_trailing_nans(sample_data)
        
    # Adjust resolution
    sample_data["dt"] = pd.to_datetime(sample_data.index).round(resolution).values

    sample_data = sample_data.groupby("dt").mean()
    # subsampling
    if subset_year:
        sample_data = sample_data.loc[
            (sample_data.index.month.isin(subset_month))
            & (sample_data.index.year == subset_year)
        ]
    sample_data = sample_data.iloc[::subsample, :]
    if normalize:
        sample_data = (sample_data - sample_data.min()) / (
            sample_data.max() - sample_data.min()
        )
    if interpolate:
        sample_data = sample_data.interpolate()
    sample_data = sample_data.replace([np.inf, -np.inf], np.nan)
    if fill_remaining_nans_with_zero:
        sample_data = sample_data.fillna(0.0)
    return sample_data


def standard_preprocessing(
    data: pd.DataFrame,
    cfg,
):
    """
    simple wrapper arround the standard preprocessing class that uses a hydra config only.
    """
    sample_data = preprocess_data(
        data,
        resolution=cfg.resolution,
        interpolate=cfg.interpolate,
        subset_year=cfg.subset_year,
        subset_month=cfg.subset_month,
        subsample=cfg.subsample,
        normalize=cfg.normalize,
        remove_trailing_nans_early=cfg.remove_trailing_nans_early,
        fill_remaining_nans_with_zero=getattr(
            cfg,
            "fill_remaining_nans_with_zero",
            False,
        ),
    )
    return sample_data


def benchmarking(X, cfg, method_to_test):
    """
    Takes in the output of the data loader and perform the predictions with a specified method.
    If anything else should happen with the data beforehand this should happen here.
    """
    preds = []
    progress_bar = _progress(
        enumerate(X, start=1),
        total=len(X),
        desc=f"{cfg.method.name.upper()} samples",
    )
    for x, sample in progress_bar:
        preds.append(method_to_test(sample, cfg.method))
    return preds


def load_label_graphs(cfg):
    data = pickle.load(open(cfg.label_path, "rb"))
    if cfg.restrict_to >= 0:
        data = data[cfg.restrict_to : cfg.restrict_to + 1]
    return data


def prepare_single_sample(
    sample_graph,
    source,
    cfg,
    preprocessing=None,
    human_readable_labels=False,
):
    sample_nodes = sorted(sample_graph.nodes)
    single_sample = source.get_frame(sample_nodes)
    if preprocessing:
        single_sample = preprocessing(single_sample, cfg.data_preprocess)
    single_sample = remove_trailing_nans(single_sample)
    labels = graph_to_label_tensor(sample_graph, human_readable=human_readable_labels)
    return single_sample, labels


def iter_joint_samples(cfg, index_col="datetime", preprocessing=None, human_readable_labels=False):
    sample_graphs = load_label_graphs(cfg)
    source = load_timeseries_source(cfg.data_path, index_col=index_col)
    sample_iterator = _progress(
        sample_graphs,
        total=len(sample_graphs),
        desc="Streaming samples",
    )

    for sample_graph in sample_iterator:
        yield prepare_single_sample(
            sample_graph,
            source,
            cfg,
            preprocessing=preprocessing,
            human_readable_labels=human_readable_labels,
        )


def load_joint_samples(cfg, index_col="datetime", preprocessing=None):
    """
    Loads and transforms the data.
    If you have additional preprocessing you can provide a function.
    Importantly, if you struggle with ram it migt be worth to load the samples individually as in 2_tutorial_benchmarking.
    This is however slower.
    """

    X = []
    Y = []
    for single_sample, labels in iter_joint_samples(
        cfg,
        index_col=index_col,
        preprocessing=preprocessing,
        human_readable_labels=True,
    ):
        X.append(single_sample)
        Y.append(labels)
    return X, Y


def save_run(out,stop_time, preds, cfg):
    # make folder with naming
    p = cfg.save_path + cfg.method.name + "_" + cfg.label_path.split("/")[-2]
    if not os.path.exists(p):
        os.makedirs(p)
    inner_p = p + "/" + str(datetime.datetime.now())[:24]
    os.makedirs(inner_p)
    out.to_csv(inner_p + "/scoring.csv")

    pd.DataFrame([stop_time], columns=["runtime"]).to_csv(
        inner_p + "/runtime.csv"
    )  # dumps to file:
    with open(inner_p + "/config.yaml", "w") as f:
        OmegaConf.save(cfg, f)
    pickle.dump(preds, open(inner_p + "/preds.p", "wb"))


def save_multi_run(out, runtimes, stop_time, preds_by_method, cfg, method_names):
    label_group = Path(cfg.label_path).parent.name
    p = Path(cfg.save_path) / ("multi_" + label_group)
    p.mkdir(parents=True, exist_ok=True)
    inner_p = p / str(datetime.datetime.now())[:24]
    inner_p.mkdir()

    out.to_csv(inner_p / "scoring.csv")
    runtime_rows = []
    for method_name in method_names:
        runtime_rows.append(
            {
                "name": method_name,
                "runtime": str(runtimes[method_name]),
            }
        )
    runtime_rows.append({"name": "overall", "runtime": str(stop_time)})
    pd.DataFrame(runtime_rows).to_csv(inner_p / "runtime.csv", index=False)

    with open(inner_p / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    preds_dir = inner_p / "preds"
    preds_dir.mkdir()
    for method_name in method_names:
        pickle.dump(preds_by_method[method_name], open(preds_dir / f"{method_name}.p", "wb"))
