import argparse
import json
import os
import pickle
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


KNOWAIR_METEO_VAR = [
    "100m_u_component_of_wind",
    "100m_v_component_of_wind",
    "2m_dewpoint_temperature",
    "2m_temperature",
    "boundary_layer_height",
    "k_index",
    "relative_humidity+950",
    "relative_humidity+975",
    "specific_humidity+950",
    "surface_pressure",
    "temperature+925",
    "temperature+950",
    "total_precipitation",
    "u_component_of_wind+950",
    "v_component_of_wind+950",
    "vertical_velocity+950",
    "vorticity+950",
]
KNOWAIR_PM25GNN_METEO_USE = [
    "2m_temperature",
    "boundary_layer_height",
    "k_index",
    "relative_humidity+950",
    "surface_pressure",
    "total_precipitation",
    "u_component_of_wind+950",
    "v_component_of_wind+950",
]


def haversine_distance_m(lon_lat: np.ndarray) -> np.ndarray:
    lon = np.deg2rad(lon_lat[:, 0].astype("float64"))
    lat = np.deg2rad(lon_lat[:, 1].astype("float64"))
    dlon = lon[None, :] - lon[:, None]
    dlat = lat[None, :] - lat[:, None]
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    dist = 6_371_000.0 * c
    np.fill_diagonal(dist, 0.0)
    return dist.astype("float32")


def add_time_features(target: np.ndarray, steps_per_day: int) -> np.ndarray:
    length, num_nodes = target.shape
    target = target.astype("float32")[..., None]
    tod = (np.arange(length, dtype="float32") % steps_per_day) / float(steps_per_day)
    tod = np.tile(tod[:, None, None], (1, num_nodes, 1))
    dow = ((np.arange(length, dtype="int64") // steps_per_day) % 7).astype("float32") / 7.0
    dow = np.tile(dow[:, None, None], (1, num_nodes, 1))
    return np.concatenate([target, tod, dow], axis=-1).astype("float32")


def broadcast_time_features(length: int, num_nodes: int, steps_per_day: int) -> np.ndarray:
    tod = (np.arange(length, dtype="float32") % steps_per_day) / float(steps_per_day)
    tod = np.tile(tod[:, None, None], (1, num_nodes, 1))
    dow = ((np.arange(length, dtype="int64") // steps_per_day) % 7).astype("float32") / 7.0
    dow = np.tile(dow[:, None, None], (1, num_nodes, 1))
    return np.concatenate([tod, dow], axis=-1).astype("float32")


def add_full_knowair_features(raw: np.ndarray) -> tuple[np.ndarray, list[str]]:
    pm25 = raw[:, :, -1:].astype("float32")
    covariates = raw[:, :, :-1].astype("float32")
    time_features = broadcast_time_features(raw.shape[0], raw.shape[1], steps_per_day=8)
    feature_description = (
        ["PM2.5 target"]
        + [f"KnowAir raw covariate {idx}" for idx in range(covariates.shape[-1])]
        + ["time of day", "day of week"]
    )
    return np.concatenate([pm25, covariates, time_features], axis=-1).astype("float32"), feature_description


def add_mage_knowair_features(raw: np.ndarray) -> tuple[np.ndarray, list[str]]:
    pm25 = raw[:, :, -1:].astype("float32")
    covariate_names = KNOWAIR_PM25GNN_METEO_USE
    covariate_indices = [KNOWAIR_METEO_VAR.index(name) for name in covariate_names]
    meteo = raw[:, :, covariate_indices].astype("float32")
    time_features = broadcast_time_features(raw.shape[0], raw.shape[1], steps_per_day=8)

    u950 = raw[:, :, KNOWAIR_METEO_VAR.index("u_component_of_wind+950")].astype("float32")
    v950 = raw[:, :, KNOWAIR_METEO_VAR.index("v_component_of_wind+950")].astype("float32")
    wind_speed = np.sqrt(np.square(u950) + np.square(v950))[..., None].astype("float32")
    wind_direction = ((np.arctan2(v950, u950) + np.pi) / (2.0 * np.pi))[..., None].astype("float32")

    feature_description = (
        ["PM2.5 target"]
        + covariate_names
        + ["time of day", "day of week", "wind speed from u/v+950", "wind direction from u/v+950"]
    )
    data = np.concatenate([pm25, meteo, time_features, wind_speed, wind_direction], axis=-1)
    if data.shape[-1] != 13:
        raise ValueError(f"KnowAir MAGE-dim feature count must be 13, got {data.shape[-1]}.")
    return data.astype("float32"), feature_description


def extend_ccaq_time_features(u: np.ndarray, input_len: int, horizon: int) -> np.ndarray:
    """Build a per-timestep month/week/hour sequence from GAGNN's per-window u."""
    num_samples = u.shape[0]
    length = num_samples + input_len + horizon - 1
    time = np.empty((length, 3), dtype="float32")
    time[input_len:input_len + num_samples] = u.astype("float32")

    first = u[0].astype("int64").copy()
    for pos in range(input_len - 1, -1, -1):
        time[pos] = first
        first[2] -= 1
        if first[2] < 0:
            first[2] = 23
            first[1] = (first[1] - 1) % 7

    last = u[-1].astype("int64").copy()
    for pos in range(input_len + num_samples, length):
        last[2] += 1
        if last[2] >= 24:
            last[2] = 0
            last[1] = (last[1] + 1) % 7
        time[pos] = last

    scale = np.array([12.0, 7.0, 24.0], dtype="float32")
    return time / scale


def write_basic_ts_dataset(
    output_dir: Path,
    dataset_name: str,
    data: np.ndarray,
    distance_m: np.ndarray,
    input_len: int,
    output_len: int,
    train_val_test_ratio: list[float],
    frequency_minutes: int,
    domain: str,
    target_description: str,
    feature_description: list[str] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fp = np.memmap(output_dir / "data.dat", dtype="float32", mode="w+", shape=data.shape)
    fp[:] = data[:]
    fp.flush()
    del fp

    np.save(output_dir / "distance_m.npy", distance_m.astype("float32"))
    with (output_dir / "adj_mx.pkl").open("wb") as f:
        pickle.dump([None, None, np.zeros((data.shape[1], data.shape[1]), dtype="float32")], f)

    desc = {
        "name": dataset_name,
        "domain": domain,
        "shape": list(data.shape),
        "num_time_steps": int(data.shape[0]),
        "num_nodes": int(data.shape[1]),
        "num_features": int(data.shape[2]),
        "feature_description": feature_description or [target_description, "time of day", "day of week"],
        "has_graph": True,
        "frequency (minutes)": frequency_minutes,
        "regular_settings": {
            "INPUT_LEN": input_len,
            "OUTPUT_LEN": output_len,
            "TRAIN_VAL_TEST_RATIO": train_val_test_ratio,
            "NORM_EACH_CHANNEL": False,
            "RESCALE": True,
            "METRICS": ["MAE", "RMSE", "MAPE", "WAPE"],
            "NULL_VAL": np.nan,
        },
    }
    with (output_dir / "desc.json").open("w", encoding="utf-8") as f:
        json.dump(desc, f, indent=4, allow_nan=True)
    print(f"Saved {dataset_name}: data={data.shape}, distance={distance_m.shape}, output={output_dir}")


def prepare_knowair(root: Path, output_root: Path, feature_mode: str) -> None:
    if feature_mode == "target_time":
        dataset_name = "KnowAir"
    elif feature_mode == "full_covariate":
        dataset_name = "KnowAir_FULLCOV"
    else:
        dataset_name = "KnowAir_MAGE13"
    raw_root = root / "raw_data" / "KnowAir_official"
    raw = np.load(raw_root / "KnowAir.npy")
    pm25 = raw[:, :, -1].astype("float32")
    cities = pd.read_csv(raw_root / "city.txt", sep=r"\s+", header=None, names=["idx", "city", "lon", "lat"])
    if len(cities) != pm25.shape[1]:
        raise ValueError(f"KnowAir city count {len(cities)} does not match data nodes {pm25.shape[1]}.")
    coords = cities[["lon", "lat"]].to_numpy(dtype="float64")
    if feature_mode == "target_time":
        data = add_time_features(pm25, steps_per_day=8)
        feature_description = ["PM2.5", "time of day", "day of week"]
    elif feature_mode == "full_covariate":
        data, feature_description = add_full_knowair_features(raw)
    else:
        data, feature_description = add_mage_knowair_features(raw)
    write_basic_ts_dataset(
        output_dir=output_root / dataset_name,
        dataset_name=dataset_name,
        data=data,
        distance_m=haversine_distance_m(coords),
        input_len=24,
        output_len=24,
        train_val_test_ratio=[0.6, 0.2, 0.2],
        frequency_minutes=180,
        domain="air quality",
        target_description="PM2.5",
        feature_description=feature_description,
    )
    cities.to_csv(output_root / dataset_name / "city_info.csv", index=False)


def reconstruct_target_sequence(x: np.ndarray, y: np.ndarray, target_channel: int) -> np.ndarray:
    num_samples, input_len, num_nodes, _ = x.shape
    horizon = y.shape[1]
    target = np.empty((num_samples + input_len + horizon - 1, num_nodes), dtype="float32")
    target[:input_len] = x[0, :, :, target_channel]
    target[input_len:input_len + num_samples] = y[:, 0, :]
    target[input_len + num_samples:] = y[-1, 1:, :]
    return target


def load_ccaq_split(zip_path: Path, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(f"{split}_x.npy") as f:
            x = np.load(f)
        with zf.open(f"{split}_y.npy") as f:
            y = np.load(f)
        with zf.open(f"{split}_u.npy") as f:
            u = np.load(f)
    return x, y, u


def reconstruct_ccaq_full_features(x: np.ndarray, y: np.ndarray, u: np.ndarray, target_channel: int) -> np.ndarray:
    num_samples, input_len, num_nodes, num_x_features = x.shape
    horizon = y.shape[1]
    length = num_samples + input_len + horizon - 1

    x_sequence = np.empty((length, num_nodes, num_x_features), dtype="float32")
    x_sequence[:input_len] = x[0].astype("float32")
    if num_samples > 1:
        x_sequence[input_len:input_len + num_samples - 1] = x[1:, -1].astype("float32")
    x_sequence[input_len + num_samples - 1:] = x[-1, -1].astype("float32")

    target = reconstruct_target_sequence(x, y, target_channel=target_channel)
    x_sequence[..., target_channel] = target
    non_target = np.concatenate([x_sequence[..., :target_channel], x_sequence[..., target_channel + 1:]], axis=-1)

    time = extend_ccaq_time_features(u, input_len=input_len, horizon=horizon)
    time = np.tile(time[:, None, :], (1, num_nodes, 1)).astype("float32")
    return np.concatenate([target[..., None], non_target, time], axis=-1).astype("float32")


def reconstruct_ccaq_mage_features(x: np.ndarray, y: np.ndarray, u: np.ndarray, target_channel: int) -> np.ndarray:
    full = reconstruct_ccaq_full_features(x, y, u, target_channel=target_channel)
    # MAGE uses input_dim=10 for CCAQ and feature_dim=input_dim-2. Keep 8 node signals
    # (target + 7 non-target x channels), then use week/hour as DOW/TOD-like time features.
    return np.concatenate([full[..., :8], full[..., 9:11]], axis=-1).astype("float32")


def prepare_ccaq(root: Path, output_root: Path, feature_mode: str) -> None:
    if feature_mode == "target_time":
        dataset_name = "CCAQ"
    elif feature_mode == "full_covariate":
        dataset_name = "CCAQ_FULLCOV"
    else:
        dataset_name = "CCAQ_MAGE10"
    raw_root = root / "raw_data" / "CCAQ_GAGNN_official"
    zip_path = raw_root / "city_air_dataset.zip"
    split_sequences = []
    split_lengths = []
    for split in ("train", "val", "test"):
        x, y, u = load_ccaq_split(zip_path, split)
        if feature_mode == "target_time":
            seq = reconstruct_target_sequence(x, y, target_channel=7)
        elif feature_mode == "full_covariate":
            seq = reconstruct_ccaq_full_features(x, y, u, target_channel=7)
        else:
            seq = reconstruct_ccaq_mage_features(x, y, u, target_channel=7)
        split_sequences.append(seq)
        split_lengths.append(int(seq.shape[0]))
        if split == "train":
            check = np.nanmean(np.abs(y[:100, 0, :] - x[1:101, -1, :, 7]))
            if check > 1e-6:
                raise ValueError(f"CCAQ sliding-window target check failed: mean abs diff={check}")
        del x, y

    stacked = np.concatenate(split_sequences, axis=0)
    total_len = int(stacked.shape[0])
    ratios = [split_lengths[0] / total_len, split_lengths[1] / total_len, split_lengths[2] / total_len]
    if feature_mode == "target_time":
        data = add_time_features(stacked, steps_per_day=24)
        feature_description = ["AQI", "time of day", "day of week"]
    elif feature_mode == "full_covariate":
        data = stacked.astype("float32")
        feature_description = (
            ["AQI target"]
            + [f"CCAQ x covariate {idx}" for idx in range(7)]
            + ["month index", "week index", "hour index"]
        )
    else:
        data = stacked.astype("float32")
        feature_description = (
            ["AQI target"]
            + [f"CCAQ x covariate {idx}" for idx in range(7)]
            + ["week index", "hour index"]
        )
        if data.shape[-1] != 10:
            raise ValueError(f"CCAQ MAGE-dim feature count must be 10, got {data.shape[-1]}.")
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open("loc_filled.npy") as f:
            coords = np.load(f).astype("float64")
    write_basic_ts_dataset(
        output_dir=output_root / dataset_name,
        dataset_name=dataset_name,
        data=data,
        distance_m=haversine_distance_m(coords),
        input_len=24,
        output_len=24 if feature_mode == "mage_dim" else 6,
        train_val_test_ratio=ratios,
        frequency_minutes=60,
        domain="air quality",
        target_description="AQI",
        feature_description=feature_description,
    )
    pd.DataFrame(coords, columns=["lon", "lat"]).to_csv(output_root / dataset_name / "station_info.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["KnowAir", "CCAQ", "all"], default="all")
    parser.add_argument("--root", default="datasets")
    parser.add_argument("--output-root", default="datasets")
    parser.add_argument("--feature-mode", choices=["target_time", "full_covariate", "mage_dim"], default="target_time")
    args = parser.parse_args()

    root = Path(args.root)
    output_root = Path(args.output_root)
    if args.dataset in {"KnowAir", "all"}:
        prepare_knowair(root, output_root, feature_mode=args.feature_mode)
    if args.dataset in {"CCAQ", "all"}:
        prepare_ccaq(root, output_root, feature_mode=args.feature_mode)


if __name__ == "__main__":
    main()
