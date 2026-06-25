"""FaST-style BasicTS helpers for LargeST LTSF baseline probes.

This module is intentionally kept under the experiment folder so the baseline
adaptations remain easy to remove. It mirrors the public FaST reproduction
setup where samples are split after sliding-window indexing and PatchSTG's
decoder predicts an arbitrary output length.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from timm.models.vision_transformer import Attention, Mlp

from basicts.data.base_dataset import BaseDataset
from basicts.scaler.base_scaler import BaseScaler


class FaSTIndexedTimeSeriesDataset(BaseDataset):
    """LargeST dataset with FaST-style sample-level sequential split."""

    def __init__(
        self,
        dataset_name: str,
        train_val_test_ratio: List[float],
        mode: str,
        input_len: int,
        output_len: int,
        overlap: bool = False,
        logger=None,
    ) -> None:
        assert mode in ["train", "valid", "test"]
        super().__init__(dataset_name, train_val_test_ratio, mode, False)
        self.input_len = input_len
        self.output_len = output_len
        self.overlap = overlap
        self.logger = logger
        self.dataset_dir = Path("datasets") / dataset_name
        self.description = self._load_description()
        self.raw_data = self._load_raw_data()
        self.data = self._slice_data()

    def _load_description(self) -> dict:
        with open(self.dataset_dir / "desc.json", "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_raw_data(self) -> np.ndarray:
        his_path = self.dataset_dir / "his.npz"
        if his_path.exists():
            return np.load(his_path)["data"].astype(np.float32, copy=False)
        data_path = self.dataset_dir / "data.dat"
        shape = tuple(self.description["shape"])
        return np.memmap(data_path, dtype="float32", mode="r", shape=shape)

    def _split_indices(self) -> np.ndarray:
        total_steps = self.input_len + self.output_len
        indices = np.arange(total_steps - 1, len(self.raw_data))
        train_end = int(self.train_val_test_ratio[0] * len(indices))
        val_end = train_end + int(self.train_val_test_ratio[1] * len(indices))
        if self.mode == "train":
            return indices[:train_end]
        if self.mode == "valid":
            return indices[train_end:val_end]
        return indices[val_end:]

    def _slice_data(self) -> np.ndarray:
        indices = self._split_indices()
        if len(indices) == 0:
            raise ValueError(
                f"No samples for {self.dataset_name} {self.mode} "
                f"L{self.input_len} H{self.output_len}."
            )
        start = int(indices[0]) - self.output_len - self.input_len + 1
        end = int(indices[-1]) + 1
        return np.asarray(self.raw_data[start:end], dtype=np.float32)

    def __getitem__(self, index: int) -> dict:
        history_data = self.data[index : index + self.input_len].astype(np.float32)
        future_data = self.data[index + self.input_len : index + self.input_len + self.output_len].astype(np.float32)
        return {"inputs": history_data, "target": future_data}

    def __len__(self) -> int:
        return len(self.data) - self.input_len - self.output_len + 1


class FaSTSampleFirstZScoreScaler(BaseScaler):
    """Normalize target channel using the FaST sample-split train prefix."""

    def __init__(
        self,
        dataset_name: str,
        train_ratio: float,
        norm_each_channel: bool,
        rescale: bool,
        input_len: int,
        output_len: int,
    ) -> None:
        super().__init__(dataset_name, train_ratio, norm_each_channel, rescale)
        self.norm_channel = 0
        dataset_dir = Path("datasets") / dataset_name
        with open(dataset_dir / "desc.json", "r", encoding="utf-8") as f:
            description = json.load(f)
        his_path = dataset_dir / "his.npz"
        if his_path.exists():
            series = np.load(his_path)["data"].astype(np.float32, copy=False)
        else:
            series = np.memmap(
                dataset_dir / "data.dat",
                dtype="float32",
                mode="r",
                shape=tuple(description["shape"]),
            )

        total_steps = input_len + output_len
        indices = np.arange(total_steps - 1, len(series))
        train_end = int(train_ratio * len(indices))
        last_train_idx = int(indices[:train_end][-1])
        train_slice = np.asarray(series[: last_train_idx + 1, :, self.norm_channel])

        if norm_each_channel:
            mu = train_slice.mean(axis=0, keepdims=True)
            sigma = train_slice.std(axis=0, keepdims=True)
            sigma[sigma == 0] = 1.0
        else:
            mu = train_slice.mean()
            sigma = train_slice.std()
            if sigma == 0:
                sigma = 1.0
        self.mu = torch.tensor(mu, dtype=torch.float32)
        self.sigma = torch.tensor(sigma, dtype=torch.float32)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        mu = self.mu.to(x.device)
        sigma = self.sigma.to(x.device)
        x[..., self.norm_channel] = (x[..., self.norm_channel] - mu) / sigma
        return x

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        mu = self.mu.to(x.device)
        sigma = self.sigma.to(x.device)
        x[..., self.norm_channel] = x[..., self.norm_channel] * sigma + mu
        return x


def _read_meta(path: str) -> np.ndarray:
    meta = pd.read_csv(path)
    locations = np.stack([meta["Lat"].values, meta["Lng"].values], axis=0)
    return locations


def _augment_align(dist_matrix: np.ndarray, auglen: int) -> np.ndarray:
    sorted_idx = np.argsort(dist_matrix.reshape(-1) * -1)
    sorted_idx = sorted_idx % dist_matrix.shape[-1]
    augidx = []
    for idx in sorted_idx:
        if idx not in augidx:
            augidx.append(idx)
        if len(augidx) == auglen:
            break
    return np.array(augidx, dtype=int)


def _kd_tree(locations: np.ndarray, times: int, axis: int) -> tuple[list[np.ndarray], int]:
    sorted_idx = np.argsort(locations[axis])
    part1 = np.sort(sorted_idx[: locations.shape[1] // 2])
    part2 = np.sort(sorted_idx[locations.shape[1] // 2 :])
    if times == 1:
        return [part1, part2], max(part1.shape[0], part2.shape[0])
    left_parts, left_max = _kd_tree(locations[:, part1], times - 1, axis ^ 1)
    right_parts, right_max = _kd_tree(locations[:, part2], times - 1, axis ^ 1)
    parts = [part1[part] for part in left_parts] + [part2[part] for part in right_parts]
    return parts, max(left_max, right_max)


def reorder_data(metapath: str, adjpath: str, recurtimes: int, spa_patchsize: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    locations = _read_meta(metapath)
    with open(adjpath, "rb") as f:
        adj = pickle.load(f)
    parts_idx, _ = _kd_tree(locations, recurtimes, 0)

    ori_parts_idx = np.array([], dtype=int)
    reo_parts_idx = np.array([], dtype=int)
    reo_all_idx = np.array([], dtype=int)
    for i, part_idx in enumerate(parts_idx):
        part_dist = adj[part_idx, :].copy()
        part_dist[:, part_idx] = 0
        if spa_patchsize - part_idx.shape[0] > 0:
            local_part_idx = _augment_align(part_dist, spa_patchsize - part_idx.shape[0])
            auged_part_idx = np.concatenate([part_idx, local_part_idx], axis=0)
        else:
            auged_part_idx = part_idx

        reo_parts_idx = np.concatenate([reo_parts_idx, np.arange(part_idx.shape[0]) + spa_patchsize * i])
        ori_parts_idx = np.concatenate([ori_parts_idx, part_idx])
        reo_all_idx = np.concatenate([reo_all_idx, auged_part_idx])

    return ori_parts_idx, reo_parts_idx, reo_all_idx


class WindowAttBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, num: int, size: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num = num
        self.size = size

        self.nnorm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.nattn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, attn_drop=0.1, proj_drop=0.1)
        self.nnorm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.nmlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.1)

        self.snorm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.sattn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, attn_drop=0.1, proj_drop=0.1)
        self.snorm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.smlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, time_len, node_len, dim = x.shape
        patch_num = self.num
        patch_size = self.size
        assert patch_num * patch_size == node_len
        x = x.reshape(batch_size, time_len, patch_num, patch_size, dim)

        qkv = self.snorm1(x.reshape(batch_size * time_len * patch_num, patch_size, dim))
        x = x + self.sattn(qkv).reshape(batch_size, time_len, patch_num, patch_size, dim)
        x = x + self.smlp(self.snorm2(x))

        qkv = self.nnorm1(x.transpose(2, 3).reshape(batch_size * time_len * patch_size, patch_num, dim))
        x = x + self.nattn(qkv).reshape(batch_size, time_len, patch_size, patch_num, dim).transpose(2, 3)
        x = x + self.nmlp(self.nnorm2(x))

        return x.reshape(batch_size, time_len, -1, dim)


class FaSTPatchSTG(nn.Module):
    """PatchSTG variant used by the FaST baseline configs."""

    def __init__(
        self,
        tem_patchsize: int,
        tem_patchnum: int,
        output_len: int,
        node_num: int,
        spa_patchsize: int,
        spa_patchnum: int,
        tod: int,
        dow: int,
        layers: int,
        factors: int,
        input_dims: int,
        node_dims: int,
        tod_dims: int,
        dow_dims: int,
        ori_parts_idx: np.ndarray,
        reo_parts_idx: np.ndarray,
        reo_all_idx: np.ndarray,
    ) -> None:
        super().__init__()
        self.node_num = node_num
        self.ori_parts_idx = ori_parts_idx
        self.reo_parts_idx = reo_parts_idx
        self.reo_all_idx = reo_all_idx
        self.tod = tod
        self.dow = dow
        dims = input_dims + tod_dims + dow_dims + node_dims

        self.input_st_fc = nn.Conv2d(
            in_channels=3,
            out_channels=input_dims,
            kernel_size=(1, tem_patchsize),
            stride=(1, tem_patchsize),
            bias=True,
        )
        self.node_emb = nn.Parameter(torch.empty(node_num, node_dims))
        nn.init.xavier_uniform_(self.node_emb)
        self.time_in_day_emb = nn.Parameter(torch.empty(tod, tod_dims))
        nn.init.xavier_uniform_(self.time_in_day_emb)
        self.day_in_week_emb = nn.Parameter(torch.empty(dow, dow_dims))
        nn.init.xavier_uniform_(self.day_in_week_emb)

        self.spa_encoder = nn.ModuleList(
            [
                WindowAttBlock(dims, 1, spa_patchnum // factors, spa_patchsize * factors, mlp_ratio=1)
                for _ in range(layers)
            ]
        )
        self.regression_conv = nn.Conv2d(
            in_channels=tem_patchnum * dims,
            out_channels=output_len,
            kernel_size=(1, 1),
            bias=True,
        )

    def forward(self, history_data: torch.Tensor, **kwargs) -> torch.Tensor:
        x = history_data[:, :, :, 0:1]
        te = history_data[:, :, :, 1:]
        embedded_x = self.embedding(x, te)
        rex = embedded_x[:, :, self.reo_all_idx, :]
        for block in self.spa_encoder:
            rex = block(rex)

        original = torch.zeros(rex.shape[0], rex.shape[1], self.node_num, rex.shape[-1], device=x.device)
        original[:, :, self.ori_parts_idx, :] = rex[:, :, self.reo_parts_idx, :]
        pred_y = self.regression_conv(original.transpose(2, 3).reshape(original.shape[0], -1, original.shape[-2], 1))
        return pred_y

    def embedding(self, x: torch.Tensor, te: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x1 = torch.cat([x, te[..., 0:1], te[..., 1:2]], dim=-1).float()
        input_data = self.input_st_fc(x1.transpose(1, 3)).transpose(1, 3)

        tod_idx = torch.clamp((te[:, -input_data.shape[1] :, :, 0] * self.tod).long(), min=0, max=self.tod - 1)
        input_data = torch.cat([input_data, self.time_in_day_emb[tod_idx]], dim=-1)

        dow_idx = torch.clamp((te[:, -input_data.shape[1] :, :, 1] * self.dow).long(), min=0, max=self.dow - 1)
        input_data = torch.cat([input_data, self.day_in_week_emb[dow_idx]], dim=-1)

        node_emb = self.node_emb.unsqueeze(0).unsqueeze(1).expand(batch_size, input_data.shape[1], -1, -1)
        return torch.cat([input_data, node_emb], dim=-1)
