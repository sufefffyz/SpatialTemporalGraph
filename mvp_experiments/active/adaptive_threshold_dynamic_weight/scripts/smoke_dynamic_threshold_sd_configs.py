#!/usr/bin/env python3
from __future__ import annotations

import importlib

import torch


def main() -> None:
    torch.set_grad_enabled(False)
    modules = [
        "baselines.GWNet.SD_dynamic_threshold",
        "baselines.DCRNN.SD_dynamic_threshold",
        "baselines.STGCN.SD_dynamic_threshold",
    ]
    for module_name in modules:
        cfg = importlib.import_module(module_name).CFG
        model = cfg.MODEL.ARCH(**cfg.MODEL.PARAM).eval()
        num_nodes = cfg.MODEL.PARAM.get("num_nodes", cfg.MODEL.PARAM.get("n_vertex"))
        num_features = len(cfg.MODEL.FORWARD_FEATURES)
        history = torch.randn(1, 12, num_nodes, num_features)
        future = torch.randn(1, 12, num_nodes, num_features)
        output = model(history, future, batch_seen=1, epoch=1, train=False)
        assert tuple(output.shape) == (1, 12, num_nodes, 1), (module_name, tuple(output.shape))
        print(f"{module_name}: {tuple(output.shape)}")


if __name__ == "__main__":
    main()
