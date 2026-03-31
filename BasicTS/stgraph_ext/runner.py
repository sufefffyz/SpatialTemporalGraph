from typing import Any, Dict, Optional

from easytorch.utils import master_only

from basicts.runners.runner_zoo.simple_tsf_runner import SimpleTimeSeriesForecastingRunner

from .adjacency import save_adjacency_snapshot


class GraphSnapshotTimeSeriesForecastingRunner(SimpleTimeSeriesForecastingRunner):
    """Runner that periodically saves learned adjacency matrices."""

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        snapshot_cfg = cfg.get("GRAPH_SNAPSHOT", {})
        self.graph_snapshot_enabled = snapshot_cfg.get("ENABLED", True)
        self.graph_snapshot_interval = int(snapshot_cfg.get("INTERVAL", 5))
        self.graph_snapshot_split = snapshot_cfg.get("SAMPLE_SPLIT", "valid")
        self.dataset_name = cfg["DATASET"]["NAME"]

    def _get_snapshot_batch(self) -> Optional[dict]:
        loader = None
        if self.graph_snapshot_split == "valid" and getattr(self, "val_data_loader", None) is not None:
            loader = self.val_data_loader
        elif self.graph_snapshot_split == "test" and getattr(self, "test_data_loader", None) is not None:
            loader = self.test_data_loader
        elif getattr(self, "train_data_loader", None) is not None:
            loader = self.train_data_loader
        elif getattr(self, "val_data_loader", None) is not None:
            loader = self.val_data_loader
        elif getattr(self, "test_data_loader", None) is not None:
            loader = self.test_data_loader

        if loader is None:
            return None
        return next(iter(loader))

    @master_only
    def _save_graph_snapshot(self, tag: str, epoch: int | str) -> None:
        if not self.graph_snapshot_enabled:
            return
        sample_batch = self._get_snapshot_batch()
        output_path = save_adjacency_snapshot(self, tag=tag, epoch=epoch, sample_batch=sample_batch)
        self.logger.info("Saved learned graph snapshot to %s", output_path)

    def on_epoch_end(self, epoch: int) -> None:
        super().on_epoch_end(epoch)
        if self.graph_snapshot_enabled and self.graph_snapshot_interval > 0 and epoch % self.graph_snapshot_interval == 0:
            self._save_graph_snapshot(tag=f"epoch_{epoch:03d}", epoch=epoch)

    def on_training_end(self, cfg: Dict, train_epoch: Optional[int] = None):
        final_epoch = train_epoch if train_epoch is not None else getattr(self, "num_epochs", "final")
        self._save_graph_snapshot(tag="final", epoch=final_epoch)
        super().on_training_end(cfg, train_epoch=train_epoch)
