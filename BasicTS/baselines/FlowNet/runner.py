from typing import Optional

from basicts.runners import WandBTimeSeriesForecastingRunner


class FlowNetOfficialWandBRunner(WandBTimeSeriesForecastingRunner):
    """WandB runner with FlowNet's official early-stopping warmup."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.early_stopping_warmup = cfg.get("TRAIN", {}).get("EARLY_STOPPING_WARMUP", 0)

    def on_validating_end(self, train_epoch: Optional[int] = None):
        if train_epoch is not None and train_epoch <= self.early_stopping_warmup:
            patience = self.early_stopping_patience
            self.early_stopping_patience = None
            super().on_validating_end(train_epoch)
            self.early_stopping_patience = patience
            self.current_patience = patience
            return

        super().on_validating_end(train_epoch)
