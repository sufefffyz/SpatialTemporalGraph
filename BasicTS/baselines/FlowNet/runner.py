from typing import Optional

import torch

from basicts.runners import WandBTimeSeriesForecastingRunner


class FlowNetOfficialWandBRunner(WandBTimeSeriesForecastingRunner):
    """WandB runner with FlowNet's official normalized loss and early-stopping warmup."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.early_stopping_warmup = cfg.get("TRAIN", {}).get("EARLY_STOPPING_WARMUP", 0)

    def forward(self, data, epoch: Optional[int] = None, iter_num: Optional[int] = None, train: bool = True, **kwargs):
        data = self.preprocessing(data)

        future_data, history_data = data["target"], data["inputs"]
        history_data = self.to_running_device(history_data)
        future_data = self.to_running_device(future_data)
        batch_size, length, num_nodes, _ = future_data.shape

        history_data = self.select_input_features(history_data)
        future_data_4_dec = self.select_input_features(future_data)
        if not train:
            future_data_4_dec[..., 0] = torch.empty_like(future_data_4_dec[..., 0])

        model_return = self.model(
            history_data=history_data,
            future_data=future_data_4_dec,
            batch_seen=iter_num,
            epoch=epoch,
            train=train,
        )
        if isinstance(model_return, torch.Tensor):
            model_return = {"prediction": model_return}
        if "inputs" not in model_return:
            model_return["inputs"] = self.select_target_features(history_data)
        if "target" not in model_return:
            model_return["target"] = self.select_target_features(future_data)

        assert list(model_return["prediction"].shape)[:3] == [batch_size, length, num_nodes], (
            "The shape of the output is incorrect. Ensure it matches [B, L, N, C]."
        )

        if not train:
            model_return = self.postprocessing(model_return)
        return model_return

    def on_validating_end(self, train_epoch: Optional[int] = None):
        if train_epoch is not None and train_epoch <= self.early_stopping_warmup:
            patience = self.early_stopping_patience
            self.early_stopping_patience = None
            super().on_validating_end(train_epoch)
            self.early_stopping_patience = patience
            self.current_patience = patience
            return

        super().on_validating_end(train_epoch)
