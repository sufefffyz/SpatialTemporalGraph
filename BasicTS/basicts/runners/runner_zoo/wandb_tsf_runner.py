import os
from typing import Dict, Optional

import wandb
from .simple_tsf_runner import SimpleTimeSeriesForecastingRunner
from easytorch.utils import master_only

class WandBTimeSeriesForecastingRunner(SimpleTimeSeriesForecastingRunner):
    """
    A WandB Runner for Time Series Forecasting: 
    Extends the SimpleTimeSeriesForecastingRunner to include WandB logging functionality.

    Args:
        cfg (Dict): Configuration dictionary.
    """

    def __init__(self, cfg: Dict):

        super().__init__(cfg)

        self.model_name = cfg['MODEL']['NAME']
        self.dataset_name = cfg['DATASET']['NAME']
        self.wandb_cfg = cfg.get('WANDB', {})
    
    def init_validation(self, cfg: Dict):
        super().init_validation(cfg)
        self.logger.info("WandB Runner: Initialized validation with WandB logging.")
        self._wandb_init(cfg)
    
    @master_only
    def _wandb_init(self, cfg: Dict):
        project = os.environ.get("WANDB_PROJECT") or self.wandb_cfg.get("PROJECT", "SpatialTemporalModel")
        entity = os.environ.get("WANDB_ENTITY") or self.wandb_cfg.get("ENTITY", None)
        mode = os.environ.get("WANDB_MODE") or self.wandb_cfg.get("MODE", None)
        run_name = os.environ.get("WANDB_NAME") or self.wandb_cfg.get("RUN_NAME", f"{self.model_name}_{self.dataset_name}")
        group = os.environ.get("WANDB_RUN_GROUP") or self.wandb_cfg.get("GROUP", None)
        tags = self.wandb_cfg.get("TAGS", None)
        env_tags = os.environ.get("WANDB_TAGS")
        if env_tags:
            tags = [tag.strip() for tag in env_tags.split(",") if tag.strip()]

        init_kwargs = {
            "project": project,
            "name": run_name,
            "config": cfg,
        }
        if entity is not None:
            init_kwargs["entity"] = entity
        if mode is not None:
            init_kwargs["mode"] = mode
        if group is not None:
            init_kwargs["group"] = group
        if tags is not None:
            init_kwargs["tags"] = tags

        wandb.init(**init_kwargs)
        
        wandb.watch(self.model, log="all")

    def on_validating_end(self, train_epoch: Optional[int] = None):
        super().on_validating_end(train_epoch)
        self._wandb_log_val(train_epoch)

    @master_only
    def _wandb_log_val(self, train_epoch):
        self.logger.info("Logging validation metrics to WandB...")

        step = train_epoch

        for metric_name in self.metrics:
            metric = self.meter_pool.get_value(f"val/{metric_name}")
            key = f"Current_{metric_name}"

            wandb.log({key: metric}, step=step)
            wandb.run.summary[key] = metric

        target_metric_name = f"val/{self.target_metrics}"
        best_metric = self.best_metrics[target_metric_name]
        best_key = f"Best_{self.target_metrics}"

        wandb.log({best_key: best_metric}, step=step)
        wandb.run.summary[best_key] = best_metric

    
    def on_training_end(self, cfg: Dict, train_epoch: Optional[int] = None):

        super().on_training_end(cfg, train_epoch)
        self._wandb_close()

    @master_only
    def _wandb_close(self):
        self.logger.info("Close the WandB run...")
        wandb.finish()
