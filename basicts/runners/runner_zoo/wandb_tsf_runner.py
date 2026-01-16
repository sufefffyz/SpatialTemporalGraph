from typing import Dict, Optional

import torch
import wandb
from .simple_tsf_runner import SimpleTimeSeriesForecastingRunner
from easytorch.utils import (TimePredictor, get_local_rank, get_logger,
                             is_master, master_only, set_env)

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
    
    def init_validation(self, cfg: Dict):
        super().init_validation(cfg)
        self.logger.info("WandB Runner: Initialized validation with WandB logging.")
        self._wandb_init(cfg)
    
    @master_only
    def _wandb_init(self, cfg: Dict):
        wandb.init(project="SpatialTemporalModel", 
                   name=f'{self.model_name}_{self.dataset_name}',
                   config=cfg)
        
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

