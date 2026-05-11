import os
from typing import Dict, Optional

import wandb
from easytorch.utils import master_only

from .simple_tsf_runner import SimpleTimeSeriesForecastingRunner


class PemsWandBTimeSeriesForecastingRunner(SimpleTimeSeriesForecastingRunner):
    """
    MVP-specific WandB runner copy.

    Unlike the stock WandB runner, this copy logs:
    - train metrics every validation epoch
    - val metrics every validation epoch
    - test metrics whenever test_pipeline runs
    """

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        self.model_name = cfg['MODEL']['NAME']
        self.dataset_name = cfg['DATASET']['NAME']
        self._wandb_initialized = False
        self.wandb_cfg = cfg.get('WANDB', {})

    def init_validation(self, cfg: Dict):
        super().init_validation(cfg)
        self._wandb_init(cfg)

    @master_only
    def _wandb_init(self, cfg: Dict):
        if self._wandb_initialized:
            return

        project = os.environ.get('WANDB_PROJECT') or self.wandb_cfg.get('PROJECT', 'SpatialTemporalModel')
        entity = os.environ.get('WANDB_ENTITY') or self.wandb_cfg.get('ENTITY', None) or None
        mode = os.environ.get('WANDB_MODE') or self.wandb_cfg.get('MODE', 'online')
        run_name = os.environ.get('WANDB_NAME') or self.wandb_cfg.get('RUN_NAME', f'{self.model_name}_{self.dataset_name}')
        group = os.environ.get('WANDB_RUN_GROUP') or self.wandb_cfg.get('GROUP', None)
        tags = self.wandb_cfg.get('TAGS', None)
        env_tags = os.environ.get('WANDB_TAGS')
        if env_tags:
            tags = [tag.strip() for tag in env_tags.split(',') if tag.strip()]

        init_kwargs = {
            'project': project,
            'name': run_name,
            'config': cfg,
            'mode': mode,
        }
        if entity is not None:
            init_kwargs['entity'] = entity
        if group is not None:
            init_kwargs['group'] = group
        if tags is not None:
            init_kwargs['tags'] = tags

        wandb.init(**init_kwargs)
        wandb.watch(self.model, log='all')
        self._wandb_initialized = True

    @master_only
    def _wandb_log_metrics(self, prefix: str, metric_names, step: int) -> None:
        payload = {}
        for metric_name in metric_names:
            meter_key = f'{prefix}/{metric_name}'
            try:
                metric_val = self.meter_pool.get_value(meter_key)
            except KeyError:
                continue
            payload[meter_key] = metric_val
        if payload:
            wandb.log(payload, step=step)
            for key, value in payload.items():
                wandb.run.summary[key] = value

    def on_validating_end(self, train_epoch: Optional[int] = None):
        super().on_validating_end(train_epoch)
        step = train_epoch if train_epoch is not None else 0
        self._wandb_log_metrics('train', ['loss', *self.metrics.keys()], step)
        self._wandb_log_metrics('val', ['loss', *self.metrics.keys()], step)

        target_metric_name = f'val/{self.target_metrics}'
        if target_metric_name in self.best_metrics:
            best_metric = self.best_metrics[target_metric_name]
            best_key = f'best/{self.target_metrics}'
            wandb.log({best_key: best_metric}, step=step)
            wandb.run.summary[best_key] = best_metric

    def on_test_end(self) -> None:
        super().on_test_end()
        # Use current epoch count if available; final best-model test reuses the latest epoch.
        step = getattr(self, 'start_epoch', 0) or 0
        self._wandb_log_metrics('test', ['loss', *self.metrics.keys()], step)
        if len(self.evaluation_horizons) > 0:
            payload = {}
            for horizon in self.evaluation_horizons:
                for metric_name in self.metrics.keys():
                    meter_key = f'test/{metric_name}@h{horizon + 1}'
                    try:
                        payload[meter_key] = self.meter_pool.get_value(meter_key)
                    except KeyError:
                        continue
            if payload:
                wandb.log(payload, step=step)
                for key, value in payload.items():
                    wandb.run.summary[key] = value

    def on_training_end(self, cfg: Dict, train_epoch: Optional[int] = None):
        super().on_training_end(cfg, train_epoch)
        self._wandb_close()

    @master_only
    def _wandb_close(self):
        if self._wandb_initialized:
            wandb.finish()
