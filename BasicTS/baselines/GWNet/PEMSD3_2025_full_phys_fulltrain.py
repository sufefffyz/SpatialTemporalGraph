import os
import sys
import torch
from easydict import EasyDict
sys.path.append(os.path.abspath(__file__ + '/../../..'))

from basicts.metrics import masked_mae, masked_mape, masked_rmse, masked_wape
from basicts.data import RecentWindowTimeSeriesForecastingDataset
from basicts.runners import PemsWandBTimeSeriesForecastingRunner
from basicts.scaler import ZScoreScaler
from basicts.utils import get_regular_settings, load_adj, load_dataset_desc

from .arch import GraphWaveNet

DATA_NAME = 'PEMSD3_2025_full_phys'
regular_settings = get_regular_settings(DATA_NAME)
INPUT_LEN = regular_settings['INPUT_LEN']
OUTPUT_LEN = regular_settings['OUTPUT_LEN']
TRAIN_VAL_TEST_RATIO = regular_settings['TRAIN_VAL_TEST_RATIO']
NORM_EACH_CHANNEL = regular_settings['NORM_EACH_CHANNEL']
RESCALE = regular_settings['RESCALE']
NULL_VAL = regular_settings['NULL_VAL']

MODEL_ARCH = GraphWaveNet
adj_mx, _ = load_adj('datasets/' + DATA_NAME + '/adj_mx.pkl', 'doubletransition')
desc = load_dataset_desc(DATA_NAME)
NUM_NODES = desc['num_nodes']

MODEL_PARAM = {
    'num_nodes': NUM_NODES,
    'supports': [torch.tensor(i, dtype=torch.float32) for i in adj_mx],
    'dropout': 0.3,
    'gcn_bool': True,
    'addaptadj': False,
    'aptinit': None,
    'in_dim': 3,
    'out_dim': OUTPUT_LEN,
    'residual_channels': 32,
    'dilation_channels': 32,
    'skip_channels': 128,
    'end_channels': 256,
    'kernel_size': 2,
    'blocks': 4,
    'layers': 2,
}

CFG = EasyDict()
CFG.DESCRIPTION = 'PeMS MVP pilot on ' + DATA_NAME + ' (fulltrain)'
CFG.GPU_NUM = 1
CFG.RUNNER = PemsWandBTimeSeriesForecastingRunner

CFG.ENV = EasyDict()
CFG.ENV.SEED = 42
CFG.ENV.DETERMINISTIC = True
CFG.ENV.CUDNN = EasyDict()
CFG.ENV.CUDNN.ENABLED = True
CFG.ENV.CUDNN.BENCHMARK = True
CFG.ENV.CUDNN.DETERMINISTIC = True

CFG.WANDB = EasyDict()
CFG.WANDB.PROJECT = os.environ.get('WANDB_PROJECT', 'SpatialTemporalModel')
CFG.WANDB.ENTITY = os.environ.get('WANDB_ENTITY', '')
CFG.WANDB.MODE = os.environ.get('WANDB_MODE', 'online')
CFG.WANDB.RUN_NAME = f'{MODEL_ARCH.__name__}_{DATA_NAME}_fulltrain'
CFG.WANDB.GROUP = DATA_NAME
CFG.WANDB.TAGS = ['pems-mvp', 'fulltrain']

CFG.DATASET = EasyDict()
CFG.DATASET.NAME = DATA_NAME
CFG.DATASET.TYPE = RecentWindowTimeSeriesForecastingDataset
CFG.DATASET.PARAM = EasyDict({
    'dataset_name': DATA_NAME,
    'train_val_test_ratio': TRAIN_VAL_TEST_RATIO,
    'input_len': INPUT_LEN,
    'output_len': OUTPUT_LEN,
    'train_recent_days': None,
})

CFG.SCALER = EasyDict()
CFG.SCALER.TYPE = ZScoreScaler
CFG.SCALER.PARAM = EasyDict({
    'dataset_name': DATA_NAME,
    'train_ratio': TRAIN_VAL_TEST_RATIO[0],
    'norm_each_channel': NORM_EACH_CHANNEL,
    'rescale': RESCALE,
})

CFG.MODEL = EasyDict()
CFG.MODEL.NAME = MODEL_ARCH.__name__
CFG.MODEL.ARCH = MODEL_ARCH
CFG.MODEL.PARAM = MODEL_PARAM
CFG.MODEL.FORWARD_FEATURES = [0, 1, 2]
CFG.MODEL.TARGET_FEATURES = [0]

CFG.METRICS = EasyDict()
CFG.METRICS.FUNCS = EasyDict({
    'MAE': masked_mae,
    'MAPE': masked_mape,
    'RMSE': masked_rmse,
    'WAPE': masked_wape,
})
CFG.METRICS.TARGET = 'MAE'
CFG.METRICS.NULL_VAL = NULL_VAL

CFG.TRAIN = EasyDict()
CFG.TRAIN.NUM_EPOCHS = 30
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    'checkpoints',
    MODEL_ARCH.__name__,
    '_'.join([DATA_NAME, 'fulltrain', str(CFG.TRAIN.NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN)])
)
CFG.TRAIN.LOSS = masked_mae
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = 'Adam'
CFG.TRAIN.OPTIM.PARAM = {
    'lr': 0.002,
    'weight_decay': 0.0001,
}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = 'MultiStepLR'
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    'milestones': [10, 20],
    'gamma': 0.5,
}
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = 64
CFG.TRAIN.DATA.SHUFFLE = True
CFG.TRAIN.CLIP_GRAD_PARAM = {'max_norm': 5.0}
CFG.TRAIN.EARLY_STOPPING_PATIENCE = 10

CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.BATCH_SIZE = 64

CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 10
CFG.TEST.DATA = EasyDict()
CFG.TEST.DATA.BATCH_SIZE = 64

CFG.EVAL = EasyDict()
CFG.EVAL.HORIZONS = [3, 6, 12]
CFG.EVAL.USE_GPU = True
