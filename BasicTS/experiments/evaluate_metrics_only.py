# pylint: disable=wrong-import-position
import os
import sys
import traceback
from argparse import ArgumentParser

sys.path.append(os.path.abspath(__file__ + "/../.."))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from easytorch.config import init_cfg
from easytorch.device import set_device_type
from easytorch.utils import get_logger, set_visible_devices


def parse_args():
    parser = ArgumentParser(description="Evaluate a BasicTS model and save metrics only.")
    parser.add_argument("-cfg", "--config", required=True, help="evaluation config")
    parser.add_argument("-ckpt", "--checkpoint", required=True, help="checkpoint path")
    parser.add_argument("-g", "--gpus", default="0", help="visible gpus")
    parser.add_argument("-d", "--device_type", default="gpu")
    parser.add_argument("-b", "--batch_size", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg_path = args.config
    ckpt_path = args.checkpoint
    while isinstance(cfg_path, str) and cfg_path.startswith(("./", ".\\")):
        cfg_path = cfg_path[2:]
    while isinstance(ckpt_path, str) and ckpt_path.startswith(("./", ".\\")):
        ckpt_path = ckpt_path[2:]

    cfg = init_cfg(cfg_path, save=True)
    set_device_type(args.device_type)
    if args.device_type != "cpu":
        set_visible_devices(args.gpus)

    logger = get_logger("easytorch-launcher")
    logger.info(f"Initializing runner '{cfg['RUNNER']}'")
    runner = cfg["RUNNER"](cfg)
    runner.init_logger(logger_name="easytorch-evaluation", log_file_name="evaluation_log")

    if runner.need_setup_graph:
        runner.setup_graph(cfg=cfg, train=False)

    try:
        if args.batch_size is not None:
            cfg.TEST.DATA.BATCH_SIZE = int(args.batch_size)
        else:
            assert "BATCH_SIZE" in cfg.TEST.DATA

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint file not found at {ckpt_path}")
        logger.info(f"Loading model checkpoint from {ckpt_path}")
        runner.load_model(ckpt_path=ckpt_path, strict=True)
        runner.test_pipeline(cfg=cfg, save_metrics=True, save_results=False)
    except BaseException as exc:
        runner.logger.error(traceback.format_exc())
        raise exc


if __name__ == "__main__":
    main()
