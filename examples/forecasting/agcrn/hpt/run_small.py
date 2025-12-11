import argparse
import random
import torch
from basicts import BasicTSLauncher
from basicts.configs import BasicTSForecastingConfig
from basicts.models.AGCRN import AGCRN, AGCRNConfig
from basicts.runners.callback import EarlyStopping, WandBLogging
from basicts.metrics import masked_mae
from basicts.utils import load_adj
# from torch.optim.lr_scheduler import MultiStepLR


def parse_args():
    parser = argparse.ArgumentParser(description="Train STModel on traffic dataset.")

    parser.add_argument("--data_name", type=str, default="PEMS07",
                        help="Dataset name, e.g. PEMS07")
    parser.add_argument("--input_len", type=int, default=12)
    parser.add_argument("--output_len", type=int, default=12)
    parser.add_argument("--n_vertex", type=int, default=883)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--embed_dim", type=int, default=2)
    parser.add_argument("--gpus", type=str, default="0")

    return parser.parse_args()


def set_seed(seed: int):
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    print("Args:", args)

    # set_seed(args.seed)

    input_len = args.input_len
    output_len = args.output_len
    DATA_NAME = args.data_name

    model_config = AGCRNConfig(
        num_nodes=args.n_vertex,
        input_dim=1,
        rnn_units=64,
        output_dim=1,
        horizon=output_len,
        num_layers=2,
        default_graph=True,
        embed_dim=args.embed_dim,
        cheb_k=2
    )


    ckpt_save_dir = (
        f"checkpoints/hypertuning/{AGCRN.__name__}/"
        f"{DATA_NAME}_100_{input_len}_{output_len}/"
        f"lr{args.lr}_weight_decay{args.weight_decay}_seed{args.seed}"
    )

    print(f"\n****** Training with seed = {args.seed}, "
          f"lr = {args.lr}, weight_decay = {args.weight_decay} ******\n")
    print("Checkpoint dir:", ckpt_save_dir)


    BasicTSLauncher.launch_training(BasicTSForecastingConfig(
        model=AGCRN,
        model_config=model_config,
        dataset_name=DATA_NAME,
        input_len=input_len,
        output_len=output_len,
        gpus=args.gpus,
        callbacks=[
            EarlyStopping(15),
            WandBLogging()
        ],
        seed=args.seed,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        norm_each_channel=False,
        rescale=True,
        metrics=["RMSE", "MAE", "MAPE", "WAPE"],
        loss=masked_mae,
        optimizer_params={
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        },
        ckpt_save_dir=ckpt_save_dir,
        # lr_scheduler=MultiStepLR,
        # lr_scheduler_params={
        #     "milestones": [1, 50],
        #     "gamma": 0.5
        # }
    ))


if __name__ == "__main__":
    main()

