import random
from basicts import BasicTSLauncher
from basicts.configs import BasicTSForecastingConfig
from basicts.models.STGCN import STGCN, STGCNConfig
from basicts.runners.callback import EarlyStopping
from basicts.metrics import masked_mae
from torch.optim.lr_scheduler import MultiStepLR
from basicts.utils import load_adj
import torch
def main():

    input_len = 12
    output_len = 12
    DATA_NAME = 'PEMS04'  # Dataset name
    adj_mx, _ = load_adj(DATA_NAME,'normlap')
    adj_mx = torch.Tensor(adj_mx[0])
    model_config = STGCNConfig(
        Ks=3,
        Kt=3,
        blocks=[[1], [64, 16, 64], [64, 16, 64], [128, 128], [12]],
        T=input_len,
        n_vertex=307,
        act_func="glu",
        graph_conv_type="cheb_graph_conv",
        gso=adj_mx,
        bias=True,
        droprate=0.5
    )

    seeds = [42]
    print("Selected seeds:", seeds)

    for seed in seeds:
        print(f"\n****** Training with seed = {seed} ******\n")
        BasicTSLauncher.launch_training(BasicTSForecastingConfig(
            model=STGCN,
            model_config=model_config,
            dataset_name="PEMS04",
            input_len=input_len,
            output_len=output_len,
            gpus="0",
            callbacks=[EarlyStopping(15)],
            seed=seed,        
            num_epochs=2,
            batch_size=64,
            norm_each_channel=False,
            rescale=True,
            metrics=["RMSE", "MAE", "MAPE","WAPE"],
            loss=masked_mae,
            optimizer_params={"lr": 0.003, "weight_decay": 0}

            # lr_scheduler=MultiStepLR,
            # lr_scheduler_params={
            #     "milestones": [1, 50],
            #     "gamma": 0.5
            # }
        ))


if __name__ == "__main__":
    main()