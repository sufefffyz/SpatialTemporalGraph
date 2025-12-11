import random
from basicts import BasicTSLauncher
from basicts.configs import BasicTSForecastingConfig
from basicts.models.DCRNN import DCRNN, DCRNNConfig
from basicts.runners.callback import EarlyStopping, GradientClipping
from basicts.metrics import masked_mae
from torch.optim.lr_scheduler import MultiStepLR
from basicts.utils import load_adj
import torch
def main():

    input_len = 12
    output_len = 12
    DATA_NAME = 'METR-LA'  # Dataset name
    adj_mx, _ = load_adj(DATA_NAME,'doubletransition')
    model_config = DCRNNConfig(
        cl_decay_steps=2000,
        horizon=output_len,
        input_dim=2,
        max_diffusion_step=2,
        num_nodes=207,
        num_rnn_layers=2,
        output_dim=1,
        rnn_units=64,
        seq_len=input_len,
        adj_mx=[torch.tensor(i) for i in adj_mx],
        use_curriculum_learning=True,
        if_time_in_day=True,
        if_day_in_week=False
    )

    seeds = [42]
    print("Selected seeds:", seeds)

    for seed in seeds:
        print(f"\n****** Training with seed = {seed} ******\n")
        BasicTSLauncher.launch_training(BasicTSForecastingConfig(
            model=DCRNN,
            model_config=model_config,
            dataset_name="METR-LA",
            input_len=input_len,
            output_len=output_len,
            gpus="0",
            callbacks=[EarlyStopping(15)],
            seed=seed,        
            num_epochs=100,
            batch_size=64,
            norm_each_channel=False,
            rescale=True,
            metrics=["RMSE", "MAE","MAPE"],
            loss=masked_mae,
            optimizer_params={"lr": 0.01,"weight_decay": 0},
            lr_scheduler=MultiStepLR,
            lr_scheduler_params={
                "milestones": [20,30,40,50],
                "gamma": 0.1
            }
        ))



if __name__ == "__main__":
    main()
