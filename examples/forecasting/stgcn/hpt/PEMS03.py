import random
from basicts import BasicTSLauncher
from basicts.configs import BasicTSForecastingConfig
from basicts.models.STGCN import STGCN, STGCNConfig
from basicts.runners.callback import EarlyStopping
from basicts.metrics import masked_mae
from torch.optim.lr_scheduler import MultiStepLR
from basicts.utils import load_adj
import torch
from basicts.runners.callback import WandBLogging

def main():

    input_len = 12
    output_len = 12
    DATA_NAME = 'PEMS03'  # Dataset name
    adj_mx, _ = load_adj(DATA_NAME,'normlap')
    adj_mx = torch.Tensor(adj_mx[0])
    
    model_config = STGCNConfig(
        Ks=3,
        Kt=3,
        blocks=[[1], [64, 16, 64], [64, 16, 64], [128, 128], [12]],
        T=input_len,
        n_vertex=358,
        act_func="glu",
        graph_conv_type="cheb_graph_conv",
        gso=adj_mx,
        bias=True,
        droprate=0.5
    )

    seeds = [42]
    learning_rates = [0.0007, 0.001, 0.003, 0.005, 0.009]  # List of different learning rates
    print("Selected seeds:", seeds)
    print("Learning rates to test:", learning_rates)

    for seed in seeds:
        for lr in learning_rates:  # Loop through each learning rate
            print(f"\n****** Training with seed = {seed} and lr = {lr} ******\n")
            BasicTSLauncher.launch_training(BasicTSForecastingConfig(
                model=STGCN,
                model_config=model_config,
                dataset_name="PEMS03",
                input_len=input_len,
                output_len=output_len,
                gpus="0",
                callbacks=[EarlyStopping(15), WandBLogging()],  # Add WandBCallback to callbacks
                seed=seed,        
                num_epochs=5,
                batch_size=64,
                norm_each_channel=False,
                rescale=True,
                metrics=["RMSE", "MAE", "MAPE","WAPE"],
                loss=masked_mae,
                optimizer_params={"lr": lr, "weight_decay": 0},  # Use the current lr
                ckpt_save_dir=f'checkpoints/hypertuning/{STGCN.__name__}/{DATA_NAME}_100_12_12/lr{lr}weight_decay0',  # Unique checkpoint directory
                # lr_scheduler=MultiStepLR,
                # lr_scheduler_params={
                #     "milestones": [1, 50],
                #     "gamma": 0.5
                # }
            ))

if __name__ == "__main__":
    main()
