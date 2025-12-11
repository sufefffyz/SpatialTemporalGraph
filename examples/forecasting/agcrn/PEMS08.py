import random
from basicts import BasicTSLauncher
from basicts.configs import BasicTSForecastingConfig
from basicts.models.AGCRN import AGCRN, AGCRNConfig
from basicts.runners.callback import EarlyStopping, GradientClipping
from basicts.metrics import masked_mae
from torch.optim.lr_scheduler import MultiStepLR

def main():

    input_len = 12
    output_len = 12

    model_config = AGCRNConfig(
        num_nodes=170,
        input_dim=1,
        rnn_units=64,
        output_dim=1,
        horizon=output_len,
        num_layers=2,
        default_graph=True,
        embed_dim=2,
        cheb_k=2
    )

    seeds = [42]
    print("Selected seeds:", seeds)

    for seed in seeds:
        print(f"\n****** Training with seed = {seed} ******\n")
        BasicTSLauncher.launch_training(BasicTSForecastingConfig(
            model=AGCRN,
            model_config=model_config,
            dataset_name="PEMS08",
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
            optimizer_params={"lr": 3e-3,"weight_decay": 0}
            # lr_scheduler=MultiStepLR,
            # lr_scheduler_params={
            #     "milestones": [25, 50],
            #     "gamma": 0.5
            # }
        ))



if __name__ == "__main__":
    main()
