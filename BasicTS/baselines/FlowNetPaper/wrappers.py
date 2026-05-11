from baselines.iTransformer.arch import iTransformer


class ITransformer4D(iTransformer):
    def forward(self, *args, **kwargs):
        prediction = super().forward(*args, **kwargs)
        if prediction.ndim == 3:
            prediction = prediction.unsqueeze(-1)
        return prediction
