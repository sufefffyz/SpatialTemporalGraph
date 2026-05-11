import torch
from torch import nn

from .time_embed import TemporalEmbedding


class FeatPatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, freq, padding=0):
        super(FeatPatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))
        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.time_embed = TemporalEmbedding(d_model, freq=freq)
        self.value_embedding = nn.Linear(patch_len, d_model)
        self.time_linear = nn.Linear((self.patch_len + 1) * d_model, d_model)
        self.initialize()
        
    def initialize(self):
        nn.init.xavier_uniform_(self.value_embedding.weight)
        nn.init.zeros_(self.value_embedding.bias)
        nn.init.xavier_uniform_(self.time_linear.weight)
        nn.init.zeros_(self.time_linear.bias)

    def forward(self, x_enc, x_mark_enc):
        # do patching
        x_enc = self.padding_patch_layer(x_enc)
        x_enc = x_enc.unfold(dimension=1, size=self.patch_len, step=self.stride)
        # Input encoding
        x_enc = self.value_embedding(x_enc)

        # x_mark_enc = self.time_embed(x_mark_enc)
        # x_mark_enc = x_mark_enc.unfold(
        #     dimension=1, size=self.patch_len, step=self.stride
        # )
        # x_mark_enc = x_mark_enc.reshape(x_enc.shape[0], x_enc.shape[1], -1)
        # x_mark_enc = x_mark_enc.unsqueeze(2).repeat(1, 1, x_enc.shape[2], 1)
        # result = torch.concat((x_enc, x_mark_enc), dim=-1)
        # result = self.time_linear(result)
        # return result
        
        return x_enc
