import torch as th
from torch import nn


class CalendarEmbedding(nn.Module):

    def __init__(self, embedding_dim):
        super(CalendarEmbedding, self).__init__()
        self.day_embed = nn.Embedding(96, embedding_dim)
        self.week_embed = nn.Embedding(7, embedding_dim)
        self.holiday_embed = nn.Embedding(2, embedding_dim)

    def forward(self, x):
        assert x.shape[-1] == 3
        return th.concat(
            [
                self.day_embed(x[..., 0]),
                self.week_embed(x[..., 1]),
                self.holiday_embed(x[..., 2]),
            ],
            dim=-1,
        )


class MeteorEmbedding(nn.Module):

    def __init__(self, embedding_dim):
        super(MeteorEmbedding, self).__init__()
        self.embed = nn.Linear(3, embedding_dim)

    def forward(self, x):
        assert x.shape[-1] == 3  # Temperature, Humidity, Wind Speed
        return self.embed(x)


class RegionEmbedding(nn.Module):

    def __init__(self, embedding_dim):
        super(RegionEmbedding, self).__init__()
        self.util_embed = nn.Embedding(10, embedding_dim)
        self.region_embed = nn.Embedding(36, embedding_dim)

    def forward(self, x):
        assert x.shape[-1] == 2  # Utilization, Region
        return th.concat(
            [self.util_embed(x[..., 0]), self.region_embed(x[..., 1])], dim=-1
        )


class LocEmbedding(nn.Module):

    def __init__(self, embedding_dim):
        super(LocEmbedding, self).__init__()
        self.embed = nn.Linear(3, embedding_dim)

    def forward(self, x):
        assert x.shape[-1] == 3  # Road Density, Latitude, Longitude
        return self.embed(x)


class FeatureFusion(nn.Module):

    def __init__(self, feat_dim):
        super(FeatureFusion, self).__init__()
        ca_dim = 4
        me_dim = 12
        re_dim = 6
        loc_embed = 12
        self.calendar = CalendarEmbedding(ca_dim)
        self.meteor = MeteorEmbedding(me_dim)
        self.region = RegionEmbedding(re_dim)
        self.loc = LocEmbedding(loc_embed)
        self.fusion = nn.Linear(48, feat_dim)
        self.initialize()

    def initialize(self):
        nn.init.xavier_normal_(self.fusion.weight)
        nn.init.zeros_(self.fusion.bias)

    def forward(self, x):
        ca_feat = self.calendar(x[..., 0:3].long())
        me_feat = self.meteor(x[..., 3:6])
        re_feat = self.region(x[..., 6:8].long())
        loc_feat = self.loc(x[..., 8:])
        feat = th.concat([ca_feat, me_feat, re_feat, loc_feat], dim=-1)
        return self.fusion(feat)
