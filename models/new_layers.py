import torch
import torch.nn as nn


class BandAttention(nn.Module):
    """Replaces: channel_spec_emb = torch.sum(torch.stack(channel_spec_emb_list), dim=0)"""
    def __init__(self, emb_size=200):
        super().__init__()
        self.score = nn.Linear(emb_size, 1)

    def forward(self, band_list):
        # band_list: list of (batch, ts, emb), length = win_level+1
        pooled = torch.stack([b.mean(dim=1) for b in band_list], dim=1)  # (batch, n_bands, emb)
        scores = self.score(pooled).squeeze(-1)                          # (batch, n_bands)
        weights = torch.softmax(scores, dim=-1)                          # (batch, n_bands)
        stacked = torch.stack(band_list, dim=1)                          # (batch, n_bands, ts, emb)
        combined = (weights.unsqueeze(-1).unsqueeze(-1) * stacked).sum(dim=1)  # (batch, ts, emb)
        return combined, weights   # weights: your band-importance output


class ChannelAttention(nn.Module):
    """Replaces: emb = self.transformer(emb).mean(dim=1)"""
    def __init__(self, emb_size=200):
        super().__init__()
        self.score = nn.Linear(emb_size, 1)

    def forward(self, x, n_channels, ts_per_channel):
        # x: (batch, n_channels*ts, emb) -- output of self.transformer(emb)
        batch, _, emb = x.shape
        x = x.view(batch, n_channels, ts_per_channel, emb)
        pooled = x.mean(dim=2)                          # (batch, n_channels, emb)
        scores = self.score(pooled).squeeze(-1)          # (batch, n_channels)
        weights = torch.softmax(scores, dim=-1)           # (batch, n_channels)
        combined = (weights.unsqueeze(-1) * pooled).sum(dim=1)  # (batch, emb)
        return combined, weights   # weights: your electrode-importance output


