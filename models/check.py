import torch
import sys
sys.path.insert(0, '.')
from models.model_multiscale import MSSEncoder
 
m = MSSEncoder(n_channels=26, emb_size=200, input_length=3200, scale1=100, win_level=3)
x = torch.randn(2, 26, 3200)  # (batch=2, channels=26, time=3200)
 
out = m(x)
print("output shape:", out.shape)                        # expect (2, 200)
print("last_channel_weights shape:", m.last_channel_weights.shape)   # expect (2, 26)
print("last_band_weights shape:", m.last_band_weights.shape)         # expect (2, 26, 4)
print("channel weights sum to 1 per sample:", m.last_channel_weights.sum(dim=-1))
print("band weights sum to 1 per channel:", m.last_band_weights.sum(dim=-1)[0, :5])
 
