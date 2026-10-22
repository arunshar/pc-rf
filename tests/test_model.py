"""Unit tests for the network components (model.py)."""
import torch

from pcrf import model as m


def test_sinusoidal_embedding_shape():
    e = m.sinusoidal_embedding(torch.rand(4), 16)
    assert e.shape == (4, 16)


def test_film_preserves_shape():
    film = m.FiLM(cond_dim=8, feat_dim=6)
    x = torch.randn(2, 6, 5, 5)
    c = torch.randn(2, 8)
    assert film(x, c).shape == x.shape


def test_satellite_encoder_shape():
    enc = m.SatelliteEncoder(in_ch=4, embed_ch=16)
    assert enc(torch.randn(2, 4, 32, 32)).shape == (2, 16, 32, 32)


def test_unet_forward_shape_and_gradients():
    net = m.UNet(in_ch=3, cond_ch=7, base=16, ch_mults=(1, 2), attn_at=())
    x = torch.randn(2, 3, 32, 32)
    t = torch.rand(2)
    cond = torch.randn(2, 7, 32, 32)
    v = net(x, t, cond)
    assert v.shape == x.shape
    v.sum().backward()
    g = sum(par.grad.abs().sum() for par in net.parameters() if par.grad is not None)
    assert g > 0


def test_unet_with_attention_runs():
    net = m.UNet(in_ch=3, cond_ch=7, base=16, ch_mults=(1, 2), attn_at=(1,))
    x = torch.randn(1, 3, 32, 32)
    t = torch.zeros(1)
    cond = torch.randn(1, 7, 32, 32)
    assert net(x, t, cond).shape == x.shape
