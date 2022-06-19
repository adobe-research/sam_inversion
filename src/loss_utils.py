import torch
from torch.nn import functional as F


def compute_mvg(d_latents, latent_name, mean_v, inv_cov_v):
    if latent_name == "W":
        _w = d_latents["W"]
        _v = F.leaky_relu(_w, negative_slope=5.0)
        dv = (_v - mean_v)
        loss = (dv.matmul(inv_cov_v).matmul(dv.T))
        return loss
    elif latent_name == "W+":
        _wp = d_latents["W+"].double()
        _vp = F.leaky_relu(_wp, negative_slope=5.0)
        loss = 0.0
        for idx in range(_vp.shape[1]):
            _v = _vp[:, idx, :]
            dv = (_v - mean_v)
            loss += (dv@inv_cov_v@dv.T)
        return loss.squeeze(0).squeeze(0)


def b_compute_mvg(d_latents, latent_name, mean_v, inv_cov_v):
    if latent_name == "W+":
        _wp = d_latents["W+"].double()
        _vp = F.leaky_relu(_wp, negative_slope=5.0)
        bs = _wp.shape[0]
        inv_cov_v = inv_cov_v.reshape(1, 512, 512).repeat(bs, 1, 1)
        loss = 0.0
        for idx in range(_vp.shape[1]):
            _v = _vp[:, idx, :]
            dv = (_v - mean_v).reshape(-1, 1, 512)
            loss += torch.bmm(torch.bmm(dv, inv_cov_v), torch.transpose(dv, 1, 2)).mean()
        return loss


def delta_loss(latent):
    loss = 0.0
    first_w = latent[:, 0, :]
    for i in range(1, latent.shape[1]):
        delta = latent[:, i, :] - first_w
        delta_loss = torch.norm(delta, 2, dim=1).mean()
        loss += delta_loss
    return loss
