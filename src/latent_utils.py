import numpy as np
import torch
from torch.nn import functional as F


def get_mvg_stats(G, device=torch.device('cuda')):
    label_c = torch.zeros([1, G.c_dim], device=device)
    buf_v = np.zeros((5_000, 512))
    buf_w = np.zeros((5_000, 512))
    for i in range(5_000):
        _z = torch.randn(1, 512).to(device)
        with torch.no_grad():
            _w = G.mapping(_z, label_c)[:, 0, :]
        _v = F.leaky_relu(_w, negative_slope=5.0)
        buf_w[i, :] = _w.cpu().numpy().reshape(512)
        buf_v[i, :] = _v.cpu().numpy().reshape(512)
    cov_v_np, cov_w_np = np.cov(buf_v.T)+np.eye(512)*1e-8, np.cov(buf_w.T)+np.eye(512)*1e-8
    inv_cov_v_np, inv_cov_w_np = np.linalg.inv(cov_v_np), np.linalg.inv(cov_w_np)
    inv_cov_v, inv_cov_w = torch.tensor(inv_cov_v_np).cuda().double(), torch.tensor(inv_cov_w_np).cuda().double()
    mean_w = torch.tensor(np.mean(buf_w, axis=0)).cuda().float()
    mean_v = F.leaky_relu(mean_w, negative_slope=5.0)
    return {
        "mean_v": mean_v,
        "mean_w": mean_w,
        "inv_cov_v": inv_cov_v,
        "inv_cov_w": inv_cov_w,
    }


def init_latent(latent_name, init_type=None, G=None, G_name="stylegan2", seed=0, device=torch.device('cuda'), net_e4e=None, img_t=None):
    np.random.seed(seed)
    if latent_name == "W+,F4,F6,F8,F10" and G_name == "stylegan2":
        # W+ is initialized with e4e encoder outputs
        with torch.no_grad():
            _wp = net_e4e(img_t, return_latents=True)[1][0:1]
        return {
            "W+": _wp.detach().clone().to(device),
            "F4": torch.zeros((1, 512, 16, 16)).to(device),
            "F6": torch.zeros((1, 512, 32, 32)).to(device),
            "F8": torch.zeros((1, 512, 64, 64)).to(device),
            "F10": torch.zeros((1, 256, 128, 128)).to(device),
        }
    elif latent_name == "W+" and G_name == "stylegan2":
        with torch.no_grad():
            _wp = net_e4e(img_t, return_latents=True)[1][0:1]
        return {
            "W+": _wp.detach().clone().to(device)
        }
    elif latent_name == "F4" and G_name == "stylegan2":
        with torch.no_grad():
            _wp = net_e4e(img_t, return_latents=True)[1][0:1]
        return {
            "W+": _wp.detach().clone().to(device),
            "F4": torch.zeros((1, 512, 16, 16)).to(device),
        }
    elif latent_name == "F6" and G_name == "stylegan2":
        with torch.no_grad():
            _wp = net_e4e(img_t, return_latents=True)[1][0:1]
        return {
            "W+": _wp.detach().clone().to(device),
            "F6": torch.zeros((1, 512, 32, 32)).to(device),
        }
    elif latent_name == "F8" and G_name == "stylegan2":
        with torch.no_grad():
            _wp = net_e4e(img_t, return_latents=True)[1][0:1]
        return {
            "W+": _wp.detach().clone().to(device),
            "F8": torch.zeros((1, 512, 64, 64)).to(device),
        }
    elif latent_name == "F10" and G_name == "stylegan2":
        with torch.no_grad():
            _wp = net_e4e(img_t, return_latents=True)[1][0:1]
        return {
            "W+": _wp.detach().clone().to(device),
            "F10": torch.zeros((1, 512, 128, 128)).to(device),
        }
