import numpy as np
import torch
from PIL import Image


def refine(d_invmaps, seg_map, tau):
    H, W = seg_map.shape
    refined = np.zeros((H, W))
    idx2latent = ["F10", "F8", "F6", "F4", "W+"]
    latent2idx = {n: i for i, n in enumerate(idx2latent)}
    # iterate through each segment index
    for v in np.unique(seg_map):
        curr_segment = (seg_map == v)
        latent_for_this_segment = "F10"
        for l_name in idx2latent:
            # check the average inv value inside the segment
            avg_val = (d_invmaps[l_name].detach().cpu()*curr_segment).sum() / curr_segment.sum()
            if avg_val <= tau:
                latent_for_this_segment = l_name
        refined[curr_segment] = latent2idx[latent_for_this_segment]
    # expand the latent map into individual binary masks
    d_refined = {name: torch.tensor((refined == idx)[None, None]) for idx, name in enumerate(idx2latent)}
    return d_refined


def b_refine(d_invmaps, seg_map, tau):
    B, H, W = seg_map.shape
    refined = np.zeros((B, H, W))
    idx2latent = ["F10", "F8", "F6", "F4", "W+"]
    latent2idx = {n: i for i, n in enumerate(idx2latent)}
    # iterate through each segment index
    for v in np.unique(seg_map):
        # do for each image in batch separately
        for bidx in range(B):
            curr_segment = (seg_map[bidx] == v)
            latent_for_this_segment = "F10"
            for l_name in idx2latent:
                # check the average inv value inside the segment
                avg_val = (d_invmaps[l_name][bidx].detach().cpu()*curr_segment).sum() / curr_segment.sum()
                if avg_val <= tau:
                    latent_for_this_segment = l_name
            refined[bidx, curr_segment] = latent2idx[latent_for_this_segment]
    # expand the latent map into individual binary masks
    d_refined = {name: torch.tensor((refined == idx)).reshape(B, 1, H, W).cuda() for idx, name in enumerate(idx2latent)}
    return d_refined


def resize_single_channel(x_np, S, k=Image.LANCZOS):
    s1, s2 = S
    img = Image.fromarray(x_np.astype(np.float32), mode='F')
    img = img.resize(S, resample=k)
    return np.asarray(img).reshape(s2, s1).clip(0, x_np.max())


def resize_binary_masks(d_refined_invmap):
    d_out = {
        "W+": d_refined_invmap["W+"][0, 0].detach().cpu().numpy(),
        "F4": resize_single_channel(d_refined_invmap["F4"][0, 0].detach().cpu().numpy(), (16, 16), Image.LANCZOS),
        "F6": resize_single_channel(d_refined_invmap["F6"][0, 0].detach().cpu().numpy(), (32, 32), Image.LANCZOS),
        "F8": resize_single_channel(d_refined_invmap["F8"][0, 0].detach().cpu().numpy(), (64, 64), Image.LANCZOS),
        "F10": resize_single_channel(d_refined_invmap["F10"][0, 0].detach().cpu().numpy(), (128, 128), Image.LANCZOS),
    }
    d_out = {k: torch.tensor(d_out[k][None, None]).cuda() for k in d_out.keys()}
    return d_out
