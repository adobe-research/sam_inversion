import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt


d_l2rgb = {
    "W+": [0, 98, 155],
    "F4": [116, 118, 120],
    "F6": [252, 137, 0],
    "F8": [97, 129, 57],
    "F10": [255, 205, 0]
}


def build_t(H=None, W=None, center_crop=False):
    if not center_crop:
        if H is not None and W is not None:
            T = transforms.Compose([
                    transforms.Resize((H, W), interpolation=Image.LANCZOS),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
                )
        else:
            T = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
                )
    else:
        raise ValueError("Do not center crop here")
    return T


def segments2rgb(seg_map, outf):
    plt.imshow(seg_map, cmap="jet")
    plt.axis("off")
    plt.savefig(outf)
    plt.close()


def view_invertibility(image_input, d_invmap, outf):
    NL = len(d_invmap.keys())
    # show first images from the current batch
    f, axs = plt.subplots(1,NL+1,figsize=(5*(NL+1), 5*1))
    axs[0].imshow(tensor2pil(image_input[0])) # target image
    for l_idx in range(NL):
        l_name = list(d_invmap.keys())[l_idx]
        axs[1+l_idx].set_title(f"latent-{l_name}")
        curr_hm = d_invmap[l_name][0].squeeze(0).detach().cpu().numpy()
        axs[1+l_idx].imshow(curr_hm, vmin=0, vmax=1)
    plt.savefig(outf)
    plt.close()


def b_view_invertibility(image_input, d_invmap, outf, rec=None, return_fig=False):
    B = image_input.shape[0]
    NL = len(d_invmap.keys())
    f, axs = plt.subplots(B,NL+2,figsize=(5*(NL+2), 5*B))
    for bidx in range(B):
        axs[bidx,0].imshow(tensor2pil(image_input[bidx]))
        for l_idx in range(NL):
            l_name = list(d_invmap.keys())[l_idx]
            axs[bidx, 2+l_idx].set_title(f"latent-{l_name}")
            curr_hm = d_invmap[l_name][bidx].squeeze(0).detach().cpu().numpy()
            axs[bidx, 2+l_idx].imshow(curr_hm, vmin=0, vmax=1)
        axs[bidx,1].imshow(tensor2pil(rec[bidx]))
    plt.savefig(outf)
    plt.close()
    if return_fig:
        return f


def tensor2pil(var):
    # var shape: (3, H, W)
    var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
    var = ((var + 1) / 2)
    var[var < 0] = 0
    var[var > 1] = 1
    var = var * 255
    return Image.fromarray(var.astype('uint8'))