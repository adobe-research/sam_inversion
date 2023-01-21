import os
import argparse
import wget
from glob import glob
import numpy as np
import torch
import torchvision
from PIL import Image
from torch.nn import functional as F
import torchvision.transforms as transforms
import lpips
from model_utils import *
from img_utils import *
from latent_utils import *
from misc import *
from mask_utils import *
from loss_utils import *


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--image_folder_path", default="test_images/00004.png")
    p.add_argument("--image_category", default="cars")
    p.add_argument("--target_H", type=int, default=192)
    p.add_argument("--target_W", type=int, default=256)
    p.add_argument("--output_path", default="output")
    p.add_argument("--inv_model_path", default=None)
    p.add_argument("--gan_type", default="stylegan2")
    p.add_argument("--e4e_path", default=None)
    p.add_argument("--gan_weights", default=None)
    p.add_argument("--latent_name", default="W+", help="the latent name to be used")
    # logging & visualizing
    p.add_argument("--save_final_latent", action="store_true")
    p.add_argument("--save_frequency", type=int, default=100)
    # optimization parameters
    p.add_argument("--num_opt_steps", type=int, default=1001)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--lr_rampdown_length", type=float, default=0.25)
    p.add_argument("--lr_rampup_length", type=float, default=0.05)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--sem_seg_name", default="hrnet_ade20k")
    p.add_argument("--lpips_type", default="vgg")
    p.add_argument("--lambda_mse", default=1, type=float)
    p.add_argument("--lambda_lpips", default=1, type=float)
    p.add_argument("--lambda_f_rec", default=5, type=float)
    p.add_argument("--lambda_delta", default=1e-3, type=float)
    p.add_argument("--lambda_mvg", default=1e-8, type=float)
    return p


if __name__ == "__main__":
    args = arguments().parse_args()
    set_random_seed(args.seed)
    # get all target images
    EXTS = ["png", "jpeg", "jpg", "JPEG"]
    l_im_paths = sorted([f for f in glob(os.path.join(args.image_folder_path, "*")) if f.split('.')[-1] in EXTS])
    # set the options based on the image category
    if args.image_category == "cars":
        if args.gan_weights is None:
            args.gan_weights = "ckpt/stylegan2-car-config-f.pkl"
            # download if not present automatically
            if not os.path.exists(args.gan_weights):
                url = "https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/stylegan2-car-config-f.pkl"
                wget.download(url, args.gan_weights)
        if args.e4e_path is None:
            args.e4e_path = "ckpt/e4e_cars_encode.pt"
            if not os.path.exists(args.e4e_path):
                url = "https://www.cs.cmu.edu/~SAMInversion/ckpt/e4e_cars_encode.pt"
                wget.download(url, args.e4e_path)

    elif args.image_category == "faces":
        if args.gan_weights is None:
            args.gan_weights = "ckpt/stylegan2-ffhq-config-f.pkl"
            if not os.path.exists(args.gan_weights):
                url = "https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/stylegan2-ffhq-config-f.pkl"
                wget.download(url, args.gan_weights)
        if args.e4e_path is None:
            args.e4e_path = "ckpt/e4e_ffhq_encode.pt"
            if not os.path.exists(args.e4e_path):
                url = "https://www.cs.cmu.edu/~SAMInversion/ckpt/e4e_ffhq_encode.pt"
                wget.download(url, args.e4e_path)

    elif args.image_category == "cats":
        if args.gan_weights is None:
            args.gan_weights = "ckpt/stylegan2-cat-config-f_sg2adapyt.pkl"
            if not os.path.exists(args.gan_weights):
                url = "https://www.cs.cmu.edu/~SAMInversion/ckpt/stylegan2-cat-config-f_sg2adapyt.pkl"
                wget.download(url, args.gan_weights)
        if args.e4e_path is None:
            args.e4e_path = "ckpt/e4e_lsuncats_encode.pt"
            if not os.path.exists(args.e4e_path):
                url = "https://www.cs.cmu.edu/~SAMInversion/ckpt/e4e_lsuncats_encode.pt"
                wget.download(url, args.e4e_path)

    # load the networks
    net_G = load_generator(args.gan_type, args.gan_weights)
    net_e4e = load_encoder(args.e4e_path)
    d_stats = get_mvg_stats(net_G)
    net_lp = lpips.LPIPS(net=args.lpips_type).cuda()

    for im_path in l_im_paths:
        # make the output directories
        bname = os.path.basename(im_path)
        for ext in EXTS:
            bname = bname.replace(f".{ext}", "")

        os.makedirs(os.path.join(args.output_path, bname, "final"), exist_ok=True)

        # load the target image
        T = build_t(W=args.target_W, H=args.target_H)
        T_full = build_t(W=None, H=None)
        img_pil = Image.open(im_path).convert("RGB")
        img_t = T(img_pil).unsqueeze(0).cuda()
        img_full_t = T_full(img_pil).unsqueeze(0).cuda()
        if args.image_category == "cars":
            img_full_t = img_full_t[:, :, 64:-64, :]

        # do not rerun the inversion if the inverted latent already exists
        if not os.path.exists(os.path.join(args.output_path, bname, "final", "inverted_latent.pt")):
            # initialize the latent codes
            d_latents_init = init_latent(latent_name=args.latent_name, G=net_G, G_name=args.gan_type, net_e4e=net_e4e, img_t=img_t)
            d_latents = {k: d_latents_init[k].detach().clone() for k in d_latents_init}
            for k in d_latents:
                d_latents[k].requires_grad=True
            # define the optimizer
            optimizer = torch.optim.Adam([d_latents[k] for k in d_latents], lr=args.lr, betas=(args.beta1, args.beta2))
            # optimization loop
            for i in range(args.num_opt_steps):
                # learning rate scheduling
                t = i / args.num_opt_steps
                lr_ramp = min(1.0, (1.0 - t) / args.lr_rampdown_length)
                lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
                lr_ramp = lr_ramp * min(1.0, t / args.lr_rampup_length)
                lr = args.lr * lr_ramp
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                log_str = f"[{bname} (step {i:04d})]: "
                rec_full = custom_forward(G=net_G, d_latents=d_latents, gan_name=args.gan_type, latent_name=args.latent_name)
                # compute the reconstruction losses using smaller 256x256 images
                rec = F.interpolate(rec_full, size=(256, 256), mode='area').clamp(-1, 1)
                # center crop vertically if needed
                if args.image_category == "cars":
                    rec = rec[:, :, 32:-32, :]
                # image reconstruction losses
                rec_losses = 0.0
                if args.lambda_mse > 0:
                    rec_losses += F.mse_loss(rec, img_t)*args.lambda_mse
                if args.lambda_lpips > 0:
                    rec_losses += net_lp(rec, img_t).mean()*args.lambda_lpips
                log_str += f"rec: {rec_losses:.3f} "
                # latent regularization
                latent_losses = 0.0
                if args.lambda_mvg > 0 and "W+" in d_latents:
                    mvg = compute_mvg(d_latents, "W+", d_stats["mean_v"], d_stats["inv_cov_v"])*args.lambda_mvg
                    latent_losses += mvg
                    log_str += f"mvg: {mvg:.3f} "
                if args.lambda_delta > 0 and "W+" in d_latents:
                    delta = delta_loss(d_latents["W+"])*args.lambda_delta
                    latent_losses += delta
                    log_str += f"delta: {delta:.3f} "
                if args.lambda_f_rec > 0:
                    frec = torch.tensor(0.0).cuda()
                    if "F4" in d_latents:
                        frec += F.mse_loss(d_latents["F4"], d_latents_init["F4"])*args.lambda_f_rec
                    if "F6" in d_latents:
                        frec += F.mse_loss(d_latents["F6"], d_latents_init["F6"])*args.lambda_f_rec
                    if "F8" in d_latents:
                        frec += F.mse_loss(d_latents["F8"], d_latents_init["F8"])*args.lambda_f_rec
                    if "F10" in d_latents:
                        frec += F.mse_loss(d_latents["F10"], d_latents_init["F10"])*args.lambda_f_rec
                    latent_losses += frec
                    log_str += f"frec: {frec:.3f} "
                # update the parameters
                optimizer.zero_grad()
                (rec_losses+latent_losses).backward()
                optimizer.step()
                if i % args.save_frequency == 0:
                    print(log_str)

            # save the final outputs
            outf = os.path.join(args.output_path, bname, "final", "reconstruction.png")
            if args.image_category == "cars":
                rec_full = rec_full[:, :, 64:-64, :]
            tensor2pil(rec_full.squeeze(0)).save(outf)
            outf = os.path.join(args.output_path, bname, "final", "inverted_latent.pt")
            d_latents_out = {k: d_latents[k].detach().clone().cuda() for k in d_latents}
            torch.save(d_latents_out, outf)
            # save the target image as well
            outf = os.path.join(args.output_path, bname, "final", "input.png")
            tensor2pil(img_full_t.squeeze(0)).save(outf)
        else:
            d_latents_out = torch.load(os.path.join(args.output_path, bname, "final", "inverted_latent.pt"))
            for k in d_latents_out: d_latents_out[k] = d_latents_out[k].cuda()
