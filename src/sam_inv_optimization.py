import os
import argparse
from glob import glob
import wget
import gdown
import numpy as np
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
    p.add_argument("--image_path", default="test_images/00004.png")
    p.add_argument("--image_category", default="cars")
    p.add_argument("--target_H", type=int, default=192)
    p.add_argument("--target_W", type=int, default=256)
    p.add_argument("--output_path", default="output")
    p.add_argument("--inv_model_path", default=None)
    p.add_argument("--gan_type", default="stylegan2")
    p.add_argument("--e4e_path", default=None)
    p.add_argument("--gan_weights", default=None)
    p.add_argument("--latent_names", default="W+,F4,F6,F8,F10", help="a comma separated list of candidate latent names to be used")
    p.add_argument("--threshold", type=float, default=0.225)
    p.add_argument("--sweep_thresholds", action="store_true")
    # logging & visualizing
    p.add_argument("--save_final_latent", action="store_true")
    p.add_argument("--save_intermediate", action="store_true")
    p.add_argument("--save_frequency", type=int, default=250)
    
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
    # editing parameters
    p.add_argument("--generate_edits", action="store_true")
    p.add_argument("--edits_folder", default="edits/cars")
    return p


if __name__=="__main__":
    args = arguments().parse_args()
    set_random_seed(args.seed)
    # make the output directories
    bname = os.path.basename(args.image_path).replace(".png","")
    os.makedirs(os.path.join(args.output_path, bname, "intermediate"), exist_ok=True)
    os.makedirs(os.path.join(args.output_path, bname, "final"), exist_ok=True)
    os.makedirs("ckpt", exist_ok=True)

    # set some of the options based on the image category
    if args.image_category=="cars":
        if args.gan_weights is None:
            args.gan_weights = "ckpt/stylegan2-car-config-f.pkl"
            # download if not present automatically
            if not os.path.exists(args.gan_weights):
                url="https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/stylegan2-car-config-f.pkl"
                wget.download(url, args.gan_weights)
        if args.e4e_path is None:
            args.e4e_path = "ckpt/e4e_cars_encode.pt"
            if not os.path.exists(args.e4e_path):
                url="https://www.cs.cmu.edu/~SAMInversion/ckpt/e4e_cars_encode.pt"
                wget.download(url, args.e4e_path)
        if args.inv_model_path is None:
            args.inv_model_path = "ckpt/invertibility_cars_sg2.pt"
            if not os.path.exists(args.inv_model_path):
                url="https://www.cs.cmu.edu/~SAMInversion/ckpt/invertibility_cars_sg2.pt"
                wget.download(url, args.inv_model_path)
        max_tau_value = 0.401
        min_tau_value = 0.1

    elif args.image_category=="faces":
        if args.gan_weights is None:
            args.gan_weights = "ckpt/stylegan2-ffhq-config-f.pkl"
            if not os.path.exists(args.gan_weights):
                url="https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/stylegan2-ffhq-config-f.pkl"
                wget.download(url, args.gan_weights)
        if args.e4e_path is None:
            args.e4e_path = "ckpt/e4e_ffhq_encode.pt"
            if not os.path.exists(args.e4e_path):
                url="https://www.cs.cmu.edu/~SAMInversion/ckpt/e4e_ffhq_encode.pt"
                wget.download(url, args.e4e_path)
        if args.inv_model_path is None:
            args.inv_model_path = "ckpt/invertibility_faces_sg2.pt"
            if not os.path.exists(args.inv_model_path):
                url="https://www.cs.cmu.edu/~SAMInversion/ckpt/invertibility_faces_sg2.pt"
                wget.download(url, args.inv_model_path)
        args.sem_seg_name = "face_parser_fused"
        args.edits_folder = "edits/faces"
        max_tau_value = 0.401
        min_tau_value = 0.1

    elif args.image_category=="cats":
        if args.gan_weights is None:
            args.gan_weights = "ckpt/stylegan2-cat-config-f_sg2adapyt.pkl"
            if not os.path.exists(args.gan_weights):
                url="https://www.cs.cmu.edu/~SAMInversion/ckpt/stylegan2-ffhq-config-f.pkl"
                wget.download(url, args.gan_weights)
        if args.e4e_path is None:
            args.e4e_path = "ckpt/e4e_lsuncats_encode.pt"
            if not os.path.exists(args.e4e_path):
                url="https://www.cs.cmu.edu/~SAMInversion/ckpt/e4e_lsuncats_encode.pt"
                wget.download(url, args.e4e_path)
        if args.inv_model_path is None:
            args.inv_model_path = "ckpt/invertibility_lsuncats_sg2.pt"
            if not os.path.exists(args.inv_model_path):
                url="https://www.cs.cmu.edu/~SAMInversion/ckpt/invertibility_lsuncats_sg2.pt"
                wget.download(url, args.inv_model_path)
        args.sem_seg_name = "detectron_coco"
        args.edits_folder = "edits/cats"
        max_tau_value = 0.451
        min_tau_value = 0.20
    
    # load the networks
    net_G = load_generator(args.gan_type, args.gan_weights)
    net_sem_seg = load_segmenter(args.sem_seg_name)
    net_e4e = load_encoder(args.e4e_path)
    d_stats = get_mvg_stats(net_G)
    net_S = load_invertibility(args.latent_names, args.inv_model_path)
    net_lp = lpips.LPIPS(net=args.lpips_type).cuda()

    # load the target image
    T = build_t(W=args.target_W,H=args.target_H)
    T_full = build_t(W=None,H=None)
    img_pil = Image.open(args.image_path).convert("RGB")
    img_t = T(img_pil).unsqueeze(0).cuda()
    img_full_t = T_full(img_pil).unsqueeze(0).cuda()

    # segment the target image
    segments = net_sem_seg.segment_pil(img_pil)
    if args.save_intermediate:
        outf = os.path.join(args.output_path, bname, f"0_segments.png")
        segments2rgb(segments, outf)

    # make the invertibility latent map
    d_invmaps = net_S(img_full_t)
    if args.save_intermediate:
        outf = os.path.join(args.output_path, bname, f"1_invmap_raw.png")
        view_invertibility(img_t, d_invmaps, outf)

    if not args.sweep_thresholds:
        l_thresholds = [args.threshold]
    else:
        l_thresholds = np.arange(min_tau_value, max_tau_value, 0.025).tolist() + [1.0]
    
    for thresh  in l_thresholds:
        # refine the invertibility map
        d_refined_invmap = refine(d_invmaps, segments, thresh)
        if args.save_intermediate:
            outf = os.path.join(args.output_path, bname, f"2_invmap_refined_T{thresh:.3f}.png")
            view_invertibility(img_t, d_refined_invmap, outf)
            # colorized invertibility map
            inv_map_rgb = np.zeros((img_full_t.shape[2], img_full_t.shape[3], 3))
            for ln in d_invmaps.keys():
                inv_map_rgb[d_refined_invmap[ln][0,0]] = d_l2rgb[ln]
            outf = os.path.join(args.output_path, bname, f"3_invmap_colorized_T{thresh:.3f}.png")
            Image.fromarray(inv_map_rgb.astype(np.uint8)).save(outf)
        
        # for non 1:1 images (cars) append zeros to the mask to make it square
        if args.target_H<args.target_W:
            pad_size = (args.target_W-args.target_H)//2
            zero_pad = torch.zeros((1,1,pad_size,args.target_W))
            for k in d_refined_invmap:
                d_refined_invmap[k] = torch.cat([zero_pad, d_refined_invmap[k], zero_pad], dim=2).cuda()

        # resize the masks
        d_refined_resized_invmap = resize_binary_masks(d_refined_invmap)
        if args.save_intermediate:
            outf = os.path.join(args.output_path, bname, f"4_invmap_refined_resized_T{thresh:.3f}.png")
            view_invertibility(img_t, d_refined_resized_invmap, outf)
        
        # do not rerun the inversion if the inverted latent already exists
        if not os.path.exists(os.path.join(args.output_path, bname, "final", f"inverted_latents_T{thresh:.3f}.pt")):
            # initialize the latent codes
            d_latents_init = init_latent(latent_name=args.latent_names, G=net_G, G_name=args.gan_type, net_e4e=net_e4e, img_t=img_t)
            d_latents = {k:d_latents_init[k].detach().clone() for k in d_latents_init}
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
                rec_full = custom_forward(G=net_G, d_latents=d_latents, gan_name=args.gan_type, latent_name=args.latent_names, d_masks=d_refined_resized_invmap)
                # compute the reconstruction losses using smaller 256x256 images
                rec = F.interpolate(rec_full, size=(256, 256), mode='area').clamp(-1,1)
                # center crop vertically if needed
                if args.target_H<args.target_W:
                    rec = rec[:,:,pad_size:-pad_size,:]
                # image reconstruction losses
                rec_losses = 0.0
                if args.lambda_mse>0: 
                    rec_losses += F.mse_loss(rec, img_t)*args.lambda_mse
                if args.lambda_lpips>0: 
                    rec_losses += net_lp(rec, img_t).mean()*args.lambda_lpips
                log_str += f"rec: {rec_losses:.3f} "
                # latent regularization
                latent_losses = 0.0
                if args.lambda_mvg>0:
                    mvg = compute_mvg(d_latents, "W+", d_stats["mean_v"], d_stats["inv_cov_v"])*args.lambda_mvg
                    latent_losses += mvg
                    log_str += f"mvg: {mvg:.3f} "
                if args.lambda_delta>0:
                    delta = delta_loss(d_latents["W+"])*args.lambda_delta
                    latent_losses += delta
                    log_str += f"delta: {delta:.3f} "
                if args.lambda_f_rec > 0:
                    frec = F.mse_loss(d_latents["F4"], d_latents_init["F4"])*args.lambda_f_rec
                    frec += F.mse_loss(d_latents["F6"], d_latents_init["F6"])*args.lambda_f_rec
                    frec += F.mse_loss(d_latents["F8"], d_latents_init["F8"])*args.lambda_f_rec
                    frec += F.mse_loss(d_latents["F10"], d_latents_init["F10"])*args.lambda_f_rec
                    latent_losses +=  frec
                    log_str += f"frec: {frec:.3f} "
                # update the parameters
                optimizer.zero_grad()
                (rec_losses+latent_losses).backward()
                optimizer.step()
                if i%args.save_frequency==0:
                    print(log_str)
                    if args.save_intermediate:
                        outf = os.path.join(args.output_path, bname, "intermediate", f"step_T{thresh:.3f}_{i:04d}.png")
                        tensor2pil(rec.squeeze(0)).save(outf)

            # save the final outputs
            outf = os.path.join(args.output_path, bname, "final", f"reconstruction_T{thresh:.3f}.png")
            if args.image_category=="cars":
                rec_full = rec_full[:,:,64:-64,:]
            tensor2pil(rec_full.squeeze(0)).save(outf)
            outf = os.path.join(args.output_path, bname, "final", f"inverted_latents_T{thresh:.3f}.pt")
            d_latents_out = {k:d_latents[k].detach().clone().cuda() for k in d_latents}
            torch.save(d_latents_out, outf)
        else:
            d_latents_out = torch.load(os.path.join(args.output_path, bname, "final", f"inverted_latents_T{thresh:.3f}.pt"))
            for k in d_latents_out: d_latents_out[k] = d_latents_out[k].cuda()

        if args.generate_edits:
            os.makedirs(os.path.join(args.output_path, bname, "edits"), exist_ok=True)
            l_edits = glob(os.path.join(args.edits_folder, "*.npy"))
            for ed in l_edits:
                ed_name = os.path.basename(ed).replace(".npy","")
                os.makedirs(os.path.join(args.output_path, bname, "edits", ed_name), exist_ok=True)
                l_ims = []
                # sweep over edit multipliers
                for ed_mul in [0, 1, 2, 3]:
                    ed_dir = torch.tensor(np.load(ed)).view(1,net_G.num_ws,512).cuda()*ed_mul
                    ed_img = edit_image(net_G.cuda(), d_latents_out, args.gan_type, ed_dir, d_refined_resized_invmap)
                    if args.image_category=="cars":
                        ed_img = ed_img[:,:,64:-64,:].clamp(-1,1)
                    l_ims.append(ed_img.detach().cpu())
                    # also save as individual compressed images
                    outf = os.path.join(args.output_path, bname, "edits", ed_name, f"T_{thresh:.3f}_mul{ed_mul}.jpg")
                    tensor2pil(ed_img.squeeze(0)).save(outf)
                outf = os.path.join(args.output_path, bname, "edits", ed_name, f"T_{thresh:.3f}.png")
                transforms.ToPILImage()(torchvision.utils.make_grid(torch.cat(l_ims), normalize=True)).save(outf)