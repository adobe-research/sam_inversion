import sys
import lpips
sys.path.append("src")
from model_utils import *
from img_utils import *
from latent_utils import *
from mask_utils import *
from loss_utils import *
from time import time
from cog import BaseModel, BasePredictor, Input, Path

class Predictor(BasePredictor):
    def setup(self):
        # load the networks
        self.cfg = get_cfg()
        self.models_dict = load_models(self.cfg)
        self.net_sem_seg = load_segmenter(self.cfg.sem_seg_name)
        self.net_lp = lpips.LPIPS(net=self.cfg.lpips_type).cuda()
        print("Warm up: pre-compile modules")
        for k in self.models_dict:
            second_generator_block = getattr(self.models_dict[k][0].synthesis, f'b{8}')
            start = time()
            x = second_generator_block(torch.ones((3,512,4,4)).cuda(),torch.ones((3,3,4,4)).cuda(), torch.ones((3,3,512)).cuda())[0].shape
            print(f"{k}: {time() - start:.2f} seconds")

    def predict(self,
        image_path: Path = Input(description="Image to be classified"),
        image_category: str = Input(choices=["faces", "cars", "cats"]),
        edit_direction: str = Input(choices=['cars - windows_closed', 'cars - model_t', 'cars - blue',
                                             'cars - windows_open', 'cars - snow', 'cars - starwars',
                                             'cars - rims', 'cars - muscle', 'cars - suv', 'cars - black',
                                             'faces - eyebrows_thick', 'faces - nose_small', 'faces - young',
                                             'faces - sad', 'faces - old', 'faces - laughing', 'faces - eyebrows_thin',
                                             'faces - glasses_remove', 'faces - glasses_add', 'faces - nose_big',
                                             'cats - eyes_large', 'cats - nose_pink', 'cats - siamese', 'cats - egyptian',
                                             'cats - surprised', 'cats - eyes_small', 'reconstruct' ]
            , default='reconstruct',
            description="Choose a property to enhance in the image or leave as 'reconstruct' for simple " \
                        "recreation of the image from the GAN latent space"),
        change_norm: float = Input(default=1., description="How much to change the latent representation of the image. Use units in scale of up to 10."),
        thresh: float = Input(default=0.225, description="Trheshold"),

    ) -> Path:
        target_w = 256
        target_h = 192
        if "-" in edit_direction:
            direction_category, edit_direction = edit_direction.split(" - ")
            assert direction_category == image_category, f"Please choose an edit direction in accordance with image category. got {direction_category} with {image_category}"

        net_G, net_e4e, net_S = self.models_dict[image_category]
        d_stats = get_mvg_stats(net_G)
    
        # load the target image
        T = build_t(W=target_w, H=target_h)
        T_full = build_t(W=target_w, H=target_h)
        
        img_pil = Image.open(image_path).convert("RGB")
        img_t = T(img_pil).unsqueeze(0).cuda()
        img_full_t = T_full(img_pil).unsqueeze(0).cuda()

        # segment the target image
        segments = self.net_sem_seg.segment_pil(img_pil)

        # make the invertibility latent map
        d_invmaps = net_S(img_full_t)

        # refine the invertibility map
        d_refined_invmap = refine(d_invmaps, segments, thresh)

        # for non 1:1 images (cars) append zeros to the mask to make it square
        pad_size = None
        if target_h < target_w:
            pad_size = (target_w-target_h)//2
            zero_pad = torch.zeros((1, 1, pad_size, target_w))
            for k in d_refined_invmap:
                d_refined_invmap[k] = torch.cat([zero_pad, d_refined_invmap[k], zero_pad], dim=2).cuda()

        # resize the masks
        d_refined_resized_invmap = resize_binary_masks(d_refined_invmap)

        rec_full, d_latents = find_latent(self.cfg, net_G, net_e4e, img_t, pad_size, d_stats, self.net_lp, d_refined_resized_invmap)

        # save the final outputs
        output_path = "output.png"
        if edit_direction == "reconstruct":
            if image_category == "cars":
                rec_full = rec_full[:, :, 64:-64, :]
            tensor2pil(rec_full.squeeze(0)).save(output_path)
        else:
            d_latents_out = {k: d_latents[k].detach().clone().cuda() for k in d_latents}
            edited_image = edit(self.cfg, image_category, edit_direction, net_G, d_latents_out, d_refined_resized_invmap, change_norm=change_norm)
            edited_image.save(output_path)
        return Path(output_path)


def load_models(cfg):
    res =  dict()
    print("Loding car models")
    res["cars"] = (
            load_generator(cfg.gan_type, "ckpt/stylegan2-car-config-f.pkl"),
            load_encoder("ckpt/e4e_cars_encode.pt"),
            load_invertibility(cfg.latent_names, "ckpt/invertibility_cars_sg2.pt")
        )
    print("Loding face models")
    res["faces"] = (
            load_generator(cfg.gan_type, "ckpt/stylegan2-ffhq-config-f.pkl"),
            load_encoder("ckpt/e4e_ffhq_encode.pt"),
            load_invertibility(cfg.latent_names, "ckpt/invertibility_faces_sg2.pt")
        )
    print("Loding cat models")
    # res["cats"] = (
    #         load_generator(cfg.gan_type, "ckpt/stylegan2-cat-config-f_sg2adapyt.pkl"),
    #         load_encoder("ckpt/e4e_cats.pt"),
    #         load_invertibility(cfg.latent_names, "ckpt/invertibility_lsuncats_sg2.pt")
    #     )
    return res



def get_cfg():
    p = argparse.Namespace()
    p.gan_type="stylegan2"
    p.latent_names="W+,F4,F6,F8,F10" # a comma separated list of candidate latent names to be used
    p.target_W = 256
    p.target_H = 192
    # optimization parameters
    p.num_opt_steps=1001
    p.lr=0.05
    p.lr_rampdown_length=0.25
    p.lr_rampup_length=0.05
    p.beta1=0.9
    p.beta2=0.999
    p.sem_seg_name="hrnet_ade20k"
    p.lpips_type="vgg"
    p.lambda_mse=1
    p.lambda_lpips=1
    p.lambda_f_rec=5
    p.lambda_delta=1e-3
    p.lambda_mvg=1e-8
    p.save_frequency=250
    # editing parameters
    p.generate_edits=True
    return p



def edit(cfg, image_category, edit_name, net_G, d_latents_out, d_refined_resized_invmap, change_norm=1.):
    edit_direciton = np.load(f"edits/{image_category}/csc_{edit_name}.npy")
    l_ims = []
    # sweep over edit multipliers
    ed_dir = torch.tensor(edit_direciton).view(1, net_G.num_ws, 512).cuda() * change_norm
    ed_img = edit_image(net_G.cuda(), d_latents_out, cfg.gan_type, ed_dir, d_refined_resized_invmap)
    if image_category == "cars":
        ed_img = ed_img[:, :, 64:-64, :].clamp(-1, 1)
    l_ims.append(ed_img.detach().cpu())
    # also save as individual compressed images
    return tensor2pil(ed_img.squeeze(0))


def find_latent(cfg, net_G, net_e4e, img_t, pad_size, d_stats, net_lp, d_refined_resized_invmap):
    # initialize the latent codes
    d_latents_init = init_latent(latent_name=cfg.latent_names, G=net_G, G_name=cfg.gan_type, net_e4e=net_e4e, img_t=img_t)
    d_latents = {k: d_latents_init[k].detach().clone() for k in d_latents_init}
    for k in d_latents:
        d_latents[k].requires_grad = True
    # define the optimizer
    optimizer = torch.optim.Adam([d_latents[k] for k in d_latents], lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))
    # optimization loop
    for i in range(cfg.num_opt_steps):
        # learning rate scheduling
        t = i / cfg.num_opt_steps
        lr_ramp = min(1.0, (1.0 - t) / cfg.lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / cfg.lr_rampup_length)
        lr = cfg.lr * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        log_str = f"[(step {i:04d})]: "
        rec_full = custom_forward(G=net_G, d_latents=d_latents, gan_name=cfg.gan_type, latent_name=cfg.latent_names, d_masks=d_refined_resized_invmap)
        # compute the reconstruction losses using smaller 256x256 images
        rec = F.interpolate(rec_full, size=(256, 256), mode='area').clamp(-1, 1)
        # center crop vertically if needed
        if cfg.target_H < cfg.target_W:
            rec = rec[:, :, pad_size:-pad_size, :]
        # image reconstruction losses
        rec_losses = 0.0
        if cfg.lambda_mse > 0:
            rec_losses += F.mse_loss(rec, img_t)*cfg.lambda_mse
        if cfg.lambda_lpips > 0:
            rec_losses += net_lp(rec, img_t).mean()*cfg.lambda_lpips
        log_str += f"rec: {rec_losses:.3f} "
        # latent regularization
        latent_losses = 0.0
        if cfg.lambda_mvg > 0:
            mvg = compute_mvg(d_latents, "W+", d_stats["mean_v"], d_stats["inv_cov_v"])*cfg.lambda_mvg
            latent_losses += mvg
            log_str += f"mvg: {mvg:.3f} "
        if cfg.lambda_delta > 0:
            delta = delta_loss(d_latents["W+"])*cfg.lambda_delta
            latent_losses += delta
            log_str += f"delta: {delta:.3f} "
        if cfg.lambda_f_rec > 0:
            frec = F.mse_loss(d_latents["F4"], d_latents_init["F4"])*cfg.lambda_f_rec
            frec += F.mse_loss(d_latents["F6"], d_latents_init["F6"])*cfg.lambda_f_rec
            frec += F.mse_loss(d_latents["F8"], d_latents_init["F8"])*cfg.lambda_f_rec
            frec += F.mse_loss(d_latents["F10"], d_latents_init["F10"])*cfg.lambda_f_rec
            latent_losses += frec
            log_str += f"frec: {frec:.3f} "
        # update the parameters
        optimizer.zero_grad()
        (rec_losses+latent_losses).backward()
        optimizer.step()
        if i % cfg.save_frequency == 0:
            print(log_str)

    return rec_full, d_latents

if __name__ == '__main__':
    m=Predictor()
    m.setup()
    m.predict(image_path=Path("car.jpg"), image_category="cars", edit_direction="cars - model_t", change_norm=1, thresh=0.225)