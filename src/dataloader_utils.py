import os
import random
import torch
import torchvision
from PIL import Image, ImageFilter
from glob import glob

def make_T(flip_horizontal=True, random_gaussian=True, H=256, W=256):
    def fn(x1, d_x2):
        d_out = {}
        # resize to 256x256
        x1 = x1.resize((W, H), Image.LANCZOS)
        for k in d_x2:
            d_out[k] = d_x2[k].resize((W, H), Image.LANCZOS)
        # randomly flip
        if flip_horizontal and random.random() > 0.5:
            x1 = x1.transpose(Image.FLIP_LEFT_RIGHT)
            for k in d_x2:
                d_out[k] = d_out[k].transpose(Image.FLIP_LEFT_RIGHT)
        # randomly apply gaussian blur
        if random_gaussian and random.random() > 0.5:
            rad = random.random()
            x1 = x1.filter(ImageFilter.GaussianBlur(radius=rad))
            for k in d_x2:
                d_out[k] = d_out[k].filter(ImageFilter.GaussianBlur(radius=rad))
        # convert to tensor
        x1 = torchvision.transforms.ToTensor()(x1)
        for k in d_x2:
            d_out[k] = torchvision.transforms.ToTensor()(d_out[k])
        # normalize
        d_out["image_target"] = x1 = torchvision.transforms.Normalize(0.5, 0.5)(x1)
        for k in d_x2:
            d_out[k] = torchvision.transforms.Normalize(0.5, 0.5)(d_out[k])
        return d_out
    return fn


class InvDS(torch.utils.data.Dataset):
    def __init__(self, f_impaths, T, l_names):
        # get list of target files using the W+ folder
        self.f_impaths = f_impaths
        self.T = T
        self.l_names = l_names

    def __len__(self):
        return len(self.f_impaths)

    def __getitem__(self, idx):
        path_target = self.f_impaths[idx].replace("reconstruction.png", "input.png")
        img_pil = Image.open(path_target)
        d_rec_pil = {}
        for n in self.l_names:
            d_rec_pil[n] = Image.open(self.f_impaths[idx].replace("/W+/", f"/{n}/"))
        return self.T(img_pil, d_rec_pil)


def make_invertibility_loader(args):
    # (train) get all reconstructions for W+
    l_wp_recs_train = sorted(glob(os.path.join(args.dataset_folder_train, "W+/*/final/reconstruction.png")))
    ds_train = InvDS(l_wp_recs_train, make_T(flip_horizontal=True, random_gaussian=True), l_names=args.latent_names)
    # (val) get all reconstructions for W+
    l_wp_recs_val = sorted(glob(os.path.join(args.dataset_folder_train, "W+/*/final/reconstruction.png")))
    ds_val = InvDS(l_wp_recs_val, make_T(flip_horizontal=False, random_gaussian=False), l_names=args.latent_names)
    train_loader = torch.utils.data.DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    return train_loader, val_loader
