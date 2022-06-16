import os
import sys
import argparse
import matplotlib.pyplot as plt
import lpips
sys.path.append("./src/nets/pytorch-deeplab-xception")
from modeling.deeplab import DeepLab
from utils.lr_scheduler import LR_Scheduler
from misc import *
from dataloader_utils import *
from model_utils import *
from img_utils import *


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--dataset_folder_train", required=True)
    p.add_argument("--dataset_folder_val", required=True)
    p.add_argument("--output_folder", required=True)
    p.add_argument("--train_val_split", type=float, default=0.8)
    p.add_argument('--base-size', type=int, default=256)
    p.add_argument('--crop-size', type=int, default=256)
    
    # latent space details
    p.add_argument("--latent_space_name", default="W+,F4,F6,F8,F10")
    
    # optimization parameters
    p.add_argument('--backbone', type=str, default='resnet', choices=['resnet', 'xception', 'drn', 'mobilenet'])
    p.add_argument('--out-stride', type=int, default=16)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--start_epoch', type=int, default=0)
    p.add_argument('--batch-size', type=int, required=True)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--test-batch-size', type=int, default=None)
    p.add_argument("--lr", type=float, default=0.0001)
    p.add_argument('--lr-scheduler', type=str, default='poly', choices=['poly', 'step', 'cos'])
    p.add_argument('--momentum', type=float, default=0.9)
    p.add_argument('--weight-decay', type=float, default=5e-4)
    
    p.add_argument("--log_frequency", type=int, default=100)
    p.add_argument("--viz_frequency", type=int, default=100)
    p.add_argument('--eval-interval', type=int, default=1)
    p.add_argument('--ckpt_interval', type=int, default=5)

    p.add_argument('--loss-type', type=str, default='ce', choices=['ce', 'focal'])

    p.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    
    p.add_argument("--lpips_type", default="vgg", choices=["vgg", "alex"])
    return p


class Trainer():
    def __init__(self, args):
        # make the output folder
        os.makedirs(os.path.join(args.output_folder, "images"), exist_ok=True)
        os.makedirs(os.path.join(args.output_folder, "ckpt"), exist_ok=True)
        
        # define the corresponding networks
        self.model = DeepLab(num_classes=8, backbone=args.backbone, output_stride=args.out_stride,
                        sync_bn=args.sync_bn, freeze_bn=False)
        self.model = torch.nn.DataParallel(self.model).cuda()
        self.net_lp = torch.nn.DataParallel(lpips.LPIPS(net=args.lpips_type, spatial=True)).cuda()

        # define a head for each latent layer to be used
        self.latent_names = args.latent_names = args.latent_space_name.split(",")
        self.d_heads = {}
        for n in self.latent_names:
            self.d_heads[n] = torch.nn.DataParallel(layer_head()).cuda()
        
        # define the dataloaders
        self.train_loader, self.val_loader = make_invertibility_loader(args)

        # define the optimizer
        params = [
            {'params': self.model.module.get_1x_lr_params(), 'lr': args.lr},
            {'params': self.model.module.get_10x_lr_params(), 'lr': args.lr*10},
        ]
        for n in self.d_heads:
            params.append(
                {"params": self.d_heads[n].parameters(), "lr": args.lr}
            )
        self.optimizer = torch.optim.SGD(params, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=False)

        self.criterion = torch.nn.MSELoss()
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr, args.epochs, len(self.train_loader))
        self.best_pred = 0
        self.f_log = os.path.join(args.output_folder, "log.txt")
        self.args = args

    def training(self, epoch):
        train_loss = 0.0
        # set everything to train mode
        self.model.train()
        for n in self.latent_names:
            self.d_heads[n].train()
        for i, sample in enumerate(self.train_loader):
            image_input = sample["image_target"].to("cuda")
            d_latent_recs = {}
            for n in self.latent_names:
                d_latent_recs[n] = sample[n].to("cuda")
            # get the ground truth LPIPS map
            d_target_hm = {}
            with torch.no_grad():
                for n in self.latent_names:
                    _rec = d_latent_recs[n]
                    d_target_hm[n] = self.net_lp(d_latent_recs[n], image_input)
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            # get intermediate prediction
            _out = self.model(image_input)
            # get prediction for each latent head
            d_out = {n: self.d_heads[n](_out) for n in self.latent_names}
            out_cmb = torch.cat([d_out[n] for n in self.latent_names], dim=0)
            target_cmb = torch.cat([d_target_hm[n] for n in self.latent_names], dim=0)
            loss = self.criterion(out_cmb, target_cmb)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()

            if (i)%args.log_frequency==0:
                curr_train_loss = train_loss/(i+1)
                log_str = f"[EP:{epoch:02d}, step-{i:04d}] Train loss: {(curr_train_loss):.3f}"
                print_and_save(log_str, self.f_log)
                
            if (i)%args.viz_frequency==0:
                outf = os.path.join(args.output_folder, f"images/ep{epoch:03d}_{i}.jpg")
                NL = len(self.latent_names)
                BS = image_input.shape[0]
                # show first images from the current batch
                f, axs = plt.subplots(3,NL+1,figsize=(5*(NL+1), 5*3))
                axs[0,0].imshow(tensor2pil(image_input[0])) # target image
                axs[1,0].imshow(tensor2pil(image_input[0])) # target image
                axs[2,0].imshow(tensor2pil(image_input[0])) # target image
                vmin = d_target_hm[n][0].min()#min(out_cmb[:,0].min(), target_cmb[:,0].min())
                vmax = d_target_hm[n][0].max()#max(out_cmb[:,0].max(), target_cmb[:,0].max())
                for _, n in enumerate(self.latent_names):
                    axs[0,1+_].set_title(f"latent-{n}")
                    axs[0,1+_].imshow(tensor2pil(d_latent_recs[n][0])) # target image
                    curr_gt = d_target_hm[n][0].squeeze(0).detach().cpu().numpy()
                    axs[1,1+_].imshow(curr_gt, cmap="Blues",vmin=vmin, vmax=vmax)
                    curr_pred = d_out[n][0].squeeze(0).detach().cpu().numpy()
                    axs[2,1+_].imshow(curr_pred, cmap="Blues",vmin=vmin, vmax=vmax)
                plt.savefig(outf)
                plt.close(f)

    def validation(self, epoch):
        val_loss = 0.0
        # set everything to eval mode
        self.model.eval()
        for n in self.latent_names:
            self.d_heads[n].eval()
        for i, sample in enumerate(self.val_loader):
            image_input = sample["image_target"].to("cuda")
            d_latent_recs = {}
            for n in self.latent_names:
                d_latent_recs[n] = sample[n].to("cuda")

            # get the ground truth LPIPS map
            d_target_hm = {}
            with torch.no_grad():
                for n in self.latent_names:
                    _rec = d_latent_recs[n]
                    d_target_hm[n] = self.net_lp(d_latent_recs[n], image_input)
            # get intermediate prediction
            _out = self.model(image_input)
            # get prediction for each latent head
            d_out = {n: self.d_heads[n](_out) for n in self.latent_names}
            out_cmb = torch.cat([d_out[n] for n in self.latent_names], dim=0)
            target_cmb = torch.cat([d_target_hm[n] for n in self.latent_names], dim=0)
            loss = self.criterion(out_cmb, target_cmb)
            val_loss += loss.item()

            if (i)%args.viz_frequency==0:
                outf = os.path.join(args.output_folder, f"images/VAL_ep{epoch:03d}_{i}.jpg")
                NL = len(self.latent_names)
                BS = image_input.shape[0]
                # show first images from the current batch
                f, axs = plt.subplots(3,NL+1,figsize=(5*(NL+1), 5*3))
                axs[0,0].imshow(tensor2pil(image_input[0])) # target image
                axs[1,0].imshow(tensor2pil(image_input[0])) # target image
                axs[2,0].imshow(tensor2pil(image_input[0])) # target image
                for _, n in enumerate(self.latent_names):
                    axs[0,1+_].set_title(f"latent-{n}")
                    axs[0,1+_].imshow(tensor2pil(d_latent_recs[n][0]))
                    curr_gt = d_target_hm[n][0].squeeze(0).detach().cpu().numpy()
                    axs[1,1+_].imshow(curr_gt, cmap="Blues",vmin=0, vmax=1)
                    curr_pred = d_out[n][0].squeeze(0).detach().cpu().numpy()
                    axs[2,1+_].imshow(curr_pred, cmap="Blues",vmin=0, vmax=1)
                plt.savefig(outf)
                plt.close(f)
        log_str = f"[EP:{epoch:02d} => Val loss: {(val_loss/(len(self.val_loader))):.3f}\n\n"
        print_and_save(log_str, self.f_log)
        return val_loss/len(self.val_loader)


if __name__ == "__main__":
    args = arguments().parse_args()
    args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
    # use synced batch norm when training on multiple GPUs
    if len(args.gpu_ids) > 1:
        args.sync_bn = True
    else:
        args.sync_bn = False
    print(args)
    # set random seed
    set_random_seed(args.seed)
    # main training loop
    T = Trainer(args)
    for ep in range(0, args.epochs):
        T.training(ep)
        if ep%args.eval_interval==(args.eval_interval-1):
            val_loss = T.validation(ep)
        if ep%args.ckpt_interval==(args.ckpt_interval-1):
            outf = os.path.join(args.output_folder, f"ckpt/{ep:03d}_main.pt")
            sd = T.model.state_dict()
            torch.save(sd, outf)
            for n in T.latent_names:
                outf = os.path.join(args.output_folder, f"ckpt/{ep:03d}_head_{n}.pt")
                sd = T.d_heads[n].state_dict()
                torch.save(sd, outf)