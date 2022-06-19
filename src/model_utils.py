import sys
import argparse
import torch


class layer_head(torch.nn.Module):
    def __init__(self,):
        super(layer_head, self).__init__()
        self.m = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(8, 4, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(4),
            torch.nn.LeakyReLU(0.2),
            torch.nn.ConvTranspose2d(4, 1, kernel_size=1, stride=1, padding=0),
            torch.nn.ReLU()
        )
    def forward(self, x):
        return self.m(x)


def load_generator(gan_name, gan_weights, device=torch.device('cuda')):
    if gan_name=="stylegan3":
        sys.path.append("./src/nets/stylegan3")
        import dnnlib
        import legacy
        with dnnlib.util.open_url(gan_weights) as f:
            G = legacy.load_network_pkl(f)['G_ema'].to(device).eval()
    elif gan_name=="stylegan2":
        sys.path.append("./src/nets/stylegan3")
        import dnnlib
        import legacy
        with dnnlib.util.open_url(gan_weights) as f:
            G = legacy.load_network_pkl(f)['G_ema'].to(device).eval()
    return G


def load_invertibility(latent_names, ckpt_path):
    sys.path.append("./src/nets/pytorch-deeplab-xception")
    from modeling.deeplab import DeepLab
    model = DeepLab(num_classes=8, backbone="resnet", output_stride=16, sync_bn=False, freeze_bn=False)
    sd = torch.load(ckpt_path)
    model.load_state_dict(sd["sd_base"])
    model.eval().cuda()
    d_heads = {}
    for name in latent_names.split(","):
        d_heads[name] = layer_head().cuda()
        d_heads[name].load_state_dict(sd[name])
        d_heads[name].eval()
    def fn_inv(x):
        with torch.no_grad():
            _out = model(x)
            d_out = {n: d_heads[n](_out) for n in latent_names.split(",")}
        return d_out
    return fn_inv


def load_segmenter(segmenter_name, H=192, W=256):
    if segmenter_name=="hrnet_ade20k":
        from segmenter_utils import segmenter_hrnet
        S = segmenter_hrnet(ds="ade20k", H=H, W=W)
    elif segmenter_name=="detic":
        from segmenter_detic import segmenter_detic
        S = segmenter_detic()
    elif segmenter_name=="face_parser_fused":
        from segmenter_utils import segmenter_face
        S = segmenter_face("ckpt/79999_iter.pth", fuse_face_regions=True)
    elif segmenter_name=="detectron_coco":
        from segmenter_utils import segmenter_detectron
        S = segmenter_detectron("coco")
    return S


def load_encoder(ckpt_path, model_type="e4e"):
    if model_type=="e4e":
        # add e4e repo to python path
        sys.path.insert(0, "src/nets/encoder4editing")
        from models.psp import pSp
        ckpt = torch.load(ckpt_path, map_location='cpu')
        opts = ckpt['opts']
        opts['checkpoint_path'] = ckpt_path
        opts['device'] = torch.device("cuda")
        opts = argparse.Namespace(**opts)
        net = pSp(opts)
        net = net.eval().cuda()
        return net


def custom_forward(G, d_latents, gan_name,latent_name, d_masks=None):
    if gan_name=="stylegan2":
        from torch_utils import misc
        if latent_name=="W+,F4,F6,F8,F10":
            assert d_masks is not None
            ws = d_latents["W+"]
            block_ws = []
            with torch.autograd.profiler.record_function('split_ws'):
                misc.assert_shape(ws, [None, G.synthesis.num_ws, G.synthesis.w_dim])
                ws = ws.to(torch.float32)
                w_idx = 0
                for res in G.synthesis.block_resolutions:
                    block = getattr(G.synthesis, f'b{res}')
                    block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                    w_idx += block.num_conv
            x = img = None
            idx=1
            for res, cur_ws in zip(G.synthesis.block_resolutions, block_ws):
                block = getattr(G.synthesis, f'b{res}')
                x, img = block(x, img, cur_ws)
                # F4 is idx=3
                if idx==3:
                    m = d_masks["F4"]
                    x = x + d_latents["F4"]*m
                # F6 is idx=4
                if idx==4:
                    m = d_masks["F6"]
                    x = x + d_latents["F6"]*m
                # F8 is idx=5
                if idx==5:
                    m = d_masks["F8"]
                    x = x + d_latents["F8"]*m
                # F10 is idx=6
                if idx==6:
                    m = d_masks["F10"]
                    x = x + d_latents["F10"]*m
                idx+=1

        elif latent_name=="W+":
            ws = d_latents["W+"]
            block_ws = []
            with torch.autograd.profiler.record_function('split_ws'):
                misc.assert_shape(ws, [None, G.synthesis.num_ws, G.synthesis.w_dim])
                ws = ws.to(torch.float32)
                w_idx = 0
                for res in G.synthesis.block_resolutions:
                    block = getattr(G.synthesis, f'b{res}')
                    block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                    w_idx += block.num_conv
            x = img = None
            for res, cur_ws in zip(G.synthesis.block_resolutions, block_ws):
                block = getattr(G.synthesis, f'b{res}')
                x, img = block(x, img, cur_ws)

        elif latent_name=="F4":
            ws = d_latents["W+"]
            block_ws = []
            with torch.autograd.profiler.record_function('split_ws'):
                misc.assert_shape(ws, [None, G.synthesis.num_ws, G.synthesis.w_dim])
                ws = ws.to(torch.float32)
                w_idx = 0
                for res in G.synthesis.block_resolutions:
                    block = getattr(G.synthesis, f'b{res}')
                    block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                    w_idx += block.num_conv
            x = img = None
            idx=1
            for res, cur_ws in zip(G.synthesis.block_resolutions, block_ws):
                block = getattr(G.synthesis, f'b{res}')
                x, img = block(x, img, cur_ws)
                # F4 is idx=3
                if idx==3:
                    x = x + d_latents["F4"]
                idx+=1
        
        elif latent_name=="F6":
            ws = d_latents["W+"]
            block_ws = []
            with torch.autograd.profiler.record_function('split_ws'):
                misc.assert_shape(ws, [None, G.synthesis.num_ws, G.synthesis.w_dim])
                ws = ws.to(torch.float32)
                w_idx = 0
                for res in G.synthesis.block_resolutions:
                    block = getattr(G.synthesis, f'b{res}')
                    block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                    w_idx += block.num_conv
            x = img = None
            idx=1
            for res, cur_ws in zip(G.synthesis.block_resolutions, block_ws):
                block = getattr(G.synthesis, f'b{res}')
                x, img = block(x, img, cur_ws)
                # F6 is idx=4
                if idx==4:
                    x = x + d_latents["F6"]
                idx+=1
        
        elif latent_name=="F8":
            ws = d_latents["W+"]
            block_ws = []
            with torch.autograd.profiler.record_function('split_ws'):
                misc.assert_shape(ws, [None, G.synthesis.num_ws, G.synthesis.w_dim])
                ws = ws.to(torch.float32)
                w_idx = 0
                for res in G.synthesis.block_resolutions:
                    block = getattr(G.synthesis, f'b{res}')
                    block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                    w_idx += block.num_conv
            x = img = None
            idx=1
            for res, cur_ws in zip(G.synthesis.block_resolutions, block_ws):
                block = getattr(G.synthesis, f'b{res}')
                x, img = block(x, img, cur_ws)
                # F8 is idx=5
                if idx==5:
                    x = x + d_latents["F8"]
                idx+=1

        elif latent_name=="F10":
            ws = d_latents["W+"]
            block_ws = []
            with torch.autograd.profiler.record_function('split_ws'):
                misc.assert_shape(ws, [None, G.synthesis.num_ws, G.synthesis.w_dim])
                ws = ws.to(torch.float32)
                w_idx = 0
                for res in G.synthesis.block_resolutions:
                    block = getattr(G.synthesis, f'b{res}')
                    block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                    w_idx += block.num_conv
            x = img = None
            idx=1
            for res, cur_ws in zip(G.synthesis.block_resolutions, block_ws):
                block = getattr(G.synthesis, f'b{res}')
                x, img = block(x, img, cur_ws)
                # F10 is idx=6
                if idx==6:
                    x = x + d_latents["F10"]
                idx+=1
    return img


def partial_forward(G, d_latents, gan_name, out_layer, d_masks):
    if gan_name=="stylegan2":
        from torch_utils import misc
        assert d_masks is not None
        ws = d_latents["W+"]
        block_ws = []
        with torch.autograd.profiler.record_function('split_ws'):
            misc.assert_shape(ws, [None, G.synthesis.num_ws, G.synthesis.w_dim])
            ws = ws.to(torch.float32)
            w_idx = 0
            for res in G.synthesis.block_resolutions:
                block = getattr(G.synthesis, f'b{res}')
                block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv
        x = img = None
        idx=1
        for res, cur_ws in zip(G.synthesis.block_resolutions, block_ws):
            block = getattr(G.synthesis, f'b{res}')
            x, img = block(x, img, cur_ws)
            # F4 is idx=3
            if idx==3:
                m = d_masks["F4"]
                x = x + d_latents["F4"]*m
                if out_layer=="F4":
                    return x
            # F6 is idx=4
            if idx==4:
                m = d_masks["F6"]
                x = x + d_latents["F6"]*m
                if out_layer=="F6":
                    return x.detach().clone()
            # F8 is idx=5
            if idx==5:
                m = d_masks["F8"]
                x = x + d_latents["F8"]*m
                if out_layer=="F8":
                    return x
            # F10 is idx=6
            if idx==6:
                m = d_masks["F10"]
                x = x + d_latents["F10"]*m
                if out_layer=="F10":
                    return x
            idx+=1
        raise ValueError("specified output layer is invalid")


def edit_image(G, d_latents, gan_name, edit_dir_wp, d_masks):
    if gan_name=="stylegan2":
        F4 = partial_forward(G, d_latents, gan_name, "F4", d_masks)
        F6 = partial_forward(G, d_latents, gan_name, "F6", d_masks)
        F8 = partial_forward(G, d_latents, gan_name, "F8", d_masks)
        F10 = partial_forward(G, d_latents, gan_name, "F10", d_masks)

        ws = d_latents["W+"].detach().clone() + edit_dir_wp
        block_ws = []
        with torch.autograd.profiler.record_function('split_ws'):
            #misc.assert_shape(ws, [None, G.synthesis.num_ws, G.synthesis.w_dim])
            ws = ws.to(torch.float32)
            w_idx = 0
            for res in G.synthesis.block_resolutions:
                block = getattr(G.synthesis, f'b{res}')
                block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv
        x = img = None
        idx=1
        for res, cur_ws in zip(G.synthesis.block_resolutions, block_ws):
            block = getattr(G.synthesis, f'b{res}')
            x, img = block(x, img, cur_ws)
            # F4 is idx=3
            if idx==3:
                m = d_masks["F4"]
                x = x*(1-m) + F4*m
            # F6 is idx=4
            if idx==4:
                m = d_masks["F6"]
                x = x*(1-m) + F6*m
            # F8 is idx=5
            if idx==5:
                m = d_masks["F8"]
                x = x*(1-m) + F8*m
            # F10 is idx=6
            if idx==6:
                m = d_masks["F10"]
                x = x*(1-m) + F10*m
            idx+=1
        return img