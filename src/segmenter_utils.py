import os
import sys
import wget
import numpy as np
import torch
import torchvision.transforms as transforms


class segmenter_hrnet:
    def __init__(self, ds="ade20k", H=192, W=256):
        # load the relevant files
        sys.path.append("./src/nets/semantic-segmentation-pytorch")
        from mit_semseg.config import cfg
        from mit_semseg.models import ModelBuilder, SegmentationModule

        if ds == "ade20k":
            model_name = "ade20k-resnet50dilated-ppm_deepsup"
            cfg_f = f"src/nets/semantic-segmentation-pytorch/config/{model_name}.yaml"
            cfg.merge_from_file(cfg_f)
            cfg.MODEL.weights_encoder = f"src/nets/semantic-segmentation-pytorch/ckpt/ade20k-resnet50dilated-ppm_deepsup/encoder_epoch_20.pth"
            if not os.path.exists(cfg.MODEL.weights_encoder):
                os.makedirs(os.path.dirname(cfg.MODEL.weights_encoder), exist_ok=True)
                wget.download("http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-resnet50dilated-ppm_deepsup/encoder_epoch_20.pth", out=cfg.MODEL.weights_encoder)
            cfg.MODEL.weights_decoder = f"src/nets/semantic-segmentation-pytorch/ckpt/ade20k-resnet50dilated-ppm_deepsup/decoder_epoch_20.pth"
            if not os.path.exists(cfg.MODEL.weights_decoder):
                os.makedirs(os.path.dirname(cfg.MODEL.weights_decoder), exist_ok=True)
                wget.download("http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-resnet50dilated-ppm_deepsup/decoder_epoch_20.pth", out=cfg.MODEL.weights_decoder)
            
        # build the network
        net_encoder = ModelBuilder.build_encoder(arch=cfg.MODEL.arch_encoder,fc_dim=cfg.MODEL.fc_dim,weights=cfg.MODEL.weights_encoder)
        net_decoder = ModelBuilder.build_decoder(arch=cfg.MODEL.arch_decoder,fc_dim=cfg.MODEL.fc_dim,num_class=cfg.DATASET.num_class,weights=cfg.MODEL.weights_decoder,use_softmax=True)
        self.segmentation_module = SegmentationModule(net_encoder, net_decoder, torch.nn.NLLLoss(ignore_index=-1))
        self.segmentation_module.eval().cuda()
        self.cfg=cfg
        self.H=H
        self.W=W
        self.ds=ds

    def transform(self, img_pil):
        normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        # 0-255 to 0-1
        img = np.float32(np.array(img_pil)) / 255.
        img = img.transpose((2, 0, 1))
        img = normalize(torch.from_numpy(img.copy()))
        return img

    def segment_pil(self, img_pil):
        img_t = self.transform(img_pil).unsqueeze(0).cuda()
        with torch.no_grad():
            pred = self.segmentation_module({"img_data": img_t}, segSize=(self.H, self.W))
            _, pred = torch.max(pred, dim=1)
        pred_np = pred.squeeze(0).cpu().numpy()
        # combine some classes together
        # 4(floor), 7(road), 14(earth)
        pred_np[pred_np==6]=3
        pred_np[pred_np==13]=3
        # 61(river), 22(water)
        pred_np[pred_np==21]=60
        # 21(car), 84(truck), 103(van)
        pred_np[pred_np==102]=20
        pred_np[pred_np==83]=20
        # 10(grass), 30(field)
        pred_np[pred_np==29]=9
        return pred_np
    
    def segment_t(self, img_t, renorm=True):
        # assume image starts with 0.5, 0.5 normalization
        if renorm:
            img_t = img_t*0.5 + 0.5
            N = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            img_t = N(img_t)
        with torch.no_grad():
            pred = self.segmentation_module({"img_data": img_t}, segSize=(self.H, self.W))
            _, pred = torch.max(pred, dim=1)
        pred_np = pred.squeeze(0).cpu().numpy()
        # fuse some classes together
        # 4(floor), 7(road), 14(earth)
        pred_np[pred_np==6]=3
        pred_np[pred_np==13]=3
        # 61(river), 22(water)
        pred_np[pred_np==21]=60
        # 21(car), 84(truck), 103(van)
        pred_np[pred_np==102]=20
        pred_np[pred_np==83]=20
        # 10(grass), 30(field)
        pred_np[pred_np==29]=9
        return pred_np


class segmenter_face:
    def __init__(self, ckpt_path="ckpt/79999_iter.pth", fuse_face_regions=True):
        # load the relevant files
        sys.path.append("./src/nets/face-parsing.PyTorch")
        from model import BiSeNet
        
        self.net = BiSeNet(n_classes=19).cuda()
        self.net.load_state_dict(torch.load(ckpt_path))
        self.net.eval()
        self.fuse_face_regions = fuse_face_regions
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def segment_pil(self, img_pil):
        img_t = self.transform(img_pil).unsqueeze(0).cuda()
        out = self.net(img_t)[0]
        parsed = out.squeeze(0).detach().cpu().numpy().argmax(0)
        if self.fuse_face_regions:
            """
            1 - skin
            2/3 - left/right brow
            4/5 - left/right eye
            7/8 - left/right ear
            10 - nose
            11 - mouth
            12/13 - upper/lower lips
            14 - neck
            17 - hair
            """
            for idx in [1,2,3,4,5,7,8,10,11,12,13,14]:
                parsed[parsed==idx]=3
        return parsed


class segmenter_detectron:
    def __init__(self, ds_name="coco", t=0.1):
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor

        if ds_name=="coco":
            c = "COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"
        
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file(c))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = t
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(c)
        self.pred = DefaultPredictor(self.cfg)

    def segment_pil(self, img_pil):
        img_np = np.array(img_pil)
        predictions = self.pred(img_np)
        n = len(predictions["instances"])
        S1,S2 = img_np.shape[0],img_np.shape[1] 
        out = np.zeros((S1,S2))
        for i in range(n)[:]:
            mask = predictions["instances"][i].pred_masks.detach().cpu().numpy()
            assert mask.shape[0]==1
            class_idx = predictions["instances"][i].pred_classes.item()
            out[mask[0]] = class_idx
        return out
    
