import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
import torch.optim as optim
import itertools
from model.warplayer import warp
from torch.nn.parallel import DistributedDataParallel as DDP
from ckpt.rifev4_25.IFNet_HDv3 import *
import torch.nn.functional as F
from model.loss import *

if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
    
class Model:
    def __init__(self, local_rank=-1):
        self.flownet = IFNet()
        self.device()
        self.optimG = AdamW(self.flownet.parameters(), lr=1e-6, weight_decay=1e-4)
        self.epe = EPE()
        self.version = 4.25
        # self.vgg = VGGPerceptualLoss().to(device)
        self.sobel = SOBEL()
        if local_rank != -1:
            self.flownet = DDP(self.flownet, device_ids=[local_rank], output_device=local_rank)

    def train(self):
        self.flownet.train()

    def eval(self):
        self.flownet.eval()

    def device(self):
        self.flownet.to(device)

    def load_model(self, path, rank=0):
        def convert(param):
            if rank == -1:
                return {
                    k.replace("module.", ""): v
                    for k, v in param.items()
                    if "module." in k
                }
            else:
                return param
        if rank <= 0:
            if torch.cuda.is_available():
                self.flownet.load_state_dict(convert(torch.load(f'{path}/flownet.pkl')), False)
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.flownet.load_state_dict(convert(torch.load(f'{path}/flownet.pkl', map_location='mps')), False)
            else:
                self.flownet.load_state_dict(convert(torch.load(f'{path}/flownet.pkl', map_location='cpu')), False)

        
    def save_model(self, path, rank=0):
        if rank == 0:
            torch.save(self.flownet.state_dict(),'{}/flownet.pkl'.format(path))

    def inference(self, img0, img1, timestep=0.5, scale=1.0, prev_flows=[], prev_masks=[]):
        imgs = torch.cat((img0, img1), 1)
        scale_list = [16/scale, 8/scale, 4/scale, 2/scale, 1/scale]
        flow, mask, merged = self.flownet(imgs, timestep, scale_list)
        return merged[-1]
    
    def inference_recursive(self, img0, img1, timestep=0.5, scale=1.0, prev_flows=[], prev_masks=[]):
        imgs = torch.cat((img0, img1), 1)
        scale_list = [16/scale, 8/scale, 4/scale, 2/scale, 1/scale]
        flow, mask, merged = self.flownet.forward_recusrive(imgs, timestep, scale_list, prev_flows=prev_flows, prev_masks=prev_masks)
        return merged[-1]
    
    def inference_global(self, img0, img1,flows, masks, betas, timestep=0.5, scale=1.0):
        imgs = torch.cat((img0, img1), 1)
        scale_list = [16/scale, 8/scale, 4/scale, 2/scale, 1/scale]
        merged = self.flownet.forward_global(imgs, flows, masks, betas, timestep, scale_list)
        return merged
    
    def flow_extractor(self, img0, img1, timestep=0.5, scale=1.0):
        imgs = torch.cat((img0, img1), 1)
        scale_list = [16/scale, 8/scale, 4/scale, 2/scale, 1/scale]
        flow, mask, merged = self.flownet(imgs, timestep, scale_list)
        return flow[-1], mask[-1]
    
    def warp(self, img0, img1, flow, mask):
        imgs = torch.cat((img0, img1), 1)
        merged = self.flownet.forward_warp(imgs, flow, mask)
        return merged
    

