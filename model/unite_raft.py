import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .update import BasicUpdateBlock
from .extractor import BasicEncoder
from .corr import CorrBlock
from model_utils import coords_grid, upflow8
from argparse import Namespace
from utils.image_utils import ImagePadder
# from .unet import UNet
from .unet import U_Net as UNet
from utils.image_utils import InputPadder
import pdb

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


def get_args():
    # This is an adapter function that converts the arguments given in out config file to the format, which the ERAFT
    # expects.
    args = Namespace(small=False,
                     dropout=False,
                     mixed_precision=False,
                     clip=1.0)
    return args



class ERAFT(nn.Module):
    def __init__(self, config, n_first_channels):
        # args:
        super(ERAFT, self).__init__()
        args = get_args()
        self.args = args
        self.image_padder = ImagePadder(min_size=32)

        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        args.corr_levels = 4
        args.corr_radius = 4

        # unet network
        self.in_channels = n_first_channels
        self.key_kernels = config['key_kernels']
        self.unet = UNet(self.in_channels, self.key_kernels)

        self.filter_size = config['filter_size']

        # feature network, context network, and update block
        self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=0,
                                    n_first_channels=self.key_kernels)
        self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=0,
                                    n_first_channels=self.key_kernels)
        self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)
    
    def change_imagesize(self, img_size):
        self.image_size = img_size
        # self.image_padder = InputPadder(img_size, mode='chairs')


    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8).to(img.device)
        coords1 = coords_grid(N, H//8, W//8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)

    def _max_filter(self, output, filter_size = 3):
        n, c ,h, w = output.shape
        center_w = filter_size // 2

        windows = F.unfold(output, kernel_size=filter_size, padding=center_w, stride=1)
        windows_s = windows.reshape([n,c,filter_size,filter_size,h,w])
        windows_sk = windows_s.reshape([n,c,filter_size*filter_size,h,w])
        
        max_wind, _ = torch.max(windows_sk, dim=2, keepdim=False)

        mask = (output >= max_wind)
        res = output * mask

        return res, mask

    def forward(self, events1, events2, iters=12, flow_init=None, upsample=True, normal=False):

        image1 = self.image_padder.pad(events1)
        image2 = self.image_padder.pad(events2)

        # image1, image2 = self.image_padder.pad(events1, events2)
        # pdb.set_trace()
        image1 = image1.contiguous()
        image2 = image2.contiguous()

        key_map1, key_map2 = self.unet([image1, image2])
        
        
        if normal:
            key_map1 = 2 * ((key_map1 - torch.min(key_map1)) / (torch.max(key_map1) - torch.min(key_map1))) - 1.0
            key_map1 = 2 * ((key_map2 - torch.min(key_map2)) / (torch.max(key_map2) - torch.min(key_map2))) - 1.0

        # for c in range(key_map1.shape[1]):
        #     key_map1[:, c, ...] = 2 * ((key_map1[:, c, ...] - torch.min(key_map1[:, c, ...])) / (torch.max(key_map1[:, c, ...]) - torch.min(key_map1[:, c, ...]))) - 1.0
        #     key_map2[:, c, ...] = 2 * ((key_map2[:, c, ...] - torch.min(key_map2[:, c, ...])) / (torch.max(key_map2[:, c, ...]) - torch.min(key_map2[:, c, ...]))) - 1.0
        
        if self.filter_size > 0:
            key_map1, mask1 = self._max_filter(key_map1, filter_size=self.filter_size)
            key_map2, mask2 = self._max_filter(key_map2, filter_size=self.filter_size)
            
            mask1, mask2 = self.image_padder.unpad(mask1), self.image_padder.unpad(mask2)
        else:
            mask1 = None
            mask2 = None


        key_map1 = key_map1.contiguous()
        key_map2 = key_map2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fmap1, fmap2 = self.fnet([key_map1, key_map2])
        
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

        corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):

            cnet = self.cnet(key_map1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        # Initialize Grids. First channel: x, 2nd channel: y. Image is just used to get the shape
        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)

            flow_predictions.append(self.image_padder.unpad(flow_up))

        unet_out1, unet_out2 = self.image_padder.unpad(key_map1), self.image_padder.unpad(key_map2)


        return  (unet_out1, unet_out2, mask1, mask2),  flow_predictions
