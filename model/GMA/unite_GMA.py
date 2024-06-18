import torch
import torch.nn as nn
import torch.nn.functional as F

from .update import GMAUpdateBlock
from .extractor import BasicEncoder
from .corr import CorrBlock
from .utils.utils import bilinear_sampler, coords_grid, upflow8
from .gma import Attention, Aggregate
from argparse import Namespace
from utils.image_utils import ImagePadder, InputPadder

import sys
sys.path.append('/home/luoxinglong/unite_raft')
from model.MIMO_unet.mimo_unet import MIMOUNet_little as MIMOUNet
from model.sknet import SK

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
                    #  dropout=False,
                     mixed_precision=False,
                     clip=1.0,
                     num_heads=1,
                     position_only=False,
                     position_and_content=False)
    return args

class RAFTGMARES(nn.Module):
    def __init__(self, config, n_first_channels):
        # args:
        super(RAFTGMARES, self).__init__()
        args = get_args()
        self.args = args

        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        args.corr_levels = 4
        args.corr_radius = 4

        if 'dropout' not in self.args:
            self.args.dropout = 0

        self.unet = MIMOUNet(n_first_channels, n_first_channels)
        self.unet_sk = SK()

        # feature network, context network, and update block
        self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout,
                                    n_first_channels=n_first_channels)
        self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout,
                                    n_first_channels=n_first_channels)
        self.update_block = GMAUpdateBlock(self.args, hidden_dim=hdim)
        self.att = Attention(args=self.args, dim=cdim, heads=self.args.num_heads, max_pos_size=160, dim_head=cdim)

        self.resnet = True
        if("without_res" in config.keys()):
            self.resnet = False
    
    def change_imagesize(self, img_size):
        self.image_size = img_size
        self.image_padder = InputPadder(img_size, mode='chairs')

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8).to(img.device)
        coords1 = coords_grid(N, H // 8, W // 8).to(img.device)

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
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def forward(self, events1, events2, iters=12, flow_init=None, upsample=True, normal=False, test_mode=False):
        """ Estimate optical flow between pair of frames """

        events1, events2 = self.image_padder.pad(events1, events2)

        events1 = events1.contiguous()
        events2 = events2.contiguous()

        unet_out = self.unet([events1, events2], with_res=self.resnet)
        image1, image2 = self.unet_sk([events1, events2], [unet_out[-1][0], unet_out[-1][1]])

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)
            # attention, att_c, att_p = self.att(inp)
            attention = self.att(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)  # index correlation volume

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow, attention)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)

            flow_predictions.append(self.image_padder.unpad(flow_up))

        map_out = []
        for maps in unet_out:
            map1, map2 = maps[0], maps[1]
            map_out.append([self.image_padder.unpad(map1), self.image_padder.unpad(map2)])
        # map_out.append([self.image_padder.unpad(unet_out[0][0]), self.image_padder.unpad(unet_out[0][1])])
        # map_out.append([self.image_padder.unpad(unet_out[1][0]), self.image_padder.unpad(unet_out[1][1])])
        # map_out.append([image1,image2])

        return map_out, flow_predictions

class RAFTGMARES_IN(nn.Module):
    def __init__(self, config, n_first_channels):
        # args:
        super(RAFTGMARES_IN, self).__init__()
        args = get_args()
        self.args = args

        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        args.corr_levels = 4
        args.corr_radius = 4

        if 'dropout' not in self.args:
            self.args.dropout = 0

        self.unet = MIMOUNet(n_first_channels, n_first_channels)
        self.unet_sk = SK()

        # feature network, context network, and update block
        self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout,
                                    n_first_channels=n_first_channels)
        self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='instance', dropout=args.dropout,
                                    n_first_channels=n_first_channels)
        self.update_block = GMAUpdateBlock(self.args, hidden_dim=hdim)
        self.att = Attention(args=self.args, dim=cdim, heads=self.args.num_heads, max_pos_size=160, dim_head=cdim)

        self.resnet = True
        if("without_res" in config.keys()):
            self.resnet = False
    
    def change_imagesize(self, img_size):
        self.image_size = img_size
        self.image_padder = InputPadder(img_size, mode='chairs')

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8).to(img.device)
        coords1 = coords_grid(N, H // 8, W // 8).to(img.device)

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
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def forward(self, events1, events2, iters=12, flow_init=None, upsample=True, normal=False, test_mode=False):
        """ Estimate optical flow between pair of frames """

        events1, events2 = self.image_padder.pad(events1, events2)

        events1 = events1.contiguous()
        events2 = events2.contiguous()

        unet_out = self.unet([events1, events2], with_res=self.resnet)
        image1, image2 = self.unet_sk([events1, events2], [unet_out[-1][0], unet_out[-1][1]])

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)
            # attention, att_c, att_p = self.att(inp)
            attention = self.att(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)  # index correlation volume

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow, attention)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)

            flow_predictions.append(self.image_padder.unpad(flow_up))

        map_out = []
        for maps in unet_out:
            map1, map2 = maps[0], maps[1]
            map_out.append([self.image_padder.unpad(map1), self.image_padder.unpad(map2)])
        # map_out.append([self.image_padder.unpad(unet_out[0][0]), self.image_padder.unpad(unet_out[0][1])])
        # map_out.append([self.image_padder.unpad(unet_out[1][0]), self.image_padder.unpad(unet_out[1][1])])
        # map_out.append([image1,image2])

        return map_out, flow_predictions