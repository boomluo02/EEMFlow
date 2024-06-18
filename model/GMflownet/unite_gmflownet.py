import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("/home/luoxinglong/unite_raft/model/GMflownet")
from update import BasicUpdateBlock
from extractor import BasicEncoder, BasicConvEncoder
from corr import CorrBlock, AlternateCorrBlock
from model_utils import bilinear_sampler, coords_grid, upflow8
from swin_transformer import POLAUpdate, MixAxialPOLAUpdate
from argparse import Namespace

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
                     dropout=False,
                     mixed_precision=False,
                     use_mix_attn=True,
                     alternate_corr=False,
                     clip=1.0)
    return args

class GMFlowNetRES(nn.Module):
    def __init__(self, config,  n_first_channels=5):
        super().__init__()
        args = get_args()
        self.args = args

        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        args.corr_levels = 4
        args.corr_radius = 4

        self.unet = MIMOUNet(n_first_channels, n_first_channels)
        self.unet_sk = SK()

        # feature network, context network, and update block
        if self.args.use_mix_attn:
            self.fnet = nn.Sequential(
                            BasicConvEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout),
                            MixAxialPOLAUpdate(embed_dim=256, depth=6, num_head=8, window_size=7)
                        )
        else:
            self.fnet = nn.Sequential(
                BasicConvEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout),
                POLAUpdate(embed_dim=256, depth=6, num_head=8, window_size=7, neig_win_num=1)
            )

        self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout)
        self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim, input_dim=cdim)

    def change_imagesize(self, img_size):
        self.image_size = img_size

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


    def forward(self, events1, events2, iters=12, flow_init=None, upsample=True):
        """ Estimate optical flow between pair of frames """

        events1 = events1.contiguous()
        events2 = events2.contiguous()

        unet_out = self.unet([events1, events2], with_res=False)
        image1, image2 = self.unet_sk([events1, events2], [unet_out[-1][0], unet_out[-1][1]])

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

        # # Self-attention update
        # fmap1 = self.transEncoder(fmap1)
        # fmap2 = self.transEncoder(fmap2)

        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        else:
            corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        # Correlation as initialization
        N, fC, fH, fW = fmap1.shape
        corrMap = corr_fn.corrMap

        #_, coords_index = torch.max(corrMap, dim=-1) # no gradient here
        softCorrMap = F.softmax(corrMap, dim=2) * F.softmax(corrMap, dim=1) # (N, fH*fW, fH*fW)

        if flow_init is not None:
            coords1 = coords1 + flow_init
        else:
            # print('matching as init')
            # mutual match selection
            match12, match_idx12 = softCorrMap.max(dim=2) # (N, fH*fW)
            match21, match_idx21 = softCorrMap.max(dim=1)

            for b_idx in range(N):
                match21_b = match21[b_idx,:]
                match_idx12_b = match_idx12[b_idx,:]
                match21[b_idx,:] = match21_b[match_idx12_b]

            matched = (match12 - match21) == 0  # (N, fH*fW)
            coords_index = torch.arange(fH*fW).unsqueeze(0).repeat(N,1).to(softCorrMap.device)
            coords_index[matched] = match_idx12[matched]

            # matched coords
            coords_index = coords_index.reshape(N, fH, fW)
            coords_x = coords_index % fW
            coords_y = coords_index // fW

            coords_xy = torch.stack([coords_x, coords_y], dim=1).float()
            coords1 = coords_xy

        # Iterative update
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

            flow_predictions.append(flow_up)
        
        map_out = []
        for maps in unet_out:
            map1, map2 = maps[0], maps[1]
            map_out.append([map1, map2])

        if self.training:
            flow_out = [flow_predictions, softCorrMap]
            return map_out, flow_out
        else:
            return map_out, flow_predictions

class GMFlowNetRES_IN(nn.Module):
    def __init__(self, config):
        super().__init__()
        args = get_args()
        self.args = args

        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        args.corr_levels = 4
        args.corr_radius = 4

        n_first_channels = 5
        self.unet = MIMOUNet(n_first_channels, n_first_channels)
        self.unet_sk = SK()

        # feature network, context network, and update block
        if self.args.use_mix_attn:
            self.fnet = nn.Sequential(
                            BasicConvEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout),
                            MixAxialPOLAUpdate(embed_dim=256, depth=6, num_head=8, window_size=7)
                        )
        else:
            self.fnet = nn.Sequential(
                BasicConvEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout),
                POLAUpdate(embed_dim=256, depth=6, num_head=8, window_size=7, neig_win_num=1)
            )

        self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='instance', dropout=args.dropout)
        self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim, input_dim=cdim)

    def change_imagesize(self, img_size):
        self.image_size = img_size

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


    def forward(self, events1, events2, iters=12, flow_init=None, upsample=True):
        """ Estimate optical flow between pair of frames """

        events1 = events1.contiguous()
        events2 = events2.contiguous()

        unet_out = self.unet([events1, events2], with_res=False)
        image1, image2 = self.unet_sk([events1, events2], [unet_out[-1][0], unet_out[-1][1]])

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

        # # Self-attention update
        # fmap1 = self.transEncoder(fmap1)
        # fmap2 = self.transEncoder(fmap2)

        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        else:
            corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        # Correlation as initialization
        N, fC, fH, fW = fmap1.shape
        corrMap = corr_fn.corrMap

        #_, coords_index = torch.max(corrMap, dim=-1) # no gradient here
        softCorrMap = F.softmax(corrMap, dim=2) * F.softmax(corrMap, dim=1) # (N, fH*fW, fH*fW)

        if flow_init is not None:
            coords1 = coords1 + flow_init
        else:
            # print('matching as init')
            # mutual match selection
            match12, match_idx12 = softCorrMap.max(dim=2) # (N, fH*fW)
            match21, match_idx21 = softCorrMap.max(dim=1)

            for b_idx in range(N):
                match21_b = match21[b_idx,:]
                match_idx12_b = match_idx12[b_idx,:]
                match21[b_idx,:] = match21_b[match_idx12_b]

            matched = (match12 - match21) == 0  # (N, fH*fW)
            coords_index = torch.arange(fH*fW).unsqueeze(0).repeat(N,1).to(softCorrMap.device)
            coords_index[matched] = match_idx12[matched]

            # matched coords
            coords_index = coords_index.reshape(N, fH, fW)
            coords_x = coords_index % fW
            coords_y = coords_index // fW

            coords_xy = torch.stack([coords_x, coords_y], dim=1).float()
            coords1 = coords_xy

        # Iterative update
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

            flow_predictions.append(flow_up)
        
        map_out = []
        for maps in unet_out:
            map1, map2 = maps[0], maps[1]
            map_out.append([map1, map2])

        if self.training:
            flow_out = [flow_predictions, softCorrMap]
            return map_out, flow_out
        else:
            return map_out, flow_predictions
    

        
