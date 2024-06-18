import os,sys
sys.path.append('/home/luoxinglong/unite_raft')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.update import BasicUpdateBlock
from model.extractor import BasicEncoder
from model.corr import CorrBlock
from model.model_utils import coords_grid, upflow8
from argparse import Namespace
from utils.image_utils import InputPadder

# from model.e2vid import UNetRecurrent as UNet
from model.unet import U_Net as UNet
from model.unet import U_Net_l as UNet

from model.MIMO_unet.mimo_unet import MIMOUNet_little as MIMOUNet
# from model.MIMO_unet.mimo_unet import MIMOUNet

from model.resnet import resnet50, resnet101
from model.sknet import SK, SK_score

from utils_luo.tools import tools, tensor_tools

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

        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        args.corr_levels = 4
        args.corr_radius = 4

        # unet network
        self.in_channels = n_first_channels
        self.unet = UNet(self.in_channels, self.in_channels)
        self.unet_sk = SK()

        # feature network, context network, and update block
        self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=0,
                                    n_first_channels=self.in_channels)
        # self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=0,
        #                             n_first_channels=self.in_channels)
        self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='instance', dropout=0,
                                    n_first_channels=self.in_channels)
        self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)
    
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

    def forward(self, events1, events2, iters=12, flow_init=None, upsample=True, normal=False):

        # image1 = self.image_padder.pad(events1)
        # image2 = self.image_padder.pad(events2)

        events1, events2 = self.image_padder.pad(events1, events2)
        # pdb.set_trace()
        events1 = events1.contiguous()
        events2 = events2.contiguous()
        # print(1)
        unet_out1, unet_out2 = self.unet([events1, events2])
    
        # image1 = unet_out1
        # image2 = unet_out2

        image1 = self.unet_sk(events1, unet_out1)
        image2 = self.unet_sk(events2, unet_out2)
        
        # image1, image2 = self.unet_sk([events1, events2], [unet_out1, unet_out2])

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

        unet_out1, unet_out2 = self.image_padder.unpad(unet_out1), self.image_padder.unpad(unet_out2)

        return  (unet_out1, unet_out2),  flow_predictions
    
    def run_unet(self, events1, events2):
        image1, image2 = self.image_padder.pad(events1, events2)
        image1 = image1.contiguous()
        image2 = image2.contiguous()

        unet_out1, unet_out2 = self.unet([image1, image2])

        return  (unet_out1, unet_out2)
    
    @classmethod
    def demo(cls):
        im = torch.zeros((3, 5, 256, 256))
        config = ''
        net = ERAFT(config,n_first_channels=5)
        net.change_imagesize((256,256))
        maps, out = net(im,im)
        tensor_tools.check_tensor(im, 'im')
        tensor_tools.check_tensor(out[0], 'out')
        tensor_tools.compute_model_size(net, im, im)
    

class ERAFT_denseCTX(nn.Module):
    def __init__(self, config, n_first_channels):
        # args:
        super(ERAFT_denseCTX, self).__init__()
        args = get_args()
        self.args = args

        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        args.corr_levels = 4
        args.corr_radius = 4

        self.in_channels = n_first_channels

        # feature network, context network, and update block
        self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=0,
                                    n_first_channels=self.in_channels)
        self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=0,
                                    n_first_channels=self.in_channels)
        self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim*2)
    
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

    def forward(self, events1, events2, d_events1, d_events2, iters=12, flow_init=None, upsample=True, normal=False):

        # image1 = self.image_padder.pad(events1)
        # image2 = self.image_padder.pad(events2)

        image1, image2, d_image1, d_image2 = self.image_padder.pad(events1, events2, d_events1, d_events2)

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        d_image1.requires_grad = False
        d_image2.requires_grad = False

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

            cnet1, d_cnet1 = self.cnet([image1, d_image1])
            d_cnet1_reg = d_cnet1.detach()

            cnet1_reg = cnet1.clone()
            cnet1_reg = cnet1.contiguous()

            net1, inp1 = torch.split(cnet1, [hdim, cdim], dim=1)
            d_net1, d_inp1 = torch.split(cnet1_reg, [hdim, cdim], dim=1)

            net = torch.cat([net1, d_net1], dim=1)
            net = torch.tanh(net)
            inp = torch.cat([inp1, d_inp1], dim=1)
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


        return  (cnet1_reg, d_cnet1_reg),  flow_predictions
    
    @classmethod
    def demo(cls):
        im = torch.zeros((3, 5, 256, 256))
        config = ''
        net = ERAFT_denseCTX(config,n_first_channels=5)
        net.change_imagesize((256,256))
        cnet,out = net(im,im,im,im)
        tensor_tools.check_tensor(im, 'im')
        tensor_tools.check_tensor(out[0], 'out')
        tensor_tools.check_tensor(cnet[0], 'ctx0')
        tensor_tools.check_tensor(cnet[1], 'd_ctx0')
        tensor_tools.compute_model_size(net, im,im,im,im)

class MIMOUNET_ERAFT(nn.Module):
    def __init__(self, config, n_first_channels):
        # args:
        super(MIMOUNET_ERAFT, self).__init__()
        args = get_args()
        self.args = args

        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        args.corr_levels = 4
        args.corr_radius = 4

        # unet network
        self.in_channels = n_first_channels
        self.unet = MIMOUNet(self.in_channels, self.in_channels)
        # self.unet = MIMOUNetPlus(self.in_channels, self.in_channels)

        # feature network, context network, and update block
        self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=0,
                                    n_first_channels=self.in_channels)
        # self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=0,
        #                             n_first_channels=self.in_channels)
        self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='instance', dropout=0,
                                    n_first_channels=self.in_channels)
        self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

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

    def forward(self, events1, events2, iters=12, flow_init=None, upsample=True, normal=False):

        events1, events2 = self.image_padder.pad(events1, events2)

        events1 = events1.contiguous()
        events2 = events2.contiguous()

        unet_out = self.unet([events1, events2], with_res=self.resnet)
        image1, image2 = unet_out[-1][0], unet_out[-1][1]
        # image1, image2 = events1, events2

        # image1 = unet_out[-1][0]*0.5 + events1*0.5
        # image2 = unet_out[-1][1]*0.5 + events2*0.5


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
        
        map_out = []
        for maps in unet_out:
            map1, map2 = maps
            map_out.append([self.image_padder.unpad(map1), self.image_padder.unpad(map2)])

        return map_out, flow_predictions
        # return (image1,image2), flow_predictions
    
    @classmethod
    def demo(cls):
        im = torch.zeros((3, 5, 256, 256))
        config = ''
        net = MIMOUNET_ERAFT(config,n_first_channels=5)
        net.change_imagesize((256,256))
        maps,out = net(im,im)
        tensor_tools.check_tensor(im, 'im')
        tensor_tools.check_tensor(out[0], 'out')
        tensor_tools.check_tensor(maps[-1][0], 'maps30')
        tensor_tools.check_tensor(maps[-1][1], 'maps31')
        tensor_tools.compute_model_size(net, im,im)


class MIMOUNET_RES_ERAFT(nn.Module):
    def __init__(self, config, n_first_channels):
        # args:
        super(MIMOUNET_RES_ERAFT, self).__init__()
        args = get_args()
        self.args = args

        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        args.corr_levels = 4
        args.corr_radius = 4

        # unet network
        self.in_channels = n_first_channels

        self.unet = MIMOUNet(self.in_channels, self.in_channels)
        self.unet_sk = SK()
        # self.unet = MIMOUNetPlus(self.in_channels, self.in_channels)
        # self.unet_res = resnet50(self.in_channels, self.in_channels)

        # feature network, context network, and update block
        self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=0,
                                    n_first_channels=self.in_channels)
        # self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=0,
        #                             n_first_channels=self.in_channels)
        self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='instance', dropout=0,
                                    n_first_channels=self.in_channels)
        
        self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

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

    def forward(self, events1, events2, iters=12, flow_init=None, upsample=True, normal=False):

        events1, events2 = self.image_padder.pad(events1, events2)

        events1 = events1.contiguous()
        events2 = events2.contiguous()

        unet_out = self.unet([events1, events2], with_res=self.resnet)

        # image1, image2 = unet_out[-1][0], unet_out[-1][1]

        # unet_prob1, unet_prob2 = self.unet_res([image1, image2])

        # map1 = unet_out[-1][0] * unet_prob1 + image1 * (1 - unet_prob1)
        # map2 = unet_out[-1][1] * unet_prob2 + image2 * (1 - unet_prob2)

        image1, image2 = self.unet_sk([events1, events2], [unet_out[-1][0], unet_out[-1][1]])

        # image1, image2 = events1, events2
    
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
        
        # if(self.training == False):
        #     print(cnet.mean())

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
        
        map_out = []
        for maps in unet_out:
            map1, map2 = maps[0], maps[1]
            map_out.append([self.image_padder.unpad(map1), self.image_padder.unpad(map2)])
        # map_out.append([self.image_padder.unpad(unet_out[0][0]), self.image_padder.unpad(unet_out[0][1])])
        # map_out.append([self.image_padder.unpad(unet_out[1][0]), self.image_padder.unpad(unet_out[1][1])])
        # map_out.append([image1,image2])

        return map_out, flow_predictions
        # return (events1, events2), flow_predictions
    
    @classmethod
    def demo(cls):
        im = torch.zeros((3, 5, 256, 256))
        config = ''
        net = MIMOUNET_RES_ERAFT(config,n_first_channels=5)
        net.change_imagesize((256,256))
        maps,out = net(im,im)
        tensor_tools.check_tensor(im, 'im')
        tensor_tools.check_tensor(out[0], 'out')
        tensor_tools.check_tensor(maps[-1][0], 'maps30')
        tensor_tools.check_tensor(maps[-1][1], 'maps31')
        tensor_tools.compute_model_size(net, im,im)

if __name__ == '__main__':
    # ERAFT.demo()
    MIMOUNET_RES_ERAFT.demo()
    # MIMOUNET_ERAFT.demo()



