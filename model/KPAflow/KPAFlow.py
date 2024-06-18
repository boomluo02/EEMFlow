import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
sys.path.append("/home/luoxinglong/meshflow/model/KPAflow")
from extractor import BasicEncoder, SmallEncoder
from corr import CorrBlock, AlternateCorrBlock
from kpautils.utils import bilinear_sampler, coords_grid, upflow8

from module import KPAEnc, KPAFlowDec, KPAEnc

sys.path.append('/home/luoxinglong/unite_raft')
from utils_luo.tools import tools, tensor_tools
from model.MIMO_unet.mimo_unet import MIMOUNet_little as MIMOUNet
from model.sknet import SK

from argparse import Namespace
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
                     clip=1.0,
                     dataset=[])
    return args


class KPAFlow(nn.Module):
    def __init__(self, config, n_first_channels):
        super().__init__()
        print('----- Model: KPA-Flow -----')

        args = get_args()
        self.args = args


        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        args.corr_levels = 4
        args.corr_radius = 4

        if 'dropout' not in self.args:
            self.args.dropout = 0

        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout, n_first_channels=n_first_channels)        
        self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout, n_first_channels=n_first_channels)
        self.update_block = KPAFlowDec(self.args, chnn=hdim)

        self.sc = 13
        self.trans = KPAEnc(args, 256, self.sc)
        self.zero = nn.Parameter(torch.zeros(12), requires_grad=False)

    def change_imagesize(self, img_size):
        self.image_size = img_size

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8, device=img.device)
        coords1 = coords_grid(N, H//8, W//8, device=img.device)

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
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)  # [N, 2, H, 8, W, 8]
        return up_flow.reshape(N, 2, 8*H, 8*W)

    def forward(self, events1, events2, iters=12, flow_init=None, upsample=True, test_mode=False, gt=None):
        """ Estimate optical flow between pair of frames """

        # image1 = 2 * (image1 / 255.0) - 1.0
        # image2 = 2 * (image2 / 255.0) - 1.0

        image1 = events1.contiguous()
        image2 = events2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])        

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

        fmap1 = self.trans(fmap1)
        fmap2 = self.trans(fmap2)

        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        else:
            corr_fn = CorrBlock(fmap1, fmap2, num_levels=self.args.corr_levels, radius=self.args.corr_radius)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        # flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow, itr)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            flow = coords1 - coords0

            # # upsample predictions
            # if up_mask is None:
            #     flow_up = upflow8(flow)
            # else:
            #     flow_up = self.upsample_flow(flow, up_mask)
        return flow
            # flow_predictions.append(flow_up)

        # if test_mode:
        #     return flow, flow_up

        # return flow_predictions, self.zero
        # return (events1, events2), flow_predictions
    
    @classmethod
    def demo(cls):
        im = torch.zeros((1, 5, 256, 256))
        config = dict({})
        net = KPAFlow(config)
        net.change_imagesize((256,256))
        _,out = net(im,im)
        tensor_tools.check_tensor(im, 'im')
        tensor_tools.check_tensor(out[0], 'out')
        tensor_tools.compute_model_size(net, im,im)


# class mimoKPAFlow(nn.Module):
#     def __init__(self, args, n_first_channels=5):
#         super().__init__()
#         print('----- Model: KPA-Flow -----')

#         args = get_args()
#         self.args = args

#         self.hidden_dim = hdim = 128
#         self.context_dim = cdim = 128
#         args.corr_levels = 4
#         args.corr_radius = 4

#         if 'dropout' not in self.args:
#             self.args.dropout = 0

#         if 'alternate_corr' not in self.args:
#             self.args.alternate_corr = False

#         self.unet = MIMOUNet(n_first_channels, n_first_channels)

#         self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)        
#         self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout)
#         self.update_block = KPAFlowDec(self.args, chnn=hdim)

#         self.sc = 13
#         self.trans = KPAEnc(args, 256, self.sc)
#         self.zero = nn.Parameter(torch.zeros(12), requires_grad=False)

#     def change_imagesize(self, img_size):
#         self.image_size = img_size

#     def freeze_bn(self):
#         for m in self.modules():
#             if isinstance(m, nn.BatchNorm2d):
#                 m.eval()

#     def initialize_flow(self, img):
#         """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
#         N, C, H, W = img.shape
#         coords0 = coords_grid(N, H//8, W//8, device=img.device)
#         coords1 = coords_grid(N, H//8, W//8, device=img.device)

#         # optical flow computed as difference: flow = coords1 - coords0
#         return coords0, coords1

#     def upsample_flow(self, flow, mask):
#         """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
#         N, _, H, W = flow.shape
#         mask = mask.view(N, 1, 9, 8, 8, H, W)
#         mask = torch.softmax(mask, dim=2)

#         up_flow = F.unfold(8 * flow, [3,3], padding=1)
#         up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

#         up_flow = torch.sum(mask * up_flow, dim=2)
#         up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)  # [N, 2, H, 8, W, 8]
#         return up_flow.reshape(N, 2, 8*H, 8*W)

#     def forward(self, events1, events2, iters=12, flow_init=None, upsample=True, test_mode=False, gt=None):
#         """ Estimate optical flow between pair of frames """

#         # image1 = 2 * (image1 / 255.0) - 1.0
#         # image2 = 2 * (image2 / 255.0) - 1.0

#         events1 = events1.contiguous()
#         events2 = events2.contiguous()

#         unet_out = self.unet([events1, events2], with_res=False)
#         image1, image2 = unet_out[-1][0], unet_out[-1][1]

#         hdim = self.hidden_dim
#         cdim = self.context_dim

#         # run the feature network
#         with autocast(enabled=self.args.mixed_precision):
#             fmap1, fmap2 = self.fnet([image1, image2])        

#         fmap1 = fmap1.float()
#         fmap2 = fmap2.float()

#         fmap1 = self.trans(fmap1)
#         fmap2 = self.trans(fmap2)

#         if self.args.alternate_corr:
#             corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
#         else:
#             corr_fn = CorrBlock(fmap1, fmap2, num_levels=self.args.corr_levels, radius=self.args.corr_radius)

#         # run the context network
#         with autocast(enabled=self.args.mixed_precision):
#             cnet = self.cnet(image1)
#             net, inp = torch.split(cnet, [hdim, cdim], dim=1)
#             net = torch.tanh(net)
#             inp = torch.relu(inp)

#         coords0, coords1 = self.initialize_flow(image1)

#         if flow_init is not None:
#             coords1 = coords1 + flow_init

#         flow_predictions = []
#         for itr in range(iters):
#             coords1 = coords1.detach()
#             corr = corr_fn(coords1) # index correlation volume

#             flow = coords1 - coords0
#             with autocast(enabled=self.args.mixed_precision):
#                 net, up_mask, delta_flow = self.update_block(net, inp, corr, flow, itr)

#             # F(t+1) = F(t) + \Delta(t)
#             coords1 = coords1 + delta_flow

#             flow = coords1 - coords0

#             # upsample predictions
#             if up_mask is None:
#                 flow_up = upflow8(flow)
#             else:
#                 flow_up = self.upsample_flow(flow, up_mask)
            
#             flow_predictions.append(flow_up)

#         if test_mode:
#             return flow, flow_up

#         return flow_predictions, self.zero
    
#     @classmethod
#     def demo(cls):
#         im = torch.zeros((1, 5, 256, 256))
#         config = dict({})
#         net = mimoKPAFlow(config)
#         net.change_imagesize((256,256))
#         _,out = net(im,im)
#         tensor_tools.check_tensor(im, 'im')
#         tensor_tools.check_tensor(out[0], 'out')
#         tensor_tools.compute_model_size(net, im,im)

    
# class mimo_res_KPAFlow(nn.Module):
#     def __init__(self, args, n_first_channels=5):
#         super().__init__()
#         print('----- Model: KPA-Flow -----')

#         args = get_args()
#         self.args = args

#         self.hidden_dim = hdim = 128
#         self.context_dim = cdim = 128
#         args.corr_levels = 4
#         args.corr_radius = 4

#         if 'dropout' not in self.args:
#             self.args.dropout = 0

#         if 'alternate_corr' not in self.args:
#             self.args.alternate_corr = False

#         self.unet = MIMOUNet(n_first_channels, n_first_channels)
#         self.unet_sk = SK()

#         self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)        
#         self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='instance', dropout=args.dropout)
#         self.update_block = KPAFlowDec(self.args, chnn=hdim)

#         self.sc = 13
#         self.trans = KPAEnc(args, 256, self.sc)
#         self.zero = nn.Parameter(torch.zeros(12), requires_grad=False)

#     def change_imagesize(self, img_size):
#         self.image_size = img_size

#     def freeze_bn(self):
#         for m in self.modules():
#             if isinstance(m, nn.BatchNorm2d):
#                 m.eval()

#     def initialize_flow(self, img):
#         """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
#         N, C, H, W = img.shape
#         coords0 = coords_grid(N, H//8, W//8, device=img.device)
#         coords1 = coords_grid(N, H//8, W//8, device=img.device)

#         # optical flow computed as difference: flow = coords1 - coords0
#         return coords0, coords1

#     def upsample_flow(self, flow, mask):
#         """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
#         N, _, H, W = flow.shape
#         mask = mask.view(N, 1, 9, 8, 8, H, W)
#         mask = torch.softmax(mask, dim=2)

#         up_flow = F.unfold(8 * flow, [3,3], padding=1)
#         up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

#         up_flow = torch.sum(mask * up_flow, dim=2)
#         up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)  # [N, 2, H, 8, W, 8]
#         return up_flow.reshape(N, 2, 8*H, 8*W)

#     def forward(self, events1, events2, iters=12, flow_init=None, upsample=True, test_mode=False, gt=None):
#         """ Estimate optical flow between pair of frames """

#         # image1 = 2 * (image1 / 255.0) - 1.0
#         # image2 = 2 * (image2 / 255.0) - 1.0

#         events1 = events1.contiguous()
#         events2 = events2.contiguous()

#         unet_out = self.unet([events1, events2], with_res=False)
#         # image1, image2 = unet_out[-1][0], unet_out[-1][1]
#         image1, image2 = self.unet_sk([events1, events2], [unet_out[-1][0], unet_out[-1][1]])

#         hdim = self.hidden_dim
#         cdim = self.context_dim

#         # run the feature network
#         with autocast(enabled=self.args.mixed_precision):
#             fmap1, fmap2 = self.fnet([image1, image2])        

#         fmap1 = fmap1.float()
#         fmap2 = fmap2.float()

#         fmap1 = self.trans(fmap1)
#         fmap2 = self.trans(fmap2)

#         if self.args.alternate_corr:
#             corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
#         else:
#             corr_fn = CorrBlock(fmap1, fmap2, num_levels=self.args.corr_levels, radius=self.args.corr_radius)

#         # run the context network
#         with autocast(enabled=self.args.mixed_precision):
#             cnet = self.cnet(image1)
#             net, inp = torch.split(cnet, [hdim, cdim], dim=1)
#             net = torch.tanh(net)
#             inp = torch.relu(inp)

#         coords0, coords1 = self.initialize_flow(image1)

#         if flow_init is not None:
#             coords1 = coords1 + flow_init

#         flow_predictions = []
#         for itr in range(iters):
#             coords1 = coords1.detach()
#             corr = corr_fn(coords1) # index correlation volume

#             flow = coords1 - coords0
#             with autocast(enabled=self.args.mixed_precision):
#                 net, up_mask, delta_flow = self.update_block(net, inp, corr, flow, itr)

#             # F(t+1) = F(t) + \Delta(t)
#             coords1 = coords1 + delta_flow

#             flow = coords1 - coords0

#             # upsample predictions
#             if up_mask is None:
#                 flow_up = upflow8(flow)
#             else:
#                 flow_up = self.upsample_flow(flow, up_mask)
            
#             flow_predictions.append(flow_up)

#         if test_mode:
#             return flow, flow_up

#         return flow_predictions, self.zero
    
#     @classmethod
#     def demo(cls):
#         im = torch.zeros((1, 5, 256, 256))
#         config = dict({})
#         net = mimo_res_KPAFlow(config)
#         net.change_imagesize((256,256))
#         _,out = net(im,im)
#         tensor_tools.check_tensor(im, 'im')
#         tensor_tools.check_tensor(out[0], 'out')
#         tensor_tools.compute_model_size(net, im,im)

def time_eval(model, batch_size=2, iters=32):  # 12
    import time

    h,w = 720, 1280

    model = model.eval()
    model = model.cuda()
    images = torch.randn(batch_size, 5, h,w).cuda()
    first = 0
    time_train = []
    for ii in range(100):
        start_time = time.time()
        model.change_imagesize((h,w))
        with torch.no_grad():
            outputs = model(images, images)

        torch.cuda.synchronize() 

        if first != 0:    #first run always takes some time for setup
            fwt = time.time() - start_time
            time_train.append(fwt)
            print ("Forward time per img (b=%d): %.4f (Mean: %.4f), FPS: %.1f" % (batch_size, fwt / batch_size, sum(time_train) / len(time_train) / batch_size, 1/(fwt/batch_size)))
        
        time.sleep(1) 
        first += 1

if __name__ == '__main__':
    # FastFlowNet.demo()
    # FastFlowNet_corr64.demo()
    config = ''
    model = KPAFlow(config, n_first_channels=5)
    time_eval(model)
