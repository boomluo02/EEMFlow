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

from utils_luo.tools import tools, tensor_tools

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
    def __init__(self, config, n_first_channels=5):
        # args:
        super(ERAFT, self).__init__()
        args = get_args()
        self.args = args
        # self.image_padder = ImagePadder(min_size=32)
        # self.image_size = config["train_img_size"]
        # self.image_padder = InputPadder(config["train_img_size"], mode='chairs')
        # self.subtype = config['subtype'].lower()
        # assert (self.subtype == 'standard' or self.subtype == 'warm_start')

        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        args.corr_levels = 4
        args.corr_radius = 4

        # feature network, context network, and update block
        self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=0,
                                    n_first_channels=n_first_channels)
        self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=0,
                                    n_first_channels=n_first_channels)
        # self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='instance', dropout=0,
        #                             n_first_channels=n_first_channels)
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
        """ Estimate optical flow between pair of frames """
        # image1 = 2 * (image1) - 1.0
        # image2 = 2 * (image2) - 1.0

        # Pad Image (for flawless up&downsampling)
        # image1 = self.image_padder.pad(image1)
        # image2 = self.image_padder.pad(image2)
        
        image1, image2 = self.image_padder.pad(events1, events2)

        image1 = image1.contiguous()
        image2 = image2.contiguous()

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
            # if (self.subtype == 'standard' or self.subtype == 'warm_start'):
            cnet = self.cnet(image1)
            # else:
            #     raise Exception
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

        return (events1, events2), flow_predictions

    @classmethod
    def demo(cls):
        h,w = 1080, 1438
        im = torch.ones((1, 5, h, w))
        config = ''
        net = ERAFT(config,n_first_channels=5)
        im = im.cuda()
        net = net.cuda()

        repeat_num = 10
        import time
        time_start = time.time()
       
        for i in range(repeat_num):
            net.change_imagesize((h,w))
            with torch.no_grad():
                _,out = net(im,im)

        time_end = time.time()
        print("Process time is {:f}".format((time_end-time_start)/repeat_num))

        tensor_tools.check_tensor(im, 'im')
        tensor_tools.check_tensor(out[0], 'out')
        # tensor_tools.compute_model_size(net, im,im)


def time_eval(model, batch_size=1, iters=32):  # 12
    import time

    h,w = 512, 960

    model = model.eval()
    model = model.cuda()
    images = torch.randn(batch_size, 5, h,w).cuda()
    first = 0
    time_train = []
    for ii in range(100):
        start_time = time.time()
        model.change_imagesize((h,w))
        # with torch.no_grad():
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
    config = ''
    model = ERAFT(config, n_first_channels=5)
    time_eval(model, batch_size=1)
