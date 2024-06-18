import loguru
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum

from einops.layers.torch import Rearrange
from einops import rearrange

import sys
sys.path.append("/home/luoxinglong/unite_raft/model/flowformer")

from FlowFormer.common import FeedForward, pyramid_retrieve_tokens, sampler, sampler_gaussian_fix, retrieve_tokens, MultiHeadAttention, MLP
# from ..encoders import twins_svt_large_context, twins_svt_large
from position_encoding import PositionEncodingSine, LinearPositionEncoding
# from .twins import PosConv
from FlowFormer.LatentCostFormer.encoder import MemoryEncoder
from FlowFormer.LatentCostFormer.decoder import MemoryDecoder
from FlowFormer.LatentCostFormer.cnn import BasicEncoder

sys.path.append("/home/luoxinglong/unite_raft/model")
from model.MIMO_unet.mimo_unet import MIMOUNet_little as MIMOUNet
from model.sknet import SK

class FlowFormerRES(nn.Module):
    def __init__(self, cfg):
        super(FlowFormerRES, self).__init__()
        self.cfg = cfg

        self.in_channels = 5
        self.unet = MIMOUNet(self.in_channels, self.in_channels)
        self.unet_sk = SK()

        self.memory_encoder = MemoryEncoder(cfg)
        self.memory_decoder = MemoryDecoder(cfg)

        self.context_encoder = BasicEncoder(output_dim=256, norm_fn='instance')

    def change_imagesize(self, img_size):
        self.image_size = img_size


    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, events1, events2, output=None, flow_init=None):
        # Following https://github.com/princeton-vl/RAFT/
        # image1 = 2 * (image1 / 255.0) - 1.0
        # image2 = 2 * (image2 / 255.0) - 1.0

        events1 = events1.contiguous()
        events2 = events2.contiguous()

        unet_out = self.unet([events1, events2], with_res=False)
        image1, image2 = self.unet_sk([events1, events2], [unet_out[-1][0], unet_out[-1][1]])

        data = {}

        if self.cfg.context_concat:
            context = self.context_encoder(torch.cat([image1, image2], dim=1))
        else:
            context = self.context_encoder(image1)
            
        cost_memory = self.memory_encoder(image1, image2, data, context)

        flow_predictions = self.memory_decoder(cost_memory, context, data, flow_init=flow_init)

        map_out = []
        for maps in unet_out:
            map1, map2 = maps[0], maps[1]
            map_out.append([map1, map2])

        return map_out, flow_predictions
