import loguru
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum

from einops.layers.torch import Rearrange
from einops import rearrange

from ..common import FeedForward, pyramid_retrieve_tokens, sampler, sampler_gaussian_fix, retrieve_tokens, MultiHeadAttention, MLP
# from ..encoders import twins_svt_large_context, twins_svt_large
from ...position_encoding import PositionEncodingSine, LinearPositionEncoding
# from .twins import PosConv
from .encoder import MemoryEncoder
from .decoder import MemoryDecoder
from .cnn import BasicEncoder
from utils.image_utils import ImagePadder, InputPadder

class FlowFormer(nn.Module):
    def __init__(self, cfg):
        super(FlowFormer, self).__init__()
        self.cfg = cfg

        self.memory_encoder = MemoryEncoder(cfg)
        self.memory_decoder = MemoryDecoder(cfg)
        # if cfg.cnet == 'twins':
        #     self.context_encoder = twins_svt_large(pretrained=self.cfg.pretrain)
        # elif cfg.cnet == 'basicencoder':
        #     self.context_encoder = BasicEncoder(output_dim=256, norm_fn='instance')

        self.context_encoder = BasicEncoder(output_dim=256, norm_fn='instance')

    def change_imagesize(self, img_size):
        self.image_size = img_size
        self.image_padder = InputPadder(img_size, mode='chairs')
    
    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, events1, events2, output=None, flow_init=None):
        # Following https://github.com/princeton-vl/RAFT/
        # image1 = 2 * (image1 / 255.0) - 1.0
        # image2 = 2 * (image2 / 255.0) - 1.0
        image1, image2 = self.image_padder.pad(events1, events2)
        image1 = image1.contiguous()
        image2 = image2.contiguous()

        data = {}

        if self.cfg.context_concat:
            context = self.context_encoder(torch.cat([image1, image2], dim=1))
        else:
            context = self.context_encoder(image1)
            
        cost_memory = self.memory_encoder(image1, image2, data, context)

        flow_predictions = self.memory_decoder(cost_memory, context, data, flow_init=flow_init)
        unpad_flow_predictions = []
        for flow_pred in flow_predictions:
            unpad_flow_predictions.append(self.image_padder.unpad(flow_pred))

        return (events1, events2), unpad_flow_predictions
