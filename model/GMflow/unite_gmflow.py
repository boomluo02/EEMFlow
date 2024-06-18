import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import CNNEncoder
from .transformer import FeatureTransformer, FeatureFlowAttention
from .matching import global_correlation_softmax, local_correlation_softmax
from .geometry import flow_warp
from .utils import normalize_img, feature_add_position
from utils.image_utils import InputPadder

import sys
sys.path.append('/home/luoxinglong/unite_raft')
from model.MIMO_unet.mimo_unet import MIMOUNet_little as MIMOUNet
from model.sknet import SK

class GMFlowRES(nn.Module):
    def __init__(self, config, n_first_channels):
        super(GMFlowRES, self).__init__()

        self.num_scales = config["num_scales"]
        self.feature_channels = config["feature_channels"]
        self.upsample_factor = config["upsample_factor"]
        self.attention_type = config["attention_type"]
        self.num_transformer_layers = config["num_transformer_layers"]
        
        self.image_size = config["train_img_size"]
        # self.image_padder = InputPadder(config["img_size"], mode='chairs')

        # CNN backbone
        self.backbone = CNNEncoder(input_channel = n_first_channels, output_dim=config["feature_channels"], num_output_scales=config["num_scales"])

        # Transformer
        self.transformer = FeatureTransformer(num_layers=config["num_transformer_layers"],
                                              d_model=config["feature_channels"],
                                              nhead=config["num_head"],
                                              attention_type=config["attention_type"],
                                              ffn_dim_expansion=config["ffn_dim_expansion"],
                                              )

        # flow propagation with self-attn
        self.feature_flow_attn = FeatureFlowAttention(in_channels=config["feature_channels"])

        # convex upsampling: concat feature0 and flow as input
        self.upsampler = nn.Sequential(nn.Conv2d(2 + config["feature_channels"], 256, 3, 1, 1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, config["upsample_factor"] ** 2 * 9, 1, 1, 0))

        self.unet = MIMOUNet(n_first_channels, n_first_channels)
        self.unet_sk = SK()

    def change_imagesize(self, img_size):
        self.image_size = img_size
        self.image_padder = InputPadder(img_size, mode='chairs')
    
    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def extract_feature(self, img0, img1):
        concat = torch.cat((img0, img1), dim=0)  # [2B, C, H, W]
        features = self.backbone(concat)  # list of [2B, C, H, W], resolution from high to low

        # reverse: resolution from low to high
        features = features[::-1]

        feature0, feature1 = [], []

        for i in range(len(features)):
            feature = features[i]
            chunks = torch.chunk(feature, 2, 0)  # tuple
            feature0.append(chunks[0])
            feature1.append(chunks[1])

        return feature0, feature1

    def upsample_flow(self, flow, feature, bilinear=False, upsample_factor=8,
                      ):
        if bilinear:
            up_flow = F.interpolate(flow, scale_factor=upsample_factor,
                                    mode='bilinear', align_corners=True) * upsample_factor

        else:
            # convex upsampling
            concat = torch.cat((flow, feature), dim=1)

            mask = self.upsampler(concat)
            b, flow_channel, h, w = flow.shape
            mask = mask.view(b, 1, 9, self.upsample_factor, self.upsample_factor, h, w)  # [B, 1, 9, K, K, H, W]
            mask = torch.softmax(mask, dim=2)

            up_flow = F.unfold(self.upsample_factor * flow, [3, 3], padding=1)
            up_flow = up_flow.view(b, flow_channel, 9, 1, 1, h, w)  # [B, 2, 9, 1, 1, H, W]

            up_flow = torch.sum(mask * up_flow, dim=2)  # [B, 2, K, K, H, W]
            up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)  # [B, 2, K, H, K, W]
            up_flow = up_flow.reshape(b, flow_channel, self.upsample_factor * h,
                                      self.upsample_factor * w)  # [B, 2, K*H, K*W]

        return up_flow

    def forward(self, events1, events2,
                attn_splits_list=None,
                corr_radius_list=None,
                prop_radius_list=None,
                pred_bidir_flow=False,
                **kwargs,
                ):

        results_dict = {}
        flow_preds = []

        # img0, img1 = normalize_img(img0, img1)  # [B, 3, H, W]
        events1, events2 = self.image_padder.pad(events1, events2)

        events1 = events1.contiguous()
        events2 = events2.contiguous()

        unet_out = self.unet([events1, events2], with_res=False)
        img0, img1 = self.unet_sk([events1, events2], [unet_out[-1][0], unet_out[-1][1]])

        # resolution low to high
        feature0_list, feature1_list = self.extract_feature(img0, img1)  # list of features

        flow = None

        assert len(attn_splits_list) == len(corr_radius_list) == len(prop_radius_list) == self.num_scales

        for scale_idx in range(self.num_scales):
            feature0, feature1 = feature0_list[scale_idx], feature1_list[scale_idx]

            if pred_bidir_flow and scale_idx > 0:
                # predicting bidirectional flow with refinement
                feature0, feature1 = torch.cat((feature0, feature1), dim=0), torch.cat((feature1, feature0), dim=0)

            upsample_factor = self.upsample_factor * (2 ** (self.num_scales - 1 - scale_idx))

            if scale_idx > 0:
                flow = F.interpolate(flow, scale_factor=2, mode='bilinear', align_corners=True) * 2

            if flow is not None:
                flow = flow.detach()
                feature1 = flow_warp(feature1, flow)  # [B, C, H, W]

            attn_splits = attn_splits_list[scale_idx]
            corr_radius = corr_radius_list[scale_idx]
            prop_radius = prop_radius_list[scale_idx]

            # add position to features
            feature0, feature1 = feature_add_position(feature0, feature1, attn_splits, self.feature_channels)

            # Transformer
            feature0, feature1 = self.transformer(feature0, feature1, attn_num_splits=attn_splits)

            # correlation and softmax
            if corr_radius == -1:  # global matching
                flow_pred = global_correlation_softmax(feature0, feature1, pred_bidir_flow)[0]
            else:  # local matching
                flow_pred = local_correlation_softmax(feature0, feature1, corr_radius)[0]

            # flow or residual flow
            flow = flow + flow_pred if flow is not None else flow_pred

            # upsample to the original resolution for supervison
            if self.training:  # only need to upsample intermediate flow predictions at training time
                flow_bilinear = self.upsample_flow(flow, None, bilinear=True, upsample_factor=upsample_factor)
                flow_preds.append(flow_bilinear)

            # flow propagation with self-attn
            if pred_bidir_flow and scale_idx == 0:
                feature0 = torch.cat((feature0, feature1), dim=0)  # [2*B, C, H, W] for propagation
            flow = self.feature_flow_attn(feature0, flow.detach(),
                                          local_window_attn=prop_radius > 0,
                                          local_window_radius=prop_radius)

            # bilinear upsampling at training time except the last one
            if self.training and scale_idx < self.num_scales - 1:
                flow_up = self.upsample_flow(flow, feature0, bilinear=True, upsample_factor=upsample_factor)
                flow_preds.append(self.image_padder.unpad(flow_up))

            if scale_idx == self.num_scales - 1:
                flow_up = self.upsample_flow(flow, feature0)
                flow_preds.append(self.image_padder.unpad(flow_up))
        
        map_out = []
        for maps in unet_out:
            map1, map2 = maps[0], maps[1]
            map_out.append([map1, map2])

        return map_out, flow_preds
