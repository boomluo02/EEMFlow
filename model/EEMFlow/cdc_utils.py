import torch
import torch.nn as nn
import torch.nn.functional as F
from utils_luo.tools import tensor_tools
from einops import rearrange
from torch import einsum
import pdb

def conv(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, isReLU=True, if_IN=False, IN_affine=False, if_BN=False):
    if isReLU:
        if if_IN:
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                          padding=((kernel_size - 1) * dilation) // 2, bias=True),
                nn.LeakyReLU(0.1, inplace=True),
                nn.InstanceNorm2d(out_planes, affine=IN_affine)
            )
        elif if_BN:
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                          padding=((kernel_size - 1) * dilation) // 2, bias=True),
                nn.LeakyReLU(0.1, inplace=True),
                nn.BatchNorm2d(out_planes, affine=IN_affine)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                          padding=((kernel_size - 1) * dilation) // 2, bias=True),
                nn.LeakyReLU(0.1, inplace=True)
            )
    else:
        if if_IN:
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                          padding=((kernel_size - 1) * dilation) // 2, bias=True),
                nn.InstanceNorm2d(out_planes, affine=IN_affine)
            )
        elif if_BN:
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                          padding=((kernel_size - 1) * dilation) // 2, bias=True),
                nn.BatchNorm2d(out_planes, affine=IN_affine)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                          padding=((kernel_size - 1) * dilation) // 2, bias=True)
            )

class WarpingLayer_no_div(nn.Module):

    def __init__(self):
        super(WarpingLayer_no_div, self).__init__()

    def forward(self, x, flow):
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()
        if x.is_cuda:
            grid = grid.cuda()
        # print(grid.shape,flo.shape,'...')
        vgrid = grid + flow
        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
        vgrid = vgrid.permute(0, 2, 3, 1)  # B H,W,C
        x_warp = F.grid_sample(x, vgrid, padding_mode='zeros')
        if x.is_cuda:
            mask = torch.ones(x.size(), requires_grad=False).cuda()
        else:
            mask = torch.ones(x.size(), requires_grad=False)  # .cuda()
        mask = F.grid_sample(mask, vgrid)
        mask = (mask >= 1.0).float()
        return x_warp * mask

def upsample2d_flow_as(inputs, target_as, mode="bilinear", if_rate=False):
    _, _, h, w = target_as.shape
    res = F.interpolate(inputs, [h, w], mode=mode, align_corners=True)
    if if_rate:
        b, c, h_, w_ = inputs.shape
        inputs[:, 0, :, :] *= (w / w_)
        inputs[:, 1, :, :] *= (h / h_)
        # pdb.set_trace()
        u_scale = (w / w_)
        v_scale = (h / h_)
        u, v = res.chunk(2, dim=1)
        u *= u_scale
        v *= v_scale
        res = torch.cat([u, v], dim=1)

        # u_scale = (w / w_)
        # v_scale = (h / h_)
        # u_scale_t = u_scale*torch.ones(b, 1, h, w)
        # v_scale_t = v_scale*torch.ones(b, 1, h, w)
        # print(inputs)
        # uv_sacle = torch.cat([u_scale_t, v_scale_t], dim=1).to(inputs.device)
        # res = torch.mul(res, uv_sacle)

    return res

class cdc_model(nn.Module):
    def __init__(self):
        super(cdc_model, self).__init__()

        class FlowEstimatorDense_temp(nn.Module):

            def __init__(self, ch_in, f_channels=(128, 128, 96, 64, 32), ch_out=2):
                super(FlowEstimatorDense_temp, self).__init__()
                N = 0
                ind = 0
                N += ch_in
                self.conv1 = conv(N, f_channels[ind])
                N += f_channels[ind]

                ind += 1
                self.conv2 = conv(N, f_channels[ind])
                N += f_channels[ind]

                ind += 1
                self.conv3 = conv(N, f_channels[ind])
                N += f_channels[ind]

                ind += 1
                self.conv4 = conv(N, f_channels[ind])
                N += f_channels[ind]

                ind += 1
                self.conv5 = conv(N, f_channels[ind])
                N += f_channels[ind]
                self.num_feature_channel = N
                ind += 1
                self.conv_last = conv(N, ch_out, isReLU=False)

            def forward(self, x):
                x1 = torch.cat([self.conv1(x), x], dim=1)
                x2 = torch.cat([self.conv2(x1), x1], dim=1)
                x3 = torch.cat([self.conv3(x2), x2], dim=1)
                x4 = torch.cat([self.conv4(x3), x3], dim=1)
                x5 = torch.cat([self.conv5(x4), x4], dim=1)
                x_out = self.conv_last(x5)
                return x5, x_out

        f_channels_es = (32, 32, 32, 16, 8)
        in_C = 64
        self.warping_layer = WarpingLayer_no_div()
        self.dense_estimator_mask = FlowEstimatorDense_temp(in_C, f_channels=f_channels_es, ch_out=3)
        self.upsample_output_conv = nn.Sequential(conv(3, 16, kernel_size=3, stride=1, dilation=1),
                                                    conv(16, 16, stride=2),
                                                    conv(16, 32, kernel_size=3, stride=1, dilation=1),
                                                    conv(32, 32, stride=2), )

    def forward(self, flow_init, feature_1, feature_2, output_level_flow=None):
        n, c, h, w = flow_init.shape
        n_f, c_f, h_f, w_f = feature_1.shape
        if h != h_f or w != w_f:
            flow_init = upsample2d_flow_as(flow_init, feature_1, mode="bilinear", if_rate=True)
        feature_2_warp = self.warping_layer(feature_2, flow_init)
        input_feature = torch.cat((feature_1, feature_2_warp), dim=1)
        feature, x_out = self.dense_estimator_mask(input_feature)
        inter_flow = x_out[:, :2, :, :]
        inter_mask = x_out[:, 2, :, :]
        inter_mask = torch.unsqueeze(inter_mask, 1)
        inter_mask = torch.sigmoid(inter_mask)
        n_, c_, h_, w_ = inter_flow.shape
        if output_level_flow is not None:
            inter_flow = upsample2d_flow_as(inter_flow, output_level_flow, mode="bilinear", if_rate=True)
            inter_mask = upsample2d_flow_as(inter_mask, output_level_flow, mode="bilinear")
            flow_init = output_level_flow
        flow_up = tensor_tools.torch_warp(flow_init, inter_flow) * (1 - inter_mask) + flow_init * inter_mask
        return flow_init, flow_up, inter_flow, inter_mask

    def output_conv(self, x):
        return self.upsample_output_conv(x)

class CFP(nn.Module):
    def __init__(self, c_dim):
        super(CFP, self).__init__()

        self.self_corr = nn.Linear(c_dim, c_dim)

    def forward(self, inp, inter_mask, flow_init, thres=0.4):

        batch, ch, ht, wd = inp.shape
        inp = inp.reshape(batch, ch, ht * wd).permute(0, 2, 1).contiguous()
        inp = self.self_corr(inp)
        self_corr = (inp * (ch ** -0.5)) @ inp.transpose(1, 2)

        # inter_mask = inter_mask.reshape(batch, 1, ht * wd).permute(0, 2, 1).contiguous()
        # mask_corr = (inter_mask * (ch ** -0.5)) @ inter_mask.transpose(1, 2)

        # mask_corr = torch.softmax(mask_corr, dim=-1)

        # confidence = torch.zeros_like(mask_corr)  
        # confidence[mask_corr <= thres] = -100
        
        # self_corr = self_corr + confidence  

        flow_attn = torch.softmax(self_corr, dim=-1)

        flow_init_ = flow_init.reshape(batch, 2, ht * wd).permute(0, 2, 1).contiguous()

        flow_add = flow_attn @ flow_init_
        flow_add = flow_add.reshape(batch, ht, wd, 2).permute(0, 3, 1, 2).contiguous()

        return flow_add

class cdc_cfp_model(nn.Module):
    def __init__(self):
        super(cdc_cfp_model, self).__init__()

        class FlowEstimatorDense_temp(nn.Module):

            def __init__(self, ch_in, f_channels=(128, 128, 96, 64, 32), ch_out=2):
                super(FlowEstimatorDense_temp, self).__init__()
                N = 0
                ind = 0
                N += ch_in
                self.conv1 = conv(N, f_channels[ind])
                N += f_channels[ind]

                ind += 1
                self.conv2 = conv(N, f_channels[ind])
                N += f_channels[ind]

                ind += 1
                self.conv3 = conv(N, f_channels[ind])
                N += f_channels[ind]

                ind += 1
                self.conv4 = conv(N, f_channels[ind])
                N += f_channels[ind]

                ind += 1
                self.conv5 = conv(N, f_channels[ind])
                N += f_channels[ind]
                self.num_feature_channel = N
                ind += 1
                self.conv_last = conv(N, ch_out, isReLU=False)

            def forward(self, x):
                x1 = torch.cat([self.conv1(x), x], dim=1)
                x2 = torch.cat([self.conv2(x1), x1], dim=1)
                x3 = torch.cat([self.conv3(x2), x2], dim=1)
                x4 = torch.cat([self.conv4(x3), x3], dim=1)
                x5 = torch.cat([self.conv5(x4), x4], dim=1)
                x_out = self.conv_last(x5)
                return x5, x_out

        f_channels_es = (32, 32, 32, 16, 8)
        in_C = 64
        self.warping_layer = WarpingLayer_no_div()
        self.dense_estimator_mask = FlowEstimatorDense_temp(in_C, f_channels=f_channels_es, ch_out=3)
        self.upsample_output_conv = nn.Sequential(conv(3, 16, kernel_size=3, stride=1, dilation=1),
                                                    conv(16, 16, stride=2),
                                                    conv(16, 32, kernel_size=3, stride=1, dilation=1),
                                                    conv(32, 32, stride=2), )
        self.cfp = CFP(32)

    def forward(self, flow_init, feature_1, feature_2, output_level_flow=None):
        n, c, h, w = flow_init.shape
        n_f, c_f, h_f, w_f = feature_1.shape
        if h != h_f or w != w_f:
            flow_init = upsample2d_flow_as(flow_init, feature_1, mode="bilinear", if_rate=True)
        feature_2_warp = self.warping_layer(feature_2, flow_init)
        input_feature = torch.cat((feature_1, feature_2_warp), dim=1)
        feature, x_out = self.dense_estimator_mask(input_feature)
        inter_flow = x_out[:, :2, :, :]
        inter_mask = x_out[:, 2, :, :]
        inter_mask = torch.unsqueeze(inter_mask, 1)
        inter_mask = torch.sigmoid(inter_mask)

        flow_add = self.cfp(feature_2, 1-inter_mask, flow_init)

        n_, c_, h_, w_ = inter_flow.shape
        if output_level_flow is not None:
            inter_flow = upsample2d_flow_as(inter_flow, output_level_flow, mode="bilinear", if_rate=True)
            inter_mask = upsample2d_flow_as(inter_mask, output_level_flow, mode="bilinear")
            flow_init = output_level_flow

        flow_up = (tensor_tools.torch_warp(flow_init, inter_flow) + flow_add) * 0.5 * (1 - inter_mask) + flow_init * inter_mask

        return flow_init, flow_up, inter_flow, inter_mask

    def output_conv(self, x):
        return self.upsample_output_conv(x)