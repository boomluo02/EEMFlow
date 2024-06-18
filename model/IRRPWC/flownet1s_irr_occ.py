from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
from .flownet_modules import conv, deconv
from .flownet_modules import concatenate_as, upsample2d_as
from .flownet_modules import initialize_msra
from .flownet_modules import WarpingLayer

class FlowNetS(nn.Module):
    def __init__(self, args):
        super(FlowNetS, self).__init__()

        def make_conv(in_planes, out_planes, kernel_size, stride):
            pad = kernel_size // 2
            return conv(in_planes, out_planes, kernel_size=kernel_size,
                        stride=stride, pad=pad, nonlinear=True, bias=True)

        self._conv3_1 = make_conv( 256,  256, kernel_size=3, stride=1)
        self._conv4   = make_conv( 256,  512, kernel_size=3, stride=2)
        self._conv4_1 = make_conv( 512,  512, kernel_size=3, stride=1)
        self._conv5   = make_conv( 512,  512, kernel_size=3, stride=2)
        self._conv5_1 = make_conv( 512,  512, kernel_size=3, stride=1)
        self._conv6   = make_conv( 512, 1024, kernel_size=3, stride=2)
        self._conv6_1 = make_conv(1024, 1024, kernel_size=3, stride=1)

        def make_deconv(in_planes, out_planes):
            return deconv(in_planes, out_planes, kernel_size=4, stride=2, pad=1,
                          nonlinear=True, bias=False)

        self._deconv5 = make_deconv(1024    , 512)
        self._deconv4 = make_deconv(1024 + 2, 256)
        self._deconv3 = make_deconv( 768 + 2, 128)
        self._deconv2 = make_deconv( 384 + 2,  64)

        self._deconv_occ5 = make_deconv(1024    , 512)
        self._deconv_occ4 = make_deconv(1024 + 1, 256)
        self._deconv_occ3 = make_deconv( 768 + 1, 128)
        self._deconv_occ2 = make_deconv( 384 + 1,  64)

        def make_predict(in_planes, out_planes):
            return conv(in_planes, out_planes, kernel_size=3, stride=1, pad=1,
                        nonlinear=False, bias=True)

        self._predict_flow6 = make_predict(1024    , 2)
        self._predict_flow5 = make_predict(1024 + 2, 2)
        self._predict_flow4 = make_predict( 768 + 2, 2)
        self._predict_flow3 = make_predict( 384 + 2, 2)
        self._predict_flow2 = make_predict( 128 + 2, 2)

        self._predict_occ6 = make_predict(1024    , 1)
        self._predict_occ5 = make_predict(1024 + 1, 1)
        self._predict_occ4 = make_predict( 768 + 1, 1)
        self._predict_occ3 = make_predict( 384 + 1, 1)
        self._predict_occ2 = make_predict( 128 + 1, 1)

        def make_upsample(in_planes, out_planes):
            return deconv(in_planes, out_planes, kernel_size=4, stride=2, pad=1,
                          nonlinear=False, bias=False)

        self._upsample_flow6_to_5 = make_upsample(2, 2)
        self._upsample_flow5_to_4 = make_upsample(2, 2)
        self._upsample_flow4_to_3 = make_upsample(2, 2)
        self._upsample_flow3_to_2 = make_upsample(2, 2)

        self._upsample_occ6_to_5 = make_upsample(1, 1)
        self._upsample_occ5_to_4 = make_upsample(1, 1)
        self._upsample_occ4_to_3 = make_upsample(1, 1)
        self._upsample_occ3_to_2 = make_upsample(1, 1)

    def forward(self, conv2_im1, conv3_im1, conv3_im2):

        conv_concat3 = torch.cat((conv3_im1, conv3_im2), dim=1)

        conv3_1 = self._conv3_1(conv_concat3)
        conv4_1 = self._conv4_1(self._conv4(conv3_1))
        conv5_1 = self._conv5_1(self._conv5(conv4_1))
        conv6_1 = self._conv6_1(self._conv6(conv5_1))

        # Flow Decoder
        predict_flow6        = self._predict_flow6(conv6_1)

        upsampled_flow6_to_5 = self._upsample_flow6_to_5(predict_flow6)
        deconv5              = self._deconv5(conv6_1)
        concat5              = concatenate_as((conv5_1, deconv5, upsampled_flow6_to_5), conv5_1, dim=1)
        predict_flow5        = self._predict_flow5(concat5)

        upsampled_flow5_to_4 = self._upsample_flow5_to_4(predict_flow5)
        deconv4              = self._deconv4(concat5)
        concat4              = concatenate_as((conv4_1, deconv4, upsampled_flow5_to_4), conv4_1, dim=1)
        predict_flow4        = self._predict_flow4(concat4)

        upsampled_flow4_to_3 = self._upsample_flow4_to_3(predict_flow4)
        deconv3              = self._deconv3(concat4)
        concat3              = concatenate_as((conv3_1, deconv3, upsampled_flow4_to_3), conv3_1, dim=1)
        predict_flow3        = self._predict_flow3(concat3)

        upsampled_flow3_to_2 = self._upsample_flow3_to_2(predict_flow3)
        deconv2              = self._deconv2(concat3)
        concat2              = concatenate_as((conv2_im1, deconv2, upsampled_flow3_to_2), conv2_im1, dim=1)
        predict_flow2        = self._predict_flow2(concat2)

        # Occ Decoder
        predict_occ6 = self._predict_occ6(conv6_1)

        upsampled_occ6_to_5 = self._upsample_occ6_to_5(predict_occ6)
        deconv_occ5         = self._deconv_occ5(conv6_1)
        concat_occ5         = concatenate_as((conv5_1, deconv_occ5, upsampled_occ6_to_5), conv5_1, dim=1)
        predict_occ5        = self._predict_occ5(concat_occ5)

        upsampled_occ5_to_4 = self._upsample_occ5_to_4(predict_occ5)
        deconv_occ4         = self._deconv_occ4(concat_occ5)
        concat_occ4         = concatenate_as((conv4_1, deconv_occ4, upsampled_occ5_to_4), conv4_1, dim=1)
        predict_occ4        = self._predict_occ4(concat_occ4)

        upsampled_occ4_to_3 = self._upsample_occ4_to_3(predict_occ4)
        deconv_occ3         = self._deconv_occ3(concat_occ4)
        concat_occ3         = concatenate_as((conv3_1, deconv_occ3, upsampled_occ4_to_3), conv3_1, dim=1)
        predict_occ3        = self._predict_occ3(concat_occ3)

        upsampled_occ3_to_2 = self._upsample_occ3_to_2(predict_occ3)
        deconv_occ2         = self._deconv_occ2(concat_occ3)
        concat_occ2         = concatenate_as((conv2_im1, deconv_occ2, upsampled_occ3_to_2), conv2_im1, dim=1)
        predict_occ2        = self._predict_occ2(concat_occ2)

        return predict_flow2, predict_flow3, predict_flow4, predict_flow5, predict_flow6, predict_occ2, predict_occ3, predict_occ4, predict_occ5, predict_occ6


class FlowNet1S(nn.Module):
    def __init__(self, args, div_flow=0.05):
        super(FlowNet1S, self).__init__()
        self._flownets = FlowNetS(args)
        self._warping_layer = WarpingLayer()
        self._div_flow = div_flow
        self._num_iters = args.num_iters

        def make_conv(in_planes, out_planes, kernel_size, stride):
            pad = kernel_size // 2
            return conv(in_planes, out_planes, kernel_size=kernel_size,
                        stride=stride, pad=pad, nonlinear=True, bias=True)

        self._conv1   = make_conv(   3,   32, kernel_size=7, stride=2)
        self._conv2   = make_conv(  32,   64, kernel_size=5, stride=2)
        self._conv3   = make_conv(  64,  128, kernel_size=5, stride=2)

        initialize_msra(self.modules())

    def forward(self, input_dict):
        im1 = input_dict['input1']
        im2 = input_dict['input2']

        conv1_im1 = self._conv1(im1)
        conv2_im1 = self._conv2(conv1_im1)
        conv3_im1 = self._conv3(conv2_im1)

        conv1_im2 = self._conv1(im2)
        conv2_im2 = self._conv2(conv1_im2)
        conv3_im2 = self._conv3(conv2_im2)
        conv3_im2_wp = conv3_im2

        output_dict = {}
        output_dict['flow2'] = []
        output_dict['flow3'] = []
        output_dict['flow4'] = []
        output_dict['flow5'] = []
        output_dict['flow6'] = []
        output_dict['occ2'] = []
        output_dict['occ3'] = []
        output_dict['occ4'] = []
        output_dict['occ5'] = []
        output_dict['occ6'] = []

        _, _, height_im, width_im = im1.size()
        
        # for iterative
        for ii in range(0, self._num_iters):
            flow2, flow3, flow4, flow5, flow6, occ2, occ3, occ4, occ5, occ6 = self._flownets(conv2_im1, conv3_im1, conv3_im2_wp)

            if ii == 0:
                output_dict['flow2'].append(flow2)
                output_dict['flow3'].append(flow3)
                output_dict['flow4'].append(flow4)
                output_dict['flow5'].append(flow5)
                output_dict['flow6'].append(flow6)
                output_dict['occ2'].append(occ2)
                output_dict['occ3'].append(occ3)
                output_dict['occ4'].append(occ4)
                output_dict['occ5'].append(occ5)
                output_dict['occ6'].append(occ6)
            else:
                output_dict['flow2'].append(flow2 + output_dict['flow2'][ii - 1])
                output_dict['flow3'].append(flow3 + output_dict['flow3'][ii - 1])
                output_dict['flow4'].append(flow4 + output_dict['flow4'][ii - 1])
                output_dict['flow5'].append(flow5 + output_dict['flow5'][ii - 1])
                output_dict['flow6'].append(flow6 + output_dict['flow6'][ii - 1])
                output_dict['occ2'].append(occ2 + output_dict['occ2'][ii - 1])
                output_dict['occ3'].append(occ3 + output_dict['occ3'][ii - 1])
                output_dict['occ4'].append(occ4 + output_dict['occ4'][ii - 1])
                output_dict['occ5'].append(occ5 + output_dict['occ5'][ii - 1])
                output_dict['occ6'].append(occ6 + output_dict['occ6'][ii - 1])

            if ii < (self._num_iters - 1):
                up_flow = upsample2d_as(output_dict['flow2'][ii], conv3_im2, mode="bilinear")
                conv3_im2_wp = self._warping_layer(conv3_im2, up_flow, height_im, width_im, self._div_flow)     

        if self.training:
            return output_dict
        else:
            output_dict_eval = {}
            up_flow_final = upsample2d_as(output_dict['flow2'][self._num_iters - 1], im1, mode="bilinear")  
            up_occ_final = upsample2d_as(output_dict['occ2'][self._num_iters - 1], im1, mode="bilinear")
            output_dict_eval['flow1'] = (1.0 / self._div_flow) * up_flow_final
            output_dict_eval['occ1'] = up_occ_final
            return output_dict_eval