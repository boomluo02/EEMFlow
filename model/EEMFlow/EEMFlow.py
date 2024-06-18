import os,sys
current_path = os.path.dirname(os.path.abspath(__file__))
proc_path = current_path.rsplit("/",1)[0]
sys.path.append(current_path)
sys.path.append(proc_path)
import torch
import torch.nn as nn
import torch.nn.functional as F
from spatial_correlation_sampler import SpatialCorrelationSampler
from utils_luo.tools import tools, tensor_tools
from utils.image_utils import InputPadder
import pdb

class Correlation(nn.Module):
    def __init__(self, max_displacement):
        super(Correlation, self).__init__()
        self.max_displacement = max_displacement
        self.kernel_size = 2*max_displacement+1
        self.corr = SpatialCorrelationSampler(1, self.kernel_size, 1, 0, 1)
        
    def forward(self, x, y):
        b, c, h, w = x.shape
        return self.corr(x, y).view(b, -1, h, w) / c


def convrelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias), 
        nn.LeakyReLU(0.1, inplace=True)
    )


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True)


class Decoder(nn.Module):
    def __init__(self, in_channels, groups):
        super(Decoder, self).__init__()
        self.in_channels = in_channels
        self.groups = groups
        self.conv1 = convrelu(in_channels, 100, 3, 1)
        self.conv2 = convrelu(100, 100, 3, 1, groups=groups)
        self.conv3 = convrelu(100, 100, 3, 1, groups=groups)
        self.conv4 = convrelu(100, 100, 3, 1, groups=groups)
        self.conv5 = convrelu(100, 64, 3, 1)
        self.conv6 = convrelu(64, 32, 3, 1)
        self.conv7 = nn.Conv2d(32, 2, 3, 1, 1)


    def channel_shuffle(self, x, groups):
        b, c, h, w = x.size()
        channels_per_group = c // groups
        x = x.view(b, groups, channels_per_group, h, w)
        x = x.transpose(1, 2).contiguous()
        x = x.view(b, -1, h, w)
        return x


    def forward(self, x):
        if self.groups == 1:
            out = self.conv7(self.conv6(self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(x)))))))
        else:
            out = self.conv1(x)
            out = self.channel_shuffle(self.conv2(out), self.groups)
            out = self.channel_shuffle(self.conv3(out), self.groups)
            out = self.channel_shuffle(self.conv4(out), self.groups)
            out = self.conv7(self.conv6(self.conv5(out)))
        return out

class EEMFlow(nn.Module):
    def __init__(self, config, groups=5, n_first_channels=5, out_mesh_size=False):
        super(EEMFlow, self).__init__()
        self.groups = groups
        self.pconv1_1 = convrelu(n_first_channels, 16, 3, 2)
        self.pconv1_2 = convrelu(16, 16, 3, 1)
        self.pconv2_1 = convrelu(16, 32, 3, 2)
        self.pconv2_2 = convrelu(32, 32, 3, 1)
        self.pconv2_3 = convrelu(32, 32, 3, 1)
        self.pconv3_1 = convrelu(32, 64, 3, 2)
        self.pconv3_2 = convrelu(64, 64, 3, 1)
        self.pconv3_3 = convrelu(64, 64, 3, 1)

        self.corr = Correlation(4)
        self.index = torch.tensor(
            [1, 3, 5, 7,
            9, 11, 13, 15, 17,
            19, 21, 22, 23, 25,
            27, 29, 30, 31, 32, 33, 35, 
            37, 38, 39, 40, 41, 42, 43, 
            45, 47, 48, 49, 50, 51, 53, 
            55, 57, 58, 59, 61, 
            63, 65, 67, 69, 71,
            73, 75, 77, 79])

        self.rconv_1 = convrelu(16, 16, 3, 1)
        self.rconv_2 = convrelu(32, 16, 3, 1)
        self.rconv_3 = convrelu(64, 16, 3, 1)

        self.decoder_1 = Decoder(69, groups)
        self.decoder_2 = Decoder(69, groups)
        self.decoder_3 = Decoder(69, groups)

        self.out_conv = nn.Conv2d(6, 2, 1, 1)

        self.out_mesh_size = out_mesh_size
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def change_imagesize(self, img_size):
        self.image_size = img_size
        self.image_padder = InputPadder(img_size, mode='chairs', eval_pad_rate=64)
    
    def upsample_flow(self, flow, orig_size):
        flow = F.interpolate(flow, size=orig_size, mode='bilinear', align_corners=False)
        return flow

    def forward(self, events1, events2):

        input_size = events1.shape[-2:]

        if(self.training):
            if(self.out_mesh_size):
                out_size = (16, 16)
            else:
                out_size = input_size
        else:
            out_size = input_size

        event1, event2 = self.image_padder.pad(events1, events2)
        f11 = self.pconv1_2(self.pconv1_1(event1))
        f21 = self.pconv1_2(self.pconv1_1(event2))
        f12 = self.pconv2_3(self.pconv2_2(self.pconv2_1(f11)))
        f22 = self.pconv2_3(self.pconv2_2(self.pconv2_1(f21)))
        f13 = self.pconv3_3(self.pconv3_2(self.pconv3_1(f12)))
        f23 = self.pconv3_3(self.pconv3_2(self.pconv3_1(f22)))

        # return  f13, f23

        pooling_size = 32
        f14_1 = F.avg_pool2d(f11, kernel_size=(pooling_size, pooling_size), stride=(pooling_size, pooling_size))
        f24_1 = F.avg_pool2d(f21, kernel_size=(pooling_size, pooling_size), stride=(pooling_size, pooling_size))

        pooling_size = 16
        f14_2 = F.avg_pool2d(f12, kernel_size=(pooling_size, pooling_size), stride=(pooling_size, pooling_size))
        f24_2 = F.avg_pool2d(f22, kernel_size=(pooling_size, pooling_size), stride=(pooling_size, pooling_size))

        pooling_size = 8
        f14_3 = F.avg_pool2d(f13, kernel_size=(pooling_size, pooling_size), stride=(pooling_size, pooling_size))
        f24_3 = F.avg_pool2d(f23, kernel_size=(pooling_size, pooling_size), stride=(pooling_size, pooling_size))

        # return  f14_3, f24_3
        
        flow_predictions = []
        
        cv_1 = torch.index_select(self.corr(f14_1, f24_1), dim=1, index=self.index.to(f14_1).long())
        r_1 = self.rconv_1(f14_1)
        # return  r_1
        cat_1 = torch.cat([cv_1, r_1], 1)
        flow_1 = self.decoder_1(cat_1)
        # flow_predictions.append(self.upsample_flow(flow_1, output_size))

        cv_2 = torch.index_select(self.corr(f14_2, f24_2), dim=1, index=self.index.to(f14_2).long())
        r_2 = self.rconv_2(f14_2)
        cat_2 = torch.cat([cv_2, r_2], 1)
        flow_2 = self.decoder_2(cat_2)
        # flow_predictions.append(self.upsample_flow(flow_2, output_size))

        cv_3 = torch.index_select(self.corr(f14_3, f24_3), dim=1, index=self.index.to(f14_3).long())
        r_3 = self.rconv_3(f14_3)
        cat_3 = torch.cat([cv_3, r_3], 1)
        flow_3 = self.decoder_3(cat_3)
        # flow_predictions.append(self.upsample_flow(flow_3, output_size))

        flow_concat = torch.cat([flow_1, flow_2, flow_3], dim=1)
        out = self.out_conv(flow_concat)
        flow_predictions.append(self.upsample_flow(out, out_size))

        return (events1, events2), flow_predictions
    
    @classmethod
    def demo(cls):
        h,w = 720, 1280
        im = torch.ones((1, 5, h, w))
        config = ''
        net = EEMFlow(config, groups=5, n_first_channels=5)
        im = im.cuda()
        net = net.cuda()

        net.change_imagesize((h,w))
        with torch.no_grad():
            _,out = net(im,im)

        tensor_tools.check_tensor(out[0], 'flow1')

    
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

    config = ''
    model = EEMFlow(config, groups=5, n_first_channels=5)
    time_eval(model)
