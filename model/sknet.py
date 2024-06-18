import torch
from torch import nn

import pdb
#from thop import profile
#from thop import clever_format


class SK(nn.Module):
    def __init__(self):
        """ Constructor
        Args:
            features: input channel dimensionality.
            M: the number of branchs.
            G: num of convolution groups.
            r: the ratio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(SK, self).__init__()
  
        # self.in_conv1 = nn.Sequential(
        #     nn.Conv2d(5, 16, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(inplace=True)
        #     )
        
        # self.in_conv2 = nn.Sequential(
        #     nn.Conv2d(5, 16, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(inplace=True)
        #     )

        self.fc = nn.Sequential(            
            nn.Conv2d(5, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(2),
            nn.ReLU(inplace=True))

        # self.gap = nn.AdaptiveAvgPool2d((1,1))
        # self.fc = nn.Sequential(nn.Conv2d(5, 16, kernel_size=1, stride=1, bias=False),
        #                         nn.InstanceNorm2d(16),
        #                         nn.ReLU(inplace=True))
        # self.fcs = nn.ModuleList([])
        # for i in range(2):
        #     self.fcs.append(
        #          nn.Conv2d(16, 5, kernel_size=1, stride=1)
        #     )
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, event, d_event):

        is_list = isinstance(event, tuple) or isinstance(event, list)
        if is_list:
            batch_dim = event[0].shape[0]
            event = torch.cat(event, dim=0)
            d_event = torch.cat(d_event, dim=0)

        feats_U = event + d_event
        feats_Z = self.fc(feats_U)

        attention_map = self.softmax(feats_Z)
        # attention_map = torch.unsqueeze(attention_map,dim=2).repeat(1,1,5,1,1) # shape: event.shape[0], 2, 5, event.shape[-2], event.shape[-1]
        # event = torch.cat([event, d_event], dim=1)
        # event = event.view(event.shape[0], 2, 5, event.shape[-2], event.shape[-1])
        # feats_V = torch.sum(event*attention_map, dim=1)
        # print(torch.mean(attention_map[:,:1,:,:]+attention_map[:,1:,:,:]))
        feats_V = attention_map[:,:1,:,:] * event + attention_map[:,1:,:,:] * d_event

        # feats_U = event + d_event
        # feats_S = self.gap(feats_U)
        # feats_Z = self.fc(feats_S)

        # attention_vectors = [fc(feats_Z) for fc in self.fcs]
        # attention_vectors = torch.cat(attention_vectors, dim=1)
        # attention_vectors = attention_vectors.view(event.shape[0], 2, 5, 1, 1)
        # attention_vectors = self.softmax(attention_vectors)
        # # print(attention_vectors.shape)
        # print(torch.mean(attention_vectors[:,0,:,:,:] + attention_vectors[:,1,:,:,:]))

        # feats_V = attention_vectors[:,0,:,:,:] * event + attention_vectors[:,1,:,:,:] * d_event


        if is_list:
            feats_V = torch.split(feats_V, [batch_dim, batch_dim], dim=0)
        return feats_V


class SK_score(nn.Module):
    def __init__(self):
        """ Constructor
        Args:
            features: input channel dimensionality.
            M: the number of branchs.
            G: num of convolution groups.
            r: the ratio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(SK_score, self).__init__()
  
        # self.in_conv1 = nn.Sequential(
        #     nn.Conv2d(5, 16, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(inplace=True)
        #     )
        
        # self.in_conv2 = nn.Sequential(
        #     nn.Conv2d(5, 16, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(inplace=True)
        #     )

        # self.fc = nn.Sequential(            
        #     nn.Conv2d(5, 16, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.InstanceNorm2d(16),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(16, 2, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.InstanceNorm2d(2),
        #     nn.ReLU(inplace=True))

        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(nn.Conv2d(5, 16, kernel_size=1, stride=1, bias=False),
                                nn.InstanceNorm2d(16),
                                nn.ReLU(inplace=True))
        self.fcs = nn.ModuleList([])
        for i in range(2):
            self.fcs.append(
                 nn.Conv2d(16, 1, kernel_size=1, stride=1)
            )
        self.softmax = nn.Softmax(dim=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        
    def forward(self, event, d_event):

        is_list = isinstance(event, tuple) or isinstance(event, list)
        if is_list:
            batch_dim = event[0].shape[0]
            event = torch.cat(event, dim=0)
            d_event = torch.cat(d_event, dim=0)


        feats_U = event + d_event
        feats_S = self.gap(feats_U)
        feats_Z = self.fc(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(event.shape[0], 2, 1, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        print(attention_vectors.shape)
        print(torch.mean(attention_vectors[:,0,:,:,:] + attention_vectors[:,1,:,:,:]))

        feats_V = attention_vectors[:,0,:,:,:] * event + attention_vectors[:,1,:,:,:] * d_event


        if is_list:
            feats_V = torch.split(feats_V, [batch_dim, batch_dim], dim=0)
        return feats_V

class SKConv(nn.Module):
    def __init__(self, features, M=2, G=32, r=16, stride=1 ,L=32):
        """ Constructor
        Args:
            features: input channel dimensionality.
            M: the number of branchs.
            G: num of convolution groups.
            r: the ratio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(SKConv, self).__init__()
        d = max(int(features/r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(features, features, kernel_size=3, stride=stride, padding=1+i, dilation=1+i, groups=G, bias=False),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True)
            ))
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(nn.Conv2d(features, d, kernel_size=1, stride=1, bias=False),
                                nn.BatchNorm2d(d),
                                nn.ReLU(inplace=True))
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                 nn.Conv2d(d, features, kernel_size=1, stride=1)
            )
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        
        batch_size = x.shape[0]
        
        feats = [conv(x) for conv in self.convs]      
        feats = torch.cat(feats, dim=1)
        feats = feats.view(batch_size, self.M, self.features, feats.shape[2], feats.shape[3])
        
        feats_U = torch.sum(feats, dim=1)
        feats_S = self.gap(feats_U)
        feats_Z = self.fc(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.M, self.features, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        
        feats_V = torch.sum(feats*attention_vectors, dim=1)
        
        return feats_V


class SKUnit(nn.Module):
    def __init__(self, in_features, mid_features, out_features, M=2, G=32, r=16, stride=1, L=32):
        """ Constructor
        Args:
            in_features: input channel dimensionality.
            out_features: output channel dimensionality.
            M: the number of branchs.
            G: num of convolution groups.
            r: the ratio for compute d, the length of z.
            mid_features: the channle dim of the middle conv with stride not 1, default out_features/2.
            stride: stride.
            L: the minimum dim of the vector z in paper.
        """
        super(SKUnit, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, mid_features, 1, stride=1, bias=False),
            nn.BatchNorm2d(mid_features),
            nn.ReLU(inplace=True)
            )
        
        self.conv2_sk = SKConv(mid_features, M=M, G=G, r=r, stride=stride, L=L)
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_features, out_features, 1, stride=1, bias=False),
            nn.BatchNorm2d(out_features)
            )
        

        if in_features == out_features: # when dim not change, input_features could be added diectly to out
            self.shortcut = nn.Sequential()
        else: # when dim not change, input_features should also change dim to be added to out
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_features, out_features, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_features)
            )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.conv2_sk(out)
        out = self.conv3(out)
        
        return self.relu(out + self.shortcut(residual))

class SKNet(nn.Module):
    def __init__(self, class_num, nums_block_list = [3, 4, 6, 3], strides_list = [1, 2, 2, 2]):
        super(SKNet, self).__init__()
        self.basic_conv = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        self.maxpool = nn.MaxPool2d(3,2,1)
        
        self.stage_1 = self._make_layer(64, 128, 256, nums_block=nums_block_list[0], stride=strides_list[0])
        self.stage_2 = self._make_layer(256, 256, 512, nums_block=nums_block_list[1], stride=strides_list[1])
        self.stage_3 = self._make_layer(512, 512, 1024, nums_block=nums_block_list[2], stride=strides_list[2])
        self.stage_4 = self._make_layer(1024, 1024, 2048, nums_block=nums_block_list[3], stride=strides_list[3])
     
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(2048, class_num)
        
    def _make_layer(self, in_feats, mid_feats, out_feats, nums_block, stride=1):
        layers=[SKUnit(in_feats, mid_feats, out_feats, stride=stride)]
        for _ in range(1,nums_block):
            layers.append(SKUnit(out_feats, mid_feats, out_feats))
        return nn.Sequential(*layers)

    def forward(self, x):
        fea = self.basic_conv(x)
        fea = self.maxpool(fea)
        fea = self.stage_1(fea)
        fea = self.stage_2(fea)
        fea = self.stage_3(fea)
        fea = self.stage_4(fea)
        fea = self.gap(fea)
        fea = torch.squeeze(fea)
        fea = self.classifier(fea)
        return fea

def SKNet26(nums_class=1000):
    return SKNet(nums_class, [2, 2, 2, 2])
def SKNet50(nums_class=1000):
    return SKNet(nums_class, [3, 4, 6, 3])
def SKNet101(nums_class=1000):
    return SKNet(nums_class, [3, 4, 23, 3])

if __name__=='__main__':
    x = torch.rand(8, 3, 224, 224)
    model = SKNet26()
    out = model(x)
    
    #flops, params = profile(model, (x, ))
    #flops, params = clever_format([flops, params], "%.5f")
    
    #print(flops, params)
    #print('out shape : {}'.format(out.shape))
