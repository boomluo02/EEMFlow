from selectors import EVENT_READ
from tkinter import E
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import RandomSampler

import pdb

def select_nearest_vector(vect, M): # vector B x N x Cn
    N = vect.shape[1]
    if M < N:
        padding = M // 2
        vect_p = F.pad(vect, pad=[0, 0, padding, padding-1, 0, 0])
        vect_multi = vect_p.unfold(dimension=1,size=M,step=1) # B x N x Cn x M

        return vect_multi.permute((0,1,3,2)) # B x N x M x Cn
    else:
        return torch.unsqueeze(vect, 2).repeat((1,1,N,1)).permute((0,2,1,3)) # B x N x N x Cn

def farthest_point_sample_batch(event, npoint):
    """
    Input:
        event: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint, C]
    """
    device = event.device
    B, N, C = event.shape
    centroids = torch.zeros(B, N, C, dtype=torch.long).to(device)       # 采样点矩阵（B, N, C）
    distance = torch.ones(B, N).to(device) * 1e10                       # 采样点到所有点距离（B, N）

    batch_indices = torch.arange(B, dtype=torch.long).to(device)        # batch_size 数组

    #farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)  # 初始时随机选择一点
    
    barycenter = torch.sum((event), 1)                                    #计算重心坐标 及 距离重心最远的点
    barycenter = barycenter/event.shape[1]
    barycenter = barycenter.view(B, 1, C)

    dist = torch.sum((event - barycenter) ** 2, -1)
    farthest = torch.max(dist,1)[1]                                     #将距离重心最远的点作为第一个点

    for i in range(npoint):

        centroids[batch_indices, farthest, :] = 1                         # 更新第i个最远点
                        
        centroid = event[batch_indices, farthest, :].view(B, 1, C)        # 取出这个最远点的event坐标

        dist = torch.sum((event - centroid) ** 2, -1)                     # 计算点集中的所有点到这个最远点的欧式距离

        mask = dist < distance

        distance[mask] = dist[mask]                                     # 更新distance，记录样本中每个点距离所有已出现的采样点的最小距离

        farthest = torch.max(distance, -1)[1]                           # 返回最远点索引
    try:
        sample = event.masked_select(centroids>0).view(B, npoint, C)
    except:
        stride = N // npoint
        idx = torch.arange(0,stride*npoint,stride)
        sample = event[:, idx, :]
        
    return sample

class LXformer(nn.Module):
    def __init__(self, C, Cn, M, dropout=0.1):
        super().__init__()

        self.nearst_events_num = M

        self.w_qs = nn.Linear(C, Cn, bias=False)
        self.w_ks = nn.Linear(C, Cn, bias=False)
        self.w_vs = nn.Linear(C, Cn, bias=False)

        # self.w_pe = nn.Linear(C, Cn, bias=False)

        self.w_sa1 = nn.Linear(Cn, 1, bias=False)
        self.w_sa2 = nn.Linear(Cn, C, bias=False)

        self.dropout = nn.Dropout(p=dropout)

        self.layer_norm = nn.LayerNorm(C, eps=1e-6)

    """
    pe: position encoder, event sequence: N x 4, distance matrix: N x N x 4, after a MLP(4 -> Cn) change shape: N x N x Cn 
    """
    def forward(self, q, k, v, pe, mask=None):

        q = self.w_qs(q) # q: B x N x C -> B x N x Cn
        k = self.w_ks(k)
        v = self.w_vs(v)

        # pe = self.w_pe(pe)

        # attention mechanism
        B, N, Cn = q.shape[0], q.shape[1], q.shape[2]
        M = self.nearst_events_num

        q_multi = torch.unsqueeze(q, 2).repeat((1,1,M,1)) # q_multi: B x N x M x Cn
        # k_multi = torch.tile(torch.unsqueeze(k, 2), (1, 1, 1, 1))
        k_multi = select_nearest_vector(k, M)
        v_multi = select_nearest_vector(v, M) # v_multi: B x N x M x Cn

        sa = q_multi - k_multi + pe # B x N x M x Cn
        sa = self.w_sa1(sa) # B x N x M x Cn
        sa = sa.view(B, 1, N, M) # B x N x M
        if mask is not None:
            sa = sa.masked_fill(mask == 0, -1e9) 
        scores = sa.softmax(dim=-1) 

        
        attn_vect =  (v_multi + pe).permute(0,3,2,1) # B x N x M x Cn -> B x Cn x M x N
        p_attn = torch.matmul(scores, attn_vect) # scores: B x N x M, v_multi: B x Cn x M x N, result shape: B x Cn x N x N
        
        p_attn = torch.sum(p_attn, axis=-1).view(B, Cn, N).permute(0, 2, 1) # B x N x Cn

        out = self.w_sa2(p_attn)

        out = self.dropout(out)

        return out


class GXformer(nn.Module):
    def __init__(self, C, Cn, M, dropout=0.1):
        super().__init__()

        self.nearst_events_num = M

        self.w_qs = nn.Linear(C, Cn, bias=False)
        self.w_ks = nn.Linear(C, Cn, bias=False)
        self.w_vs = nn.Linear(C, Cn, bias=False)

        self.w_pe = nn.Linear(C, Cn, bias=False)

        self.w_sa1 = nn.Linear(Cn, 1, bias=False)
        self.w_sa2 = nn.Linear(Cn, C, bias=False)

        self.dropout = nn.Dropout(p=dropout)

        self.layer_norm = nn.LayerNorm(C, eps=1e-6)
    
    def position_encoding(self, events_embeding, M):
        N = events_embeding.shape[1]
        events_embeding_multi = torch.unsqueeze(events_embeding, 2).repeat((1,1,M,1)) # B x N x M x C
        events_embeding_m = farthest_point_sample_batch(events_embeding, M) # B x M x C
        events_embeding_multi_m = torch.unsqueeze(events_embeding_m, 1).repeat((1,N,1,1)) # B x N x M x C
        pe = events_embeding_multi - events_embeding_multi_m
        return pe

    """
    pe: position encoder, event sequence: N x 4, distance matrix: N x N x 4, after a MLP(4 -> Cn) change shape: N x N x Cn 
    """
    def forward(self, events_feature, mask=None):

        q = self.w_qs(events_feature) # q: B x N x C -> B x N x Cn
        k = self.w_ks(events_feature)
        v = self.w_vs(events_feature)

        # attention mechanism
        B, N, Cn = q.shape[0], q.shape[1], q.shape[2]
        M = self.nearst_events_num

        pe = self.position_encoding(events_feature, M) # B x M x C
        pe = self.w_pe(pe) # B x M x Cn

        q_multi = torch.unsqueeze(q, 2).repeat((1,1,M,1)) # q_multi: B x N x M x Cn
        # k_multi = torch.tile(torch.unsqueeze(k, 2), (1, 1, 1, 1))
        k_m = farthest_point_sample_batch(k, M) # B x M x Cn
        v_m = farthest_point_sample_batch(v, M) # B x M x Cn

        k_multi = torch.unsqueeze(k_m, 1).repeat((1,N,1,1)) # k_multi: B x N x M x Cn
        v_multi = torch.unsqueeze(v_m, 1).repeat((1,N,1,1)) # v_multi: B x N x M x Cn

        sa = q_multi - k_multi + pe # B x N x M x Cn
        sa = self.w_sa1(sa) # B x N x M x Cn
        sa = sa.view(B, 1, N, M) # B x N x M
        if mask is not None:
            sa = sa.masked_fill(mask == 0, -1e9) 
        scores = sa.softmax(dim=-1) 

        attn_vect =  (v_multi + pe).permute(0,3,2,1) # B x N x M x Cn -> B x Cn x M x N
        p_attn = torch.matmul(scores, attn_vect) # scores: B x N x M, v_multi: B x Cn x M x N, result shape: B x Cn x N x N
        
        p_attn = torch.sum(p_attn, axis=-1).view(B, Cn, N).permute(0, 2, 1) # B x N x Cn

        out = self.w_sa2(p_attn)

        out = self.dropout(out)

        return out

class EventTransformer(nn.Module): # events: B x N x 4
    def __init__(self, C, Cn, M=8, image_size=(180, 240), dropout=0.5):
        super().__init__()
        self.H, self.W = image_size[0], image_size[1]
        self.nearest_events_num = M
        self.mlp_1 = nn.Linear(4, C, bias=False)
        self.w_pe = nn.Linear(4, Cn, bias=False)
        
        self.LN = nn.LayerNorm([C])
        self.dropout = nn.Dropout(p=dropout)
        self.gelu = nn.GELU()

        self.lx_former = LXformer(C, Cn, M, dropout=dropout)
        self.gx_former = GXformer(C, Cn, M, dropout=dropout)
    
    def position_encoding(self, events_embeding, M): # events_embeding: BxNx4
        events_embeding_multi = torch.unsqueeze(events_embeding, 2).repeat((1,1,M,1)) # B x N x M x 4
        events_embeding_multi_t = select_nearest_vector(events_embeding, M)
        pe = events_embeding_multi - events_embeding_multi_t

        return pe
    
    def events_sequence_to_image(self, events_feature, events, H, W, normalize=True): # N x C
        device = events_feature.device

        # events_feature = torch.abs(events_feature)
        events_feature = self.gelu(self.LN(events_feature))
        events = events
        N, C = events_feature.shape[0], events_feature.shape[1]
        events_space = torch.zeros([H, W, C], device=device)
        # events_sum = torch.zeros([H, W, C], device=device)
        x = torch.floor(events[:,0])
        y = torch.floor(events[:,1])

        p = events[:,3]
        for c in range(C):
            channel = torch.full([N], c, device=device)
            # index = torch.stack((y,x,channel), dim=1).flatten().long()
            # idx = torch.split(index, [3]*N, dim=0)
            idx=[y.long(), x.long(), channel.long()]
            events_space.index_put_(idx, p*events_feature[:,c], accumulate=True)
            # events_sum.index_put_(idx, p**2, accumulate=True)

        # print(p[0]*events_feature[0])
        # print(events_space[y.long()[0]][x.long()[0]])
        # pdb.set_trace()
        
        # events_sum[(events_sum == 0)] += 1.0

        # events_space = events_space / events_sum
        events_sapce = events_space.to(device=device) # H X W x C

        if normalize:
            mask = torch.nonzero(events_sapce, as_tuple=True)
            if mask[0].size()[0] > 0:
                mean = events_sapce[mask].mean()
                std = events_sapce[mask].std()
                if std > 0:
                    events_sapce[mask] = (events_sapce[mask] - mean) / std
                else:
                    events_sapce[mask] = events_sapce[mask] - mean

        return events_sapce


    def forward(self, events): # events: B x N x 4
        is_list = isinstance(events, tuple) or isinstance(events, list)
        if is_list:
            batch_dim = events[0].shape[0]
            batch_dim_list = [batch_dim] * len(events)
            events = torch.cat(events, dim=0)
        
        lx_in = self.mlp_1(events) # B x N x C
        lx_in = self.dropout(lx_in)

        pe = self.position_encoding(events, self.nearest_events_num)
        pe = self.w_pe(pe)

        lx_sa = self.lx_former(lx_in, lx_in, lx_in, pe)

        lx_out = lx_in + lx_sa

        gx_sa = self.gx_former(lx_out)

        gx_out = lx_out + gx_sa

        image_list = []
        for b in range(gx_out.shape[0]):
            sc_in = self.events_sequence_to_image(gx_out[b], events[b], self.H, self.W) # shape: H x W x C
            # sc_in = self.LN(sc_in)
            image_list.append(sc_in)
        out = torch.stack(image_list, dim=0).permute(0,3,1,2) # B x C x H x W 

        if is_list:
            out = torch.split(out, batch_dim_list, dim=0)

        return out
        

# if __name__ == '__main__':
#     C = 32
#     Cn = 64
#     N = 1024
#     model = EventTransformer(C, Cn)

#     train_data_path = '/home/luoxinglong/workspace/rpg_event_representation_learning/N-Caltech101/validation'
#     dataset = NCaltech101(train_data_path, augmentation=False)
#     idx = 100
#     events, label = dataset[idx]

#     events = torch.from_numpy(events)
#     sample = RandomSampler(events, replacement=True, num_samples=N)
#     # print(sample)
#     events_sample = torch.cat([torch.unsqueeze(events[idx],0) for idx in sample])
    
#     out = model(events_sample) # shape: N x (4 + C)
#     print(out)


















