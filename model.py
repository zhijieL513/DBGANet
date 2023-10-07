import torch
import math
import torch.nn as nn
import torch.nn.functional as F

def GaussProjection(x, mean, std):
    sigma = math.sqrt(2 * math.pi) * std
    x_out = torch.exp(-(x - mean) ** 2 / (2 * std ** 2)) / sigma
    return x_out

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels=32, reduction_ratio=2, pool_types=['avg']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types

    def forward(self, x):
        x = x.permute(0, 3, 2, 1)
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw
        
        # Gauss modulation
        mean = torch.mean(channel_att_sum).detach()
        std = torch.std(channel_att_sum).detach()
        scale = GaussProjection(channel_att_sum, mean, std).unsqueeze(2).unsqueeze(3).expand_as(x)

        #scale = scale / torch.max(scale)
        return (x * scale).permute(0, 3, 2, 1)
    

import os
import sys
import copy
import math
from torch.autograd import Variable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from centroid import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] ="0,1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class GroupAttention(nn.Module):
    """
    LSA: self attention within a group
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., ws=1):
        assert ws != 1
        super(GroupAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.ws = ws

    def forward(self, x, H, W):
        B, N, C = x.shape
        h_group, w_group = H // self.ws, W // self.ws

        total_groups = h_group * w_group

        x = x.reshape(B, h_group, self.ws, w_group, self.ws, C).transpose(2, 3)

        qkv = self.qkv(x).reshape(B, total_groups, -1, 3, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5)
        # B, hw, ws*ws, 3, n_head, head_dim -> 3, B, hw, n_head, ws*ws, head_dim
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, hw, n_head, ws*ws, head_dim
        attn = (q @ k.transpose(-2, -1)) * self.scale  # B, hw, n_head, ws*ws, ws*ws
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(
            attn)  # attn @ v-> B, hw, n_head, ws*ws, head_dim -> (t(2,3)) B, hw, ws*ws, n_head,  head_dim
        attn = (attn @ v).transpose(2, 3).reshape(B, h_group, w_group, self.ws, self.ws, C)
        x = attn.transpose(2, 3).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class GroupBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, ws=8):
        super().__init__()
#         del self.attn
        if ws == 1:
            self.attn = Attention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, sr_ratio)
        else:
            self.attn = GroupAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, ws)
            
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.ws = ws
        
    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class Transformer(nn.Module):
    def __init__(self, d_model=64):
        super(Transformer, self).__init__()

        self.GroupBlock = GroupBlock(d_model,8, ws=10)
#         self.pe = PositionalEncodingLearned1D(d_model)

        
    def forward(self,x,H,W,d):
        
        for i in range(d):
            x = self.GroupBlock(x,H,W)
        x = x.permute(0, 2, 1) 
 
        return x



def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, n_pts, k)
    return idx


def get_graph_feature(coor, nor, k=20, idx=None):
    # coor:(B, 3, N)
    # nor:(B, 3, N)
    batch_size, num_dims_c, ncells  = coor.shape
    _, num_dims_n, _  = coor.shape
    coor = coor.view(batch_size, -1, ncells)
    if idx is None:
        idx = knn(coor, k=k)   # (B, N, k)
        
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] ="0,1"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     device = torch.device('cpu')
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * ncells # (B, 1, 1)
    idx = idx + idx_base # (B, N, k)
    idx = idx.view(-1) # (B*N*k, )
 
    coor = coor.transpose(2, 1).contiguous()   # (B, N, 3)
    nor = nor.transpose(2, 1).contiguous()   # (B, N, 3)
    
    # coor
    coor_feature = coor.view(batch_size * ncells, -1)[idx, :]
    coor_feature = coor_feature.view(batch_size, ncells, k, num_dims_c)
    coor = coor.view(batch_size, ncells, 1, num_dims_c).repeat(1, 1, k, 1)
    coor_feature = torch.cat((coor_feature, coor), dim=3).permute(0, 3, 1, 2).contiguous()
    
    # normal vector
    nor_feature = nor.view(batch_size * ncells, -1)[idx, :]
    nor_feature = nor_feature.view(batch_size, ncells, k, num_dims_n)
    nor = nor.view(batch_size, ncells, 1, num_dims_n).repeat(1, 1, k, 1)
    nor_feature = torch.cat((nor_feature, nor), dim=3).permute(0, 3, 1, 2).contiguous()
    
    return coor_feature, nor_feature # (B, 2*3, N, k)


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

    
class My_Seg(nn.Module):
    def __init__(self, num_classes=15, num_neighbor=20):
        super(My_Seg, self).__init__()

        self.k = num_neighbor
        

        self.stn_c1 = STNkd(k=3)
        self.bn1_c = nn.BatchNorm2d(64)

        self.bn2_c = nn.BatchNorm2d(128)
        self.bn3_c = nn.BatchNorm2d(256)
        self.bn4_c = nn.BatchNorm2d(256)
        self.bn5_c = nn.BatchNorm1d(256)
        self.conv1_c = nn.Sequential(nn.Conv2d(3*2, 64, kernel_size=1, bias=False),
                                   self.bn1_c,
                                   nn.LeakyReLU(negative_slope=0.2))
        
        self.conv2_c = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn2_c,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv3_c = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, bias=False),
                                   self.bn3_c,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv5_c = nn.Sequential(nn.Conv1d(448, 256, kernel_size=1, bias=False),
                                     self.bn5_c,
                                     nn.LeakyReLU(negative_slope=0.2))
        

        self.stn_n1 = STNkd(k=3)
        self.bn1_n = nn.BatchNorm2d(64)

        self.bn2_n = nn.BatchNorm2d(128)
        self.bn3_n = nn.BatchNorm2d(256)
        self.bn4_n = nn.BatchNorm2d(256)
        self.bn5_n = nn.BatchNorm1d(256)
        self.conv1_n = nn.Sequential(nn.Conv2d(3*2, 64, kernel_size=1, bias=False),
                                     self.bn1_n,
                                     nn.LeakyReLU(negative_slope=0.2))

        self.conv2_n = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                     self.bn2_n,
                                     nn.LeakyReLU(negative_slope=0.2))

        self.conv3_n = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, bias=False),
                                     self.bn3_n,
                                     nn.LeakyReLU(negative_slope=0.2))

        self.conv5_n = nn.Sequential(nn.Conv1d(448, 256, kernel_size=1, bias=False),
                                     self.bn5_n,
                                     nn.LeakyReLU(negative_slope=0.2))
        
        
        self.pred1 = nn.Sequential(nn.Conv1d(512, 512, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(512),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.pred2 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(256),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.pred3 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(128),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=0.6)
        self.dp2 = nn.Dropout(p=0.6)
        self.dp3 = nn.Dropout(p=0.6)
        self.pred4 = nn.Sequential(nn.Conv1d(128, num_classes, kernel_size=1, bias=False))
        

        self.transformer1 = Transformer(64)
        self.transformer2 = Transformer(128)
        self.transformer3 = Transformer(256)
        
        
        self.ChannelAMM = ChannelGate()
        self.num_classes = num_classes


        
        
        # 第1个图
        self.FFN1_n = nn.Sequential(torch.nn.Conv1d(64, 64*2, 1), nn.LeakyReLU(negative_slope=0.2), 
                                    nn.Dropout(0.1),torch.nn.Conv1d(64*2, 64, 1))
        self.res_linear1_1n = torch.nn.Conv1d(3, 64, 1)
        self.dropout1 = nn.Dropout(0.1)
        self.norm1 = nn.BatchNorm1d(64)

 
        # 第2个图
        self.FFN2_n = nn.Sequential(torch.nn.Conv1d(128, 128*2, 1), nn.LeakyReLU(negative_slope=0.2), 
                                    nn.Dropout(0.1),torch.nn.Conv1d(128*2, 128, 1))
        self.res_linear2_1n = torch.nn.Conv1d(64, 128, 1)
        self.dropout2 = nn.Dropout(0.1)
        self.norm2 = nn.BatchNorm1d(128)
        
        # 第3个图
        self.FFN3_n = nn.Sequential(torch.nn.Conv1d(256, 256*2, 1), nn.LeakyReLU(negative_slope=0.2), 
                                    nn.Dropout(0.1),torch.nn.Conv1d(256*2, 256, 1))
        self.res_linear3_1n = torch.nn.Conv1d(128, 256, 1)
        self.dropout3 = nn.Dropout(0.1)
        self.norm3 = nn.BatchNorm1d(256)

        
        self.centroid = centroid()
        self.pe1 = nn.Sequential(nn.Linear(42, 64), nn.ReLU())
        self.pe2 = nn.Sequential(nn.Linear(42, 128), nn.ReLU())
        self.pe3 = nn.Sequential(nn.Linear(42, 256), nn.ReLU())
        
    def forward(self, x):
        # x:(B, 6, N)
        coor = x[:, :3, :] # (B, 3, N)
        nor = x[:, 3:, :] # (B, 3, N)
        batch_size = x.size(0) # B
        ncells = x.size(2) # N
        
        cent0,label,weight = self.centroid(x)
        cent = cent0.view(batch_size,-1)
#         print(weight.shape)
        
        # coor input transform
        input_trans_c = self.stn_c1(coor) # (B, 3, 3)
        coor = coor.transpose(2, 1) # (B, 3, N) -> (B, N, 3)
        coor = torch.bmm(coor, input_trans_c)
        coor = coor.transpose(2, 1) # (B, N, 3) -> (B, 3, N)
        
        # nor input transform
        input_trans_n = self.stn_n1(nor) # (B, 3, 3)
        nor = nor.transpose(2, 1) # (B, 3, N) -> (B, N, 3)
        nor = torch.bmm(nor, input_trans_n)
        nor = nor.transpose(2, 1) # (B, N, 3) -> (B, 3, N)
#         print("coor, nor:",coor.shape,nor.shape)
        


        coor1, nor1 = get_graph_feature(coor, nor, k=self.k) # (B, 3, N) -> (B, 3*2, N, k)
        
        coor1 = self.conv1_c(coor1) # (B, 3*2, N, k) -> (B, 64, N, k)
        coor1 = coor1.max(dim=-1, keepdim=False)[0] # (B, 64, N, k) -> (B, 64, N)
        
        coor1= coor1.permute(0, 2, 1)
        cent1 = self.pe1(cent).unsqueeze(1)
        coor1 = coor1 + cent1
        coor1 = self.transformer1(coor1,100,100,2)

        
        nor1 = self.conv1_n(nor1) # (B, 3*2, N, k) -> (B, 64, N, k)
        nor1 = self.ChannelAMM(nor1)
        nor1 = nor1.max(dim=-1, keepdim=False)[0] # (B, 64, N, k) -> (B, 64, N)
        
    
        coor2, nor2 = get_graph_feature(coor1, nor1, k=self.k) # (B, 64, N) -> (B, 64*2, N, k)
        
        coor2 = self.conv2_c(coor2) # (B, 64*2, N, k) -> (B, 64, N, k)
        coor2 = coor2.max(dim=-1, keepdim=False)[0] # (B, 64, N, k) -> (B, 64, N)
        
        coor2= coor2.permute(0, 2, 1)
        cent2 = self.pe2(cent).unsqueeze(1)
        coor2 = coor2 + cent2
        coor2 = self.transformer2(coor2,100,100,3)

        
        nor2 = self.conv2_n(nor2) # (B, 64*2, N, k) -> (B, 64, N, k)
        nor2 = self.ChannelAMM(nor2)
        nor2 = nor2.max(dim=-1, keepdim=False)[0] # (B, 64, N, k) -> (B, 64, N)


        coor3, nor3 = get_graph_feature(nor2, coor2, k=self.k) # (B, 64, N) -> (B, 64*2, N, k)

        coor3 = self.conv3_c(coor3) # (B, 64*2, N, k) -> (B, 128, N, k)
        coor3 = coor3.max(dim=-1, keepdim=False)[0] # (B, 128, N, k) -> (B, 128, N)

        coor3= coor3.permute(0, 2, 1)
        cent3 = self.pe3(cent).unsqueeze(1)
        coor3 = coor3 + cent3
        coor3 = self.transformer3(coor3,100,100,2)
        
        
        nor3 = self.conv3_n(nor3) # (B, 64*2, N, k) -> (B, 128, N, k)
        nor3 = self.ChannelAMM(nor3)
        nor3 = nor3.max(dim=-1, keepdim=False)[0] # (B, 128, N, k) -> (B, 128, N)
 
        
        coor = torch.cat((coor1, coor2, coor3), dim=1) # (B, 64+64+128, N)
        coor = self.conv5_c(coor) # (B, 256, N) -> (B, 512, N)
        nor = torch.cat((nor1, nor2, nor3), dim=1) # (B, 64+64+128, N)
        nor = self.conv5_n(nor) # (B, 256, N) -> (B, 512, N)
        
        x = torch.cat((coor, nor), dim=1) # (B, 512*2, N)
        
        
        x = self.pred1(x) # (B, 512*2, N) -> (B, 512, N)
        x = self.dp1(x)
        x = self.pred2(x) # (B, 512, N) -> (B, 256, N)
        x = self.dp2(x)
        x = self.pred3(x) # (B, 256, N) -> (B, 128, N)
        x = self.dp3(x)
        
        x = self.pred4(x) # (B, 128, N) -> (B, 15, N)
        

        
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x,cent0.permute(0, 2, 1),label,weight.mean(axis=0)
    
# import torch
# import torchvision
# from thop import profile
# input0 = torch.rand(1, 6, 10000).cuda()
# model = My_Seg(num_classes=15, num_neighbor=32).cuda() 
# out = model(input0)
# print(out[1].shape) 

# flops, params = profile(model, (input0,))
# print('flops: ', flops, 'params: ', params)
# print('flops: %.2f G, params: %.2f M' % (flops / 1e9, params / 1e6))