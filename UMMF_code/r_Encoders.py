
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
from random import randrange
from einops import repeat, rearrange

class sMRIClassifier(nn.Module):
    """
    sMRI 模态分类器
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        
        ####################### sMRI深度特征提取模块 #######################
        # 3D卷积层定义
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(5, 5, 5), stride=1, padding=0),  # 3D卷积
            nn.LeakyReLU(),  # 激活函数
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)  # 最大池化
        )  # 卷积块1
        self.conv2 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=(5, 5, 5), stride=1, padding=0),  # 3D卷积
            nn.LeakyReLU(),  # 激活函数
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)  # 最大池化
        )  # 卷积块2
        self.conv3 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(5, 5, 5), stride=1, padding=0),  # 3D卷积
            nn.LeakyReLU(),  # 激活函数
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)  # 最大池化
        )  # 卷积块3
        self.conv4 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=(5, 5, 5), stride=1, padding=0),  # 3D卷积
            nn.LeakyReLU(),  # 激活函数
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)  # 最大池化
        )  # 卷积块4

        self.conv1_bn = nn.BatchNorm3d(16)  # 归一化层1
        self.conv2_bn = nn.BatchNorm3d(32)  # 归一化层2
        self.conv3_bn = nn.BatchNorm3d(64)  # 归一化层3
        self.conv4_bn = nn.BatchNorm3d(64)  # 归一化层4

        # 多层感知机（MLP），用于从卷积提取的深度特征中提取全局特征
        # self.mlp1 = nn.Sequential(nn.Linear(28224, 4096), nn.ReLU(), nn.BatchNorm1d(4096), nn.Dropout(p=0.3))
        self.mlp1 = nn.Sequential(nn.Linear(110592, 4096), nn.ReLU(), nn.BatchNorm1d(4096), nn.Dropout(p=0.3))
        self.mlp2 = nn.Sequential(nn.Linear(4096, 1024), nn.ReLU(), nn.BatchNorm1d(1024), nn.Dropout(p=0.3))
        self.mlp3 = nn.Sequential(nn.Linear(1024, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(p=0.3))
        self.mlp4 = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(p=0.3))
        self.mlp5 = nn.Sequential(nn.Linear(128, 2), nn.BatchNorm1d(2), nn.ReLU())

    def forward(self, sMRI_deep):
        """
        前向传播。
        参数：
        - sMRI_deep: sMRI 的深度特征数据，形状为 (batch_size, num_channels, depth, height, width)

        返回：
        - 最终的深度特征 (feat_sMRI_deep)
        """
        sMRI_deep_x1 = self.conv1(sMRI_deep)
        sMRI_deep_x2 = self.conv1_bn(sMRI_deep_x1)
        sMRI_deep_x3 = self.conv2(sMRI_deep_x2)
        sMRI_deep_x4 = self.conv2_bn(sMRI_deep_x3)
        sMRI_deep_x5 = self.conv3(sMRI_deep_x4)
        sMRI_deep_x6 = self.conv3_bn(sMRI_deep_x5)
        sMRI_deep_x7 = self.conv4(sMRI_deep_x6)
        sMRI_deep_x8 = self.conv4_bn(sMRI_deep_x7)

        # 展平并通过MLP提取全局特征
        sMRI_deep_x9_vec = sMRI_deep_x8.view(sMRI_deep_x8.size(0), -1)
        sMRI_deep_x10 = self.mlp1(sMRI_deep_x9_vec)
        sMRI_deep_x11 = self.mlp2(sMRI_deep_x10)
        # feat_sMRI_deep = self.mlp4(self.mlp3(sMRI_deep_x11))
        feat_sMRI_deep = self.mlp3(sMRI_deep_x11)

        return feat_sMRI_deep

    def _conv_layer_set(self, in_channels, out_channels):
        """
        定义3D卷积块，包括卷积、激活函数和池化操作。
        参数：
        - in_channels: 输入通道数
        - out_channels: 输出通道数
        返回：
        - 一个包含卷积、激活和池化操作的神经网络模块
        """
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(5, 5, 5), stride=1, padding=0),  # 3D卷积
            nn.LeakyReLU(),  # 激活函数
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)  # 最大池化
        )
    


class E2EBlock(torch.nn.Module):
    '''E2Eblock.'''

    def __init__(self, in_planes, planes, example, bias=False):
        super(E2EBlock, self).__init__()
        self.d = example.size(3)
        self.cnn1 = torch.nn.Conv2d(in_planes, planes, (1, self.d), bias=bias)
        self.cnn2 = torch.nn.Conv2d(in_planes, planes, (self.d, 1), bias=bias)

    def forward(self, x):
        a = self.cnn1(x)
        b = self.cnn2(x)
        return torch.cat([a] * self.d, 3) + torch.cat([b] * self.d, 2)


class Edge2Edge(nn.Module):
    def __init__(self, channel, dim, filters):
        super(Edge2Edge, self).__init__()
        self.channel = channel
        self.dim = dim
        self.filters = filters
        self.row_conv = nn.Conv2d(channel, filters, (1, dim))
        self.col_conv = nn.Conv2d(channel, filters, (dim, 1))

    # implemented by two conv2d with line filter
    def forward(self, x):
        size = x.size()
        row = self.row_conv(x)
        col = self.col_conv(x)
        row_ex = row.expand(size[0], self.filters, self.dim, self.dim)
        col_ex = col.expand(size[0], self.filters, self.dim, self.dim)
        return row_ex + col_ex

class Edge2Node(nn.Module):
    def __init__(self, channel, dim, filters):
        super(Edge2Node, self).__init__()
        self.channel = channel
        self.dim = dim
        self.filters = filters
        self.row_conv = nn.Conv2d(channel, filters, (1, dim))
        self.col_conv = nn.Conv2d(channel, filters, (dim, 1))

    def forward(self, x):
        row = self.row_conv(x)
        col = self.col_conv(x)
        return row + col.permute(0, 1, 3, 2)
class Node2Graph(nn.Module):
    def __init__(self, channel, dim, filters):
        super(Node2Graph, self).__init__()
        self.channel = channel
        self.dim = dim
        self.filters = filters
        self.conv = nn.Conv2d(channel, filters, (dim, 1))

    def forward(self, x):
        return self.conv(x)



class BNCNN(nn.Module):
    """
    Adjusted KawaharaBNCNN for Binary Classification
    """

    def __init__(self, example):
        """
        Initialize BrainNetCNN for binary classification.

        Args:
            example: Example input tensor to determine the input dimension.
        """
        super(BNCNN, self).__init__()
        self.d = example.size(3)  # Input matrix dimension

        # Define network components
        self.e2econv1 = E2EBlock(1, 32, example, bias=True)  # First Edge-to-Edge layer
        self.e2econv2 = E2EBlock(32, 32, example, bias=True)  # Second Edge-to-Edge layer
        self.E2N = Edge2Node(32, self.d, 64)  # Edge-to-Node layer
        self.N2G = Node2Graph(64, self.d, 256)  # Node-to-Graph layer

        # Fully connected layers
        self.fc1 = nn.Linear(256, 128)  # Fully connected layer 1
        self.fc2 = nn.Linear(128, 256)  # Fully connected layer 2
        self.fc3 = nn.Linear(30, 1)  # Binary classification output

        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize weights with Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        """
        Forward pass of the network.
        """
        out = F.leaky_relu(self.e2econv1(x), negative_slope=0.33)
        out = F.leaky_relu(self.e2econv2(out), negative_slope=0.33)
        out = F.leaky_relu(self.E2N(out), negative_slope=0.33)
        out = F.dropout(F.leaky_relu(self.N2G(out), negative_slope=0.33), p=0.5)

        # Flatten and pass through fully connected layers
        out = out.view(out.size(0), -1)
        out = F.dropout(F.relu(self.fc1(out)), p=0.5)
        out = F.dropout(F.relu(self.fc2(out)), p=0.5)
        return out
    

def process_dynamic_fc(timeseries, window_size, window_stride, dynamic_length=None, sampling_init=None, self_loop=True):
    """
    处理动态功能连接矩阵（Dynamic FC）。
    参数:
    - timeseries: 输入的时间序列数据，形状为 (batch_size, timepoints, num_nodes)
    - window_size: 滑动窗口大小
    - window_stride: 滑动窗口步长
    - dynamic_length: 动态序列的长度
    - sampling_init: 采样起点
    - self_loop: 是否保留自连接
    返回:
    - dynamic_fc: 动态功能连接矩阵
    - sampling_points: 滑动窗口起点列表
    """
    if dynamic_length is None:
        dynamic_length = timeseries.shape[1]
        sampling_init = 0
    else:
        if isinstance(sampling_init, int):
            assert timeseries.shape[1] > sampling_init + dynamic_length
    assert sampling_init is None or isinstance(sampling_init, int)
    assert timeseries.ndim == 3
    assert dynamic_length > window_size

    if sampling_init is None:
        sampling_init = randrange(timeseries.shape[1] - dynamic_length + 1)
    sampling_points = list(range(sampling_init, sampling_init + dynamic_length - window_size, window_stride))

    dynamic_fc_list = []
    for i in sampling_points:
        fc_list = []
        for _t in timeseries:
            fc = corrcoef(_t[i:i + window_size].T)
            if not self_loop:
                fc -= torch.eye(fc.shape[0])
            fc_list.append(fc)
        dynamic_fc_list.append(torch.stack(fc_list))
    return torch.stack(dynamic_fc_list, dim=1), sampling_points


def corrcoef(x):
    """
    计算相关系数矩阵。
    参数:
    - x: 输入矩阵，形状为 (num_nodes, window_size)
    返回:
    - 相关系数矩阵，形状为 (num_nodes, num_nodes)
    """
    mean_x = torch.mean(x, 1, keepdim=True)
    xm = x.sub(mean_x.expand_as(x))
    c = xm.mm(xm.t())
    c = c / (x.size(1) - 1)
    d = torch.diag(c)
    stddev = torch.pow(d, 0.5)
    c = c.div(stddev.expand_as(c))
    c = c.div(stddev.expand_as(c).t())
    c = torch.clamp(c, -1.0, 1.0)
    return c


class fMRIClassifier(nn.Module):
    """
    基于 fMRI 数据的分类模型，包含 GNN 和 Transformer 模块。
    """
    def __init__(self, input_dim=116, hidden_dim=256, num_classes=2, num_heads=1, num_layers=2, sparsity=30, dropout=0.1, readout='sero', cls_token = 'sum'):
        super().__init__()
        assert cls_token in ['sum', 'mean', 'param']
        if cls_token == 'sum':
            self.cls_token = lambda x: x.sum(0)  # 对时间步特征求和
        elif cls_token == 'mean':
            self.cls_token = lambda x: x.mean(0)  # 对时间步特征求均值
        elif cls_token == 'param':
            self.cls_token = lambda x: x[-1]  # 使用最后一个时间步的特征
        else:
            raise ValueError("Invalid cls_token specified!")
        

        if readout == 'garo':
            readout_module = ModuleGARO  # 全局注意力读取
        elif readout == 'sero':
            readout_module = ModuleSERO  # 自注意力读取
        elif readout == 'mean':
            readout_module = ModuleMeanReadout  # 均值读取
        else:
            raise ValueError("Invalid readout module specified!")
        
        self.num_classes = num_classes
        self.sparsity = sparsity

        self.initial_linear = nn.Linear(input_dim, hidden_dim)  # fMRI特征线性变换
        self.gnn_layers = nn.ModuleList()  # 存储GNN层的模块列表
        self.readout_modules = nn.ModuleList()  # 存储读取模块的列表
        self.transformer_modules = nn.ModuleList()  # 存储Transformer模块
        self.linear_layers = nn.ModuleList()  # 分类的线性层
        self.dropout = nn.Dropout(dropout)  # dropout层

        for i in range(num_layers):
            self.gnn_layers.append(LayerGIN(hidden_dim, hidden_dim, hidden_dim))  # GIN层，用于更新顶点特征
            self.readout_modules.append(
                readout_module(hidden_dim=hidden_dim, input_dim=input_dim, dropout=0.1))  # 读取模块
            self.transformer_modules.append(
                ModuleTransformer(hidden_dim, 2 * hidden_dim, num_heads=num_heads, dropout=0.1))  # Transformer模块
            self.linear_layers.append(nn.Linear(hidden_dim, num_classes))  # 分类层

    def forward(self, fMRI_v, fMRI_a):
        """
        前向传播函数。
        参数:
        - fMRI_v: 节点特征，形状 (batch_size, segment_num, num_nodes, feature_dim)
        - fMRI_a: 邻接矩阵，形状 (batch_size, segment_num, num_nodes, num_nodes)
        返回:
        - logit_fMRI: fMRI的分类结果、
        - fMRI_feature: fMRI的特征
        """
        logit_fMRI = 0.0  # 初始化fMRI分类结果
        latent_list = []  # 存储每层的潜在特征
        minibatch_size, num_timepoints, num_nodes = fMRI_a.shape[:3]  # 获取fMRI数据的形状

        ############################### fMRI特征处理 ###############################
        attention = {'node-attention': [], 'time-attention': []}  # 存储节点和时间注意力信息
        h1 = fMRI_v  # 节点特征输入
        h2 = rearrange(h1, 'b t n c -> (b t n) c')  # 重塑形状以适配线性层
        h3 = self.initial_linear(h2)  # 初始线性变换，将输入特征嵌入到hidden_dim
        # 构建稀疏邻接矩阵
        a1 = self._collate_adjacency(fMRI_a)

        for layer, (G, R, T, L) in enumerate(
                zip(self.gnn_layers, self.readout_modules, self.transformer_modules, self.linear_layers)):
            h = G(h3, a1)  # GNN层，更新节点特征
            h_bridge = rearrange(h, '(b t n) c -> t b n c', t=num_timepoints, b=minibatch_size, n=num_nodes)
            h_readout, node_attn = R(h_bridge, node_axis=2)  # 特征读取模块，获取读取特征和节点注意力
            h_attend, time_attn = T(h_readout)  # Transformer层，获取时间依赖特征和时间注意力
            fMRI_latent = self.cls_token(h_attend)  # 聚合时间特征，获得最终的潜在特征
            logit_fMRI  = logit_fMRI + self.dropout(L(fMRI_latent))  # 通过线性层计算fMRI分类结果

            # 存储每层的节点和时间注意力
            attention['node-attention'].append(node_attn)
            attention['time-attention'].append(time_attn)
            latent_list.append(fMRI_latent)
        # 整合注意力信息
        attention['node-attention'] = torch.stack(attention['node-attention'], dim=1).detach().cpu()
        attention['time-attention'] = torch.stack(attention['time-attention'], dim=1).detach().cpu()
        fMRI_latent_1 = torch.stack(latent_list, dim=1)  # (batch_size, num_layers, hidden_dim)
        feat_fMRI = torch.mean(fMRI_latent_1, dim=1)  # 平均池化得到最终的fMRI特征

        return feat_fMRI,attention

    def _collate_adjacency(self, a):
        """
        构建稀疏邻接矩阵。
        """
        i_list, v_list = [], []
        for sample, _dyn_a in enumerate(a):
            for timepoint, _a in enumerate(_dyn_a):
                thresholded_a = (_a > _a.mean())  # 简单阈值化
                _i = thresholded_a.nonzero(as_tuple=False)
                _v = torch.ones(len(_i))
                _i =_i +( sample * a.shape[1] * a.shape[2] + timepoint * a.shape[2])
                i_list.append(_i)
                v_list.append(_v)
        _i = torch.cat(i_list).T.to(a.device)
        _v = torch.cat(v_list).to(a.device)
        return torch.sparse_coo_tensor(_i, _v, 
                                       (a.shape[0] * a.shape[1] * a.shape[2], a.shape[0] * a.shape[1] * a.shape[3]))



class LayerGIN(nn.Module):
    """
    GNN的Graph Isomorphism Network (GIN) 层，用于节点特征更新。
    """
    def __init__(self, input_dim, hidden_dim, output_dim, epsilon=True):
        super().__init__()
        if epsilon:
            self.epsilon = nn.Parameter(torch.Tensor([[0.0]]))  # 是否包含自环
        else:
            self.epsilon = 0.0
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim), nn.BatchNorm1d(output_dim), nn.ReLU())  # MLP用于特征映射

    def forward(self, v, a):
        """
        前向传播，更新节点特征。
        参数：
        - v: 节点特征，形状 (batch_size * segment_num * num_nodes, hidden_dim)
        - a: 稀疏邻接矩阵，形状 (batch_size * segment_num * num_nodes, batch_size * segment_num * num_nodes)
        返回：
        - v_combine: 更新后的节点特征
        """
        
        v_aggregate = torch.sparse.mm(a, v)  # 聚合邻居节点特征
        v_aggregate += self.epsilon * v  # 自环
        v_combine = self.mlp(v_aggregate)  # 通过MLP更新特征
        return v_combine
    
class ModuleMeanReadout(nn.Module):
    """
    简单的特征读取模块，通过对节点维度求均值获取图的特征表示。
    """
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, node_axis=1):
        """
        前向传播。
        参数：
        - x: 输入特征张量
        - node_axis: 节点的轴（维度）
        返回：
        - x.mean(node_axis): 对节点维度求均值的图特征
        - torch.zeros(...): 返回一个零张量，表示没有额外的注意力矩阵
        """
        return x.mean(node_axis), torch.zeros(size=[1, 1, 1], dtype=torch.float32)

class ModuleSERO(nn.Module):
    """
    自注意力读取模块，通过对节点特征进行注意力加权来计算全局特征。
    """
    def __init__(self, hidden_dim, input_dim, dropout=0.1, upscale=1.0):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(hidden_dim, round(upscale * hidden_dim)),  # 映射到更高维度
            nn.BatchNorm1d(round(upscale * hidden_dim)),
            nn.GELU())  # 激活函数
        self.attend = nn.Linear(round(upscale * hidden_dim), input_dim)  # 计算注意力权重
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, node_axis=1):
        """
        前向传播。
        参数：
        - x: 输入特征张量 (segment_num, batch_size, num_nodes, hidden_dim)
        - node_axis: 节点的轴（通常为2）
        返回：
        - 全局特征表示
        - 节点注意力权重
        """
        x_readout = x.mean(node_axis)  # 对节点维度求均值，形状为 (segment_num, batch_size, hidden_dim)
        x_shape = x_readout.shape  # 获取张量的形状
        x_embed = self.embed(x_readout.reshape(-1, x_shape[-1]))  # 通过嵌入层映射
        x_graphattention = torch.sigmoid(
            self.attend(x_embed)).view(*x_shape[:-1], -1)  # 计算注意力权重
        permute_idx = list(range(node_axis)) + [len(x_graphattention.shape) - 1] + list(range(node_axis, len(x_graphattention.shape) - 1))
        x_graphattention = x_graphattention.permute(permute_idx)  # 调整注意力张量的维度顺序
        return (x * self.dropout(x_graphattention.unsqueeze(-1))).mean(node_axis), x_graphattention.permute(1, 0, 2)
    
class ModuleGARO(nn.Module):
    """
    全局注意力读取模块，通过计算查询-键匹配获取节点的全局注意力。
    """
    def __init__(self, hidden_dim, dropout=0.1, upscale=1.0, **kwargs):
        super().__init__()
        self.embed_query = nn.Linear(hidden_dim, round(upscale * hidden_dim))  # 查询向量嵌入
        self.embed_key = nn.Linear(hidden_dim, round(upscale * hidden_dim))  # 键向量嵌入
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, node_axis=1):
        """
        前向传播。
        参数：
        - x: 输入特征张量，形状为 (..., num_nodes, ..., hidden_dim)
        - node_axis: 节点的轴
        返回：
        - 全局特征
        - 注意力矩阵
        """
        x_q = self.embed_query(x.mean(node_axis, keepdims=True))  # 对所有节点取均值后嵌入
        x_k = self.embed_key(x)  # 键向量嵌入
        x_graphattention = torch.sigmoid(
            torch.matmul(x_q, rearrange(x_k, 't b n c -> t b c n')) / np.sqrt(x_q.shape[-1])).squeeze(2)
        return (x * self.dropout(x_graphattention.unsqueeze(-1))).mean(node_axis), x_graphattention.permute(1, 0, 2)
    
class ModuleTransformer(nn.Module):
    """
    Transformer模块，用于建模时间依赖关系。
    """
    def __init__(self, input_dim, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(input_dim, num_heads)  # 多头注意力
        self.layer_norm1 = nn.LayerNorm(input_dim)  # 归一化层
        self.layer_norm2 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout)  # Dropout层
        self.dropout2 = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim))  # MLP用于特征更新

    def forward(self, x):
        """
        前向传播。
        参数：
        - x: 输入特征，形状为 (segment_num, batch_size, hidden_dim)
        返回：
        - x_attend: 时间依赖建模后的特征
        - attn_matrix: 时间步之间的注意力矩阵
        """
        x_attend, attn_matrix = self.multihead_attn(x, x, x)  # 自注意力
        x_attend = self.dropout1(x_attend)  # 添加Dropout
        x_attend = self.layer_norm1(x_attend)  # 第一层归一化
        x_attend2 = self.mlp(x_attend)  # 通过MLP更新特征
        x_attend = x_attend + self.dropout2(x_attend2)  # 残差连接
        x_attend = self.layer_norm2(x_attend)  # 第二层归一化
        return x_attend, attn_matrix  # 返回特征和注意力矩阵


