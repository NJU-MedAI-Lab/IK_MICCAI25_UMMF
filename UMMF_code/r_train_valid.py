 
from r_Encoders import * # 导入标准库模块
import argparse  # 用于命令行参数解析，帮助获取程序运行时传入的参数。
import os  # 提供操作系统相关的功能，如处理文件和目录。
# 导入 numpy 库，用于科学计算和数组操作
import numpy as np  # 引入 numpy 并命名为 np，以便更简洁地使用 numpy 的各种功能。
# 导入 PyTorch 库及其相关模块
import torch  # 导入 PyTorch 核心库，用于张量操作和 GPU 加速计算。
import torch.nn as nn  # 导入 PyTorch 的神经网络模块，用于构建和训练神经网络模型。
import torch.optim as optim  # 导入 PyTorch 的优化器模块，用于定义优化算法，如 SGD、Adam 等。
import torch.nn.functional as F  # 导入 PyTorch 的功能模块，用于调用不同的功能性函数，例如激活函数和损失函数。

# from models.fusion_modules import SumFusion, ConcatFusion, FiLM, GatedFusion, ConcatFusion3
# 从 models.fusion_modules 中导入多个融合模块：
# - SumFusion：将输入模态特征求和进行融合。
# - ConcatFusion：将输入模态特征拼接进行融合。
# - FiLM：基于特征变换的融合方法。
# - GatedFusion：基于门控机制的模态融合模块。
# - ConcatFusion3：多模态拼接融合，可能是结合三个输入模态。


# 导入 pickle 模块
import pickle  # 用于在文件中序列化和反序列化对象，例如保存和加载模型的状态字典、配置文件等。
# 导入 Python 调试工具 pdb
import pdb  # Python 的内置调试工具，用于设置断点并查看代码执行中的状态以进行调试。
# 导入 einops 库
import einops  # 一个用于重排列张量形状的库，方便地进行张量维度转换和操作。
# 例如，可以使用 einops 来重排图像张量的维度，以便于符合模型所需的输入格式。
# 导入 ml_collections 库的 ConfigDict 类
from ml_collections import ConfigDict  # 从 ml_collections 库中导入 ConfigDict，用于构建和管理模型的配置信息。
# ConfigDict 类似于一个字典，专门用于存储模型超参数、设置等信息，使得代码更易于管理和修改。 
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os


class GSPlugin():
    def __init__(self, gs_flag=True):
        # 构造函数，初始化GSPlugin对象
        super().__init__()

        dtype = torch.cuda.FloatTensor  # 定义数据类型为GPU上的浮点数
        with torch.no_grad():# 禁用梯度计算的主要目的是节省内存和计算资源，因为这里的操作并不需要进行反向传播，也不需要计算梯度。
            # self.Pl = torch.autograd.Variable(torch.eye(768).type(dtype)) # 之前的代码，矩阵为768维
            self.Pl = torch.autograd.Variable(torch.eye(512).type(dtype))  # 创建一个512x512的单位矩阵，并将其转化为不需要梯度更新的变量
        self.exp_count = 0  # 初始化训练次数计数器

    # @torch.no_grad()
    def before_update(self, model, before_batch_input, batch_index, len_dataloader, train_exp_counter):
    # 在每次参数更新前调用的函数，用于调整梯度
        """	
        输入参数：
            model：要调整梯度的模型。
            before_batch_input：当前 batch 的输入数据，用于计算输入特征的均值。
            batch_index：当前 batch 在整个数据集中的索引，表示训练过程中的当前进度。
            len_dataloader：数据集的总 batch 数量，用于计算动态的参数值。
            train_exp_counter：训练次数计数器，表示当前训练是第几次。
        """
        lamda = batch_index / len_dataloader + 1  # 计算lambda值，用于动态调整alpha
        alpha = 1.0 * 0.1 ** lamda  # 根据lambda计算alpha值，用于矩阵更新

        if train_exp_counter != 0:
            for n, w in model.named_parameters():
                if n == "module.weight":  # 只对名为 "module.weight" 的参数进行处理
                    r = torch.mean(before_batch_input, 0, keepdim=True)  # 使用 keepdim=True 保持均值张量的维度一致
                    k = torch.mm(self.Pl, r.T)  # 计算 Pl 矩阵与 r 的转置的乘积

                    # 更新 Pl 矩阵，执行低秩更新 - 使用 out-of-place 操作
                    self.Pl = self.Pl - torch.mm(k, k.T) / (alpha + torch.mm(k.T, r))

                    # 使用 Frobenius 范数对 Pl 矩阵进行正则化
                    pnorm2 = torch.norm(self.Pl, p='fro')
                    epsilon = 1e-8  # 添加 epsilon 防止除零错误
                    self.Pl = self.Pl / (pnorm2 + epsilon)  # 正则化 Pl

                    # 使用更新后的 Pl 矩阵调整模型参数的梯度
                    # 使用 out-of-place 操作，防止修改视图引起的问题
                    w.grad = torch.mm(w.grad, self.Pl.T).clone()
    def before_update_1(self, model, before_batch_input, batch_index, len_dataloader, train_exp_counter):
        # 在每次参数更新前调用的函数，用于调整梯度
        """	
        输入参数：
            model：要调整梯度的模型。
            before_batch_input：当前 batch 的输入数据，用于计算输入特征的均值。
            batch_index：当前 batch 在整个数据集中的索引，表示训练过程中的当前进度。
            len_dataloader：数据集的总 batch 数量，用于计算动态的参数值。
            train_exp_counter：训练次数计数器，表示当前训练是第几次。
        """
        lamda = batch_index / len_dataloader + 1  # 计算lambda值，用于动态调整alpha
        alpha = 1.0 * 0.1 ** lamda  # 根据lambda计算alpha值，用于矩阵更新
        # x_mean = torch.mean(strategy.mb_x, 0, True)  # 计算样本均值的注释掉代码
        if train_exp_counter != 0:
            for n, w in model.named_parameters():
                # 遍历模型的参数
                if n == "module.weight":
                    # 只对名为"module.weight"的参数进行处理
                    r = torch.mean(before_batch_input, 0, True)  # 计算当前batch输入的均值，沿着第0个维度
                    k = torch.mm(self.Pl, torch.t(r))  # 计算Pl矩阵与r的转置的乘积
                    # 更新Pl矩阵，执行低秩更新
                    self.Pl = torch.sub(self.Pl, torch.mm(k, torch.t(k)) / (alpha + torch.mm(k, r)))

                    # Frobenius范数用于正则化矩阵，确保矩阵数值稳定
                    pnorm2 = torch.norm(self.Pl.data, p='fro')
                    self.Pl.data = self.Pl.data / pnorm2  # 将Pl矩阵的每个元素除以其Frobenius范数，使其归一化

                    # 使用更新后的Pl矩阵调整模型参数的梯度
                    w.grad.data = torch.mm(w.grad.data, torch.t(self.Pl.data))




class ConcatFusion3(nn.Module):
    def __init__(self, input_dim=512, output_dim=100):
        super(ConcatFusion3, self).__init__()
        self.fc_out = nn.Linear(input_dim, output_dim)

    def forward(self, x, y, z):
        output = torch.cat((x, y, z), dim=1)
        output = self.fc_out(output)
        return x, y, z, output

class SumFusion(nn.Module):
    def __init__(self, input_dim=512, output_dim=100):
        super(ConcatFusion3, self).__init__()
        self.fc_out = nn.Linear(input_dim, output_dim)

    def forward(self, x, y, z):
        output = torch.cat((x, y, z), dim=1)
        output = self.fc_out(output)
        return x, y, z, output



class Modal3Classifier(nn.Module):
    def __init__(self, args):
        super(Modal3Classifier, self).__init__()

        # 获取融合方法类型
        fusion = 'concat'

        # 设置数据集类别数，HIVC 数据集有 2 类
        if args.dataset == 'HIVC':
            n_classes = 2

        # 根据融合方法选择不同的融合模块
        if fusion == 'sum':
            # 使用 Sum 融合方法
            self.fusion_module = SumFusion(output_dim=n_classes)
        elif fusion == 'concat':
            # 使用 Concatenation 融合方法
            if args.gs_flag:
                # 使用 ConcatFusion3 进行融合，输入维度为 768，输出维度为 n_classes
                self.fusion_module = ConcatFusion3(input_dim=256, output_dim=n_classes)
                '''
                class ConcatFusion3(nn.Module):
                    def __init__(self, input_dim=512, output_dim=100):
                        super(ConcatFusion3, self).__init__()
                        self.fc_out = nn.Linear(input_dim, output_dim)

                    def forward(self, x, y, z):
                        output = torch.cat((x, y, z), dim=1)
                        output = self.fc_out(output)
                        return x, y, z, output
                '''
        elif fusion == 'film':
            # 如果使用 FILM 融合方法，这部分还未实现
            pass
        elif fusion == 'gated':
            # 如果使用 Gated 融合方法，这部分还未实现
            pass
        else:
            # 如果给定的融合方法不正确，抛出错误
            raise NotImplementedError('Incorrect fusion method: {}!'.format(fusion))

        # 设置模型配置
        model_config = ConfigDict(dict(model_type='base'))

        # 初始化三个模态的编码器
        self.mae_a = fMRIClassifier(input_dim=116, hidden_dim=256, num_classes=2) # fMRI-音频模态编码器
        self.mae_v = BNCNN(example=torch.zeros(1, 1, 116, 116))  # DTI模态编码器
        self.mae_t = sMRIClassifier()  # 文本模态编码器

        self.args = args

    def forward(self, dyn_v, dyn_a, DTI, sMRI):

        # 前向传播fMRI模态
        a, attention= self.mae_a.forward(dyn_v, dyn_a)
        # 前向传播DTI模态
        v = self.mae_v.forward(DTI.float())
        # 前向传播视觉模态
        t = self.mae_t.forward(sMRI)

        # 对fMRI、DTI和文本模态进行平均池化

        return a, v, t,attention

class CrossModalFusion(nn.Module):
    def __init__(self):
        super(CrossModalFusion, self).__init__()

    def forward(self, feature_p, feature_q, feature_r, uncertainty_p, uncertainty_q, uncertainty_r):
        # 融合方式：残差融合
        # 计算残差连接后的特征融合，将原始特征和通过不确定性加权的特征结合
        fusion_p = feature_p + (feature_p * uncertainty_q + feature_p * uncertainty_r)
        fusion_q = feature_q + (feature_q * uncertainty_p + feature_q * uncertainty_r)
        fusion_r = feature_r + (feature_r * uncertainty_p + feature_r * uncertainty_q)
        return fusion_p, fusion_q, fusion_r



def train_epoch(args, epoch, model, device, dataloader, optimizer, scheduler,
                gs_plugin=None, writer=None, gs_flag=True, av_alpha=0.5,
                txt_history=None, img_history=None, audio_history=None):
    """
    进行一个 epoch 的训练。

    参数:
    - args: 包含从命令行解析出来的各种配置项的对象 (Namespace)。
    - epoch: 当前训练的 epoch 数 (int)，用于记录训练轮次。
    - model: 要被训练的深度学习模型实例 (nn.Module)。
    - device: 模型训练的设备 (torch.device)，如 GPU 或 CPU。
    - dataloader: 数据加载器 (DataLoader)，用于提供训练数据。
    - optimizer: 优化器 (torch.optim.Optimizer)，用于优化模型的参数。
    - scheduler: 学习率调度器 (torch.optim.lr_scheduler)，用于调整学习率。
    - gs_plugin: 用于特殊梯度修正的插件 (GSPlugin) 或 `None`，用于修正模型中的梯度。
    - writer: TensorBoard 日志记录器 (SummaryWriter) 或 `None`，用于将训练过程中的数据写入日志进行可视化。
    - gs_flag: 是否启用 GS 模式 (bool)，用于控制是否使用 `gs_plugin` 修正梯度。
    - av_alpha: 控制音频和视觉模态损失加权比例的系数 (float)，用于加权融合不同模态的损失。
    - txt_history: 记录文本模态训练历史的对象 (History) 或 `None`，用于追踪和分析文本模态的表现。
    - img_history: 记录视觉模态训练历史的对象 (History) 或 `None`，用于追踪和分析视觉模态的表现。
    - audio_history: 记录音频模态训练历史的对象 (History) 或 `None`，用于追踪和分析音频模态的表现。

    返回:
    - 每个模态和总的平均损失值。
    """
    # 定义损失函数为交叉熵损失
    criterion = nn.CrossEntropyLoss()
    # 定义激活函数 Softmax、ReLU 和 Tanh
    softmax = nn.Softmax(dim=1)
    relu = nn.ReLU()
    tanh = nn.Tanh()

    # 设置模型为训练模式
    model.train()
    print("Start training ... ")

    # 初始化损失值，用于记录整个 epoch 中各模态的损失
    _loss = 0
    _loss_a = 0  # 音频模态的损失
    _loss_v = 0  # 视觉模态的损失
    _loss_t = 0  # 文本模态的损失
    len_dataloader = len(dataloader)  # 数据加载器的长度
    window_size = 40
    window_stride = 20
    dynamic_length = 200
    for batch_step, data_packet in enumerate(dataloader):
        # 根据参数 args.lorb 和 args.modal3 来处理输入数据
        # 获取输入数据，包括文本 token、padding 掩码、图像、音频谱图、标签和样本索引
        idx = data_packet['idx']
        subjid = data_packet['subjid:']
        fMRI = data_packet['fMRI']  # 提取 fMRI 数据
        sMRI_trad = data_packet['sMRI_trad']  # 提取 sMRI 传统特征
        sMRI_deep_brainspace = data_packet['sMRI_deep_brainspace']  # 提取 sMRI 深度特征 (brainspace)
        sMRI_deep_mnispace = data_packet['sMRI_deep_mnispace']  # 提取 sMRI 深度特征 (MNI space)
        DTI = data_packet['DTI']  # 提取 DTI 数据
        label = data_packet['label']  # 提取标签

        fMRI_data = fMRI
        sMRI_data = sMRI_deep_brainspace
        #sMRI_data = sMRI_deep_brainspace
        DTI_data = DTI
        sMRI_data = sMRI_data.unsqueeze(1)
        sMRI_data = sMRI_data / 255.0
        DTI_data = DTI_data.unsqueeze(1)
        # 将数据移动到指定设备（例如 GPU）
        dyn_fc, sampling_points = process_dynamic_fc(
            fMRI_data,  # 增加 batch 维度
            window_size=window_size,
            window_stride=window_stride,
            dynamic_length=dynamic_length
        )
        dyn_v = repeat(torch.eye(dyn_fc.shape[-1]), 'n1 n2 -> b t n1 n2', t=dyn_fc.shape[1], b=6)



        # 在反向传播之前，先将优化器的梯度缓存清零
        optimizer.zero_grad()

        # 如果启用了 gs_plugin
        if gs_flag:
            # 进行前向传播，得到音频、视觉和文本模态的输出
            dyn_v = dyn_v.float().to(device)
            dyn_fc = dyn_fc.float().to(device)
            DTI_data = DTI_data.float().to(device)
            sMRI_data = sMRI_data.float().to(device)
            label = label.long().to(device)
            a, v, t,_ = model(dyn_v, dyn_fc, DTI_data, sMRI_data)

            # 通过模型的融合模块对音频模态进行输出
            out_a = model.fusion_module.fc_out(a.clone())
            # 计算音频模态的损失
            loss_a = criterion(out_a, label.clone())
            # 反向传播计算音频模态的梯度
            loss_a.backward(retain_graph=True)

            # 使用 gs_plugin 修正音频模态的梯度
            gs_plugin.before_update(model.fusion_module.fc_out, a, 
                                    batch_step, len_dataloader, gs_plugin.exp_count)
            # 使用优化器更新模型参数
            optimizer.step()
            # 再次清空梯度缓存，以便处理下一个模态
            optimizer.zero_grad()

            # 更新 gs_plugin 的计数器
            gs_plugin.exp_count += 1
            
            # 对视觉模态进行相似的处理
            out_v = model.fusion_module.fc_out(v.clone())
            loss_v = criterion(out_v, label.clone())
            loss_v.backward(retain_graph=True)

            # 使用 gs_plugin 修正视觉模态的梯度
            gs_plugin.before_update(model.fusion_module.fc_out, v, 
                                    batch_step, len_dataloader, gs_plugin.exp_count)
            # 使用优化器更新模型参数
            optimizer.step()
            optimizer.zero_grad()

            # 更新 gs_plugin 的计数器
            gs_plugin.exp_count += 1

            # 如果启用了文本模态的处理
            out_t = model.fusion_module.fc_out(t.clone())
            # 计算文本模态的损失
            loss_t = criterion(out_t, label.clone())
            # 反向传播计算文本模态的梯度
            loss_t.backward(retain_graph=True)

            # 使用 gs_plugin 修正文本模态的梯度
            gs_plugin.before_update(model.fusion_module.fc_out, t, 
                                    batch_step, len_dataloader, gs_plugin.exp_count)
            optimizer.step()
            optimizer.zero_grad()

            # 更新 gs_plugin 的计数器
            gs_plugin.exp_count += 1
            #print(f"model_a_out:{out_a},model_v_out:{out_v}, model_t_out:{out_t}")
            # 清除所有模型参数的梯度缓存，确保没有残留梯度
            for n, p in model.named_parameters():
                if p.grad is not None:
                    del p.grad

            # 累加各个模态的损失，用于统计
            _loss += (loss_a  + loss_v + loss_t).item()
            _loss_a += loss_a.item()
            _loss_v += loss_v.item()
            _loss_t += loss_t.item()

        # 如果未启用 gs_plugin，则打印错误信息并退出程序
        else:
            print("MLA do not support this mode")
            exit(0)
        del a, v, t, out_a, out_v, out_t, loss_a, loss_v, loss_t
        torch.cuda.empty_cache()
    # 调整学习率
    scheduler.step()
    
    # 返回每个模态的平均损失
    
    return _loss / len_dataloader, _loss_a / len_dataloader, _loss_v / len_dataloader, _loss_t / len_dataloader

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class RandomNetworkPrediction(nn.Module):
    def __init__(self, input_length=2, hidden_dim=64, output_dim=2):
        super(RandomNetworkPrediction, self).__init__()

        # 随机初始化网络，不可训练
        self.random_network = nn.Sequential(
            nn.Linear(input_length, 64),  # 第一层全连接
            nn.LeakyReLU(negative_slope=2.5 * 10**-1),  # 激活函数 LeakyReLU，负斜率为 0.25
            nn.Linear(64, 128),  # 第二层全连接
            nn.LeakyReLU(negative_slope=2.5 * 10**-1),  # 激活函数 LeakyReLU
            nn.Linear(128, output_dim)  # 第三层全连接
        )
        for param in self.random_network.parameters():
            param.requires_grad = False  # 将随机网络的参数设置为不可训练

        # 可训练的预测网络
        self.prediction_network = nn.Sequential(
            nn.Linear(input_length, 64),  # 第一层全连接
            nn.LeakyReLU(negative_slope=2.5 * 10**-1),  # 激活函数 LeakyReLU
            nn.Linear(64, 128),  # 第二层全连接
            nn.LeakyReLU(negative_slope=2.5 * 10**-1),  # 激活函数 LeakyReLU
            nn.Linear(128, output_dim)  # 第三层全连接
        )

        # 初始化权重
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 对于 Linear 层，使用正交初始化
                init.orthogonal_(m.weight, gain=init.calculate_gain('leaky_relu', param=2.5 * 10**-1))
                if m.bias is not None:
                    m.bias.data.zero_()  # 将偏置初始化为 0

    def forward(self, x):
        # 通过随机网络的输出
        random_output = self.random_network(x)  # (batch_size, output_dim)
        # 通过预测网络的输出
        predicted_output = self.prediction_network(x)  # (batch_size, output_dim)
        # 计算两个特征之间的平方误差
        uncertainty_map = (predicted_output - random_output).pow(2)/ 2  # (batch_size,)
        return uncertainty_map, random_output, predicted_output  # 返回不确定性图、随机网络输出和预测网络输出

class MultiModalRNPModel(nn.Module):
    def __init__(self, input_length=2, hidden_dim=64, output_dim=2):
        super(MultiModalRNPModel, self).__init__()
        # 三个 RNP 模块，每个模态对应一个
        self.rnp_1 = RandomNetworkPrediction(input_length, hidden_dim, output_dim)  # 第一个 RNP 模块
        self.rnp_2 = RandomNetworkPrediction(input_length, hidden_dim, output_dim)  # 第二个 RNP 模块
        self.rnp_3 = RandomNetworkPrediction(input_length, hidden_dim, output_dim)  # 第三个 RNP 模块

    def forward(self, x_1, x_2, x_3):
        # 计算每个模态的不确定性图
        uncertainty_1, random_output_1, predicted_output_1 = self.rnp_1(x_1)  # 计算第一个模态的不确定性图及输出
        uncertainty_2, random_output_2, predicted_output_2 = self.rnp_2(x_2)  # 计算第二个模态的不确定性图及输出
        uncertainty_3, random_output_3, predicted_output_3 = self.rnp_3(x_3)  # 计算第三个模态的不确定性图及输出

        # 返回每个模态的不确定性图、随机网络输出和预测网络输出
        return (uncertainty_1, random_output_1, predicted_output_1,
                uncertainty_2, random_output_2, predicted_output_2,
                uncertainty_3, random_output_3, predicted_output_3)

class RandomNetworkPrediction_256(nn.Module):
    def __init__(self, input_length=256, hidden_dim=128, output_dim=2):
        super(RandomNetworkPrediction_256, self).__init__()

        # 随机初始化网络，不可训练
        self.random_network = nn.Sequential(
            nn.Linear(input_length, 128),  # 第一层全连接
            nn.LeakyReLU(negative_slope=2.5 * 10**-1),  # 激活函数 LeakyReLU，负斜率为 0.25
            nn.Linear(128, 64),  # 第二层全连接
            nn.LeakyReLU(negative_slope=2.5 * 10**-1),  # 激活函数 LeakyReLU
            nn.Linear(64, output_dim)  # 第三层全连接
        )
        for param in self.random_network.parameters():
            param.requires_grad = False  # 将随机网络的参数设置为不可训练

        # 可训练的预测网络
        self.prediction_network = nn.Sequential(
            nn.Linear(input_length, 128),  # 第一层全连接
            nn.LeakyReLU(negative_slope=2.5 * 10**-1),  # 激活函数 LeakyReLU
            nn.Linear(128, 64),  # 第二层全连接
            nn.LeakyReLU(negative_slope=2.5 * 10**-1),  # 激活函数 LeakyReLU
            nn.Linear(64, output_dim)  # 第三层全连接
        )

        # 初始化权重
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 对于 Linear 层，使用正交初始化
                init.orthogonal_(m.weight, gain=init.calculate_gain('leaky_relu', param=2.5 * 10**-1))
                if m.bias is not None:
                    m.bias.data.zero_()  # 将偏置初始化为 0

    def forward(self, x):
        # 通过随机网络的输出
        random_output = self.random_network(x)  # (batch_size, output_dim)
        # 通过预测网络的输出
        predicted_output = self.prediction_network(x)  # (batch_size, output_dim)
        # 计算两个特征之间的平方误差
        uncertainty_map = (predicted_output - random_output).pow(2)/ 2  # (batch_size,)
        return uncertainty_map, random_output, predicted_output  # 返回不确定性图、随机网络输出和预测网络输出

class MultiModalRNPModel_256(nn.Module):
    def __init__(self, input_length=256, hidden_dim=128, output_dim=2):
        super(MultiModalRNPModel_256, self).__init__()
        # 三个 RNP 模块，每个模态对应一个
        self.rnp_1 = RandomNetworkPrediction_256(input_length, hidden_dim, output_dim)  # 第一个 RNP 模块
        self.rnp_2 = RandomNetworkPrediction_256(input_length, hidden_dim, output_dim)  # 第二个 RNP 模块
        self.rnp_3 = RandomNetworkPrediction_256(input_length, hidden_dim, output_dim)  # 第三个 RNP 模块
    def forward(self, x_1, x_2, x_3):
    # 计算每个模态的不确定性图
        uncertainty_1, random_output_1, predicted_output_1 = self.rnp_1(x_1)  # 计算第一个模态的不确定性图及输出
        uncertainty_2, random_output_2, predicted_output_2 = self.rnp_2(x_2)  # 计算第二个模态的不确定性图及输出
        uncertainty_3, random_output_3, predicted_output_3 = self.rnp_3(x_3)  # 计算第三个模态的不确定性图及输出
        # 返回每个模态的不确定性图、随机网络输出和预测网络输出
        return (uncertainty_1, random_output_1, predicted_output_1,
                uncertainty_2, random_output_2, predicted_output_2,
                uncertainty_3, random_output_3, predicted_output_3)

# Training function
def train_RNP(model, device, dataloader, rnp_model=MultiModalRNPModel,learning_rate=0.0005):
    criterion_mse = nn.MSELoss()
    print("Start RNP_training ... ")
    n_classes = 2
    len_dataloader = len(dataloader)
    window_size = 40
    window_stride = 20
    dynamic_length = 200
    optimizer = optim.Adam(rnp_model.parameters(), lr=learning_rate)  # 使用 Adam 优化器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80, eta_min=1e-20)
    rnp_model.train()  # 切换模型到训练模式
    model.eval()
    total_loss_1, total_loss_2, total_loss_3 = 0.0, 0.0, 0.0  # 初始化每个模态的总损失
    for step, data_packet in enumerate(dataloader):
        # 根据 `args.lorb` 和 `args.modal3` 来处理输入数据
        # 获取输入数据，包括文本 token、padding 掩码、图像、音频谱图、标签和样本索引
        idx = data_packet['idx']
        subjid = data_packet['subjid:']
        fMRI = data_packet['fMRI']  # 提取 fMRI 数据
        sMRI_trad = data_packet['sMRI_trad']  # 提取 sMRI 传统特征
        sMRI_deep_brainspace = data_packet['sMRI_deep_brainspace']  # 提取 sMRI 深度特征 (brainspace)
        sMRI_deep_mnispace = data_packet['sMRI_deep_mnispace']  # 提取 sMRI 深度特征 (MNI space)
        DTI = data_packet['DTI']  # 提取 DTI 数据
        label = data_packet['label']  # 提取标签

        fMRI_data = fMRI
        sMRI_data = sMRI_deep_brainspace
        # sMRI_data = sMRI_deep_brainspace
        DTI_data = DTI
        sMRI_data = sMRI_data.unsqueeze(1)
        sMRI_data = sMRI_data / 255.0
        DTI_data = DTI_data.unsqueeze(1)
        # 将数据移动到指定设备（例如 GPU）
        dyn_fc, sampling_points = process_dynamic_fc(
            fMRI_data,  # 增加 batch 维度
            window_size=window_size,
            window_stride=window_stride,
            dynamic_length=dynamic_length
        )
        dyn_v = repeat(torch.eye(dyn_fc.shape[-1]), 'n1 n2 -> b t n1 n2', t=dyn_fc.shape[1], b=6)       
        # 如果启用了 GS 插件
        # 进行前向传播，得到音频、视觉和文本模态的输出特征
        dyn_v = dyn_v.float().to(device)
        dyn_fc = dyn_fc.float().to(device)
        DTI_data = DTI_data.float().to(device)
        sMRI_data = sMRI_data.float().to(device)
        label = label.long().to(device)
        a, v, t,_ = model(dyn_v, dyn_fc, DTI_data, sMRI_data)
        # 通过模型的融合模块对每个模态的输出进行分类
        out_a = model.fusion_module.fc_out(a)  # 音频模态的输出
        out_v = model.fusion_module.fc_out(v)  # 视觉模态的输出
        out_t = model.fusion_module.fc_out(t)  # 文本模态的输出
        (uncertainty_1, random_output_1, predicted_output_1,
            uncertainty_2, random_output_2, predicted_output_2,
            uncertainty_3, random_output_3, predicted_output_3) = rnp_model(out_a, out_v, out_t)
        
        # 损失函数计算
        losses = []

        # 对于每个模态分别计算损失
        for random_output, predicted_output, rnp in zip(
                [random_output_1, random_output_2, random_output_3],
                [predicted_output_1, predicted_output_2, predicted_output_3],
                [rnp_model.rnp_1, rnp_model.rnp_2, rnp_model.rnp_3]):
            
            # 计算 MSE 部分的损失
            mse_loss = F.mse_loss(predicted_output, random_output)

            # 计算正则化项 R(ω) = ||ω||^2
            l2_reg = 0
            for param in rnp.parameters():
                l2_reg += torch.norm(param, 2) ** 2
            
            regularization_factor=1e-4
            # 组合 MSE 损失和正则化项
            total_loss = mse_loss + regularization_factor * l2_reg

            # 将损失加入损失列表
            losses.append(total_loss)
        #print(f"训练batch{step}:整体RNP_loss的情况{losses}")
        # 更新模型参数
        optimizer.zero_grad()  # 清空梯度
        for loss in losses:
            loss.backward(retain_graph=True)  # 反向传播每个模态的损失
        optimizer.step()  # 更新模型参数

        # # 添加梯度裁剪，避免梯度爆炸
        # max_norm = 2.0  # 设置最大范数（可以根据经验调试）
        # torch.nn.utils.clip_grad_norm_(rnp_model.parameters(), max_norm)
        # 累加每个模态的总损失
        total_loss_1 += losses[0].item()
        total_loss_2 += losses[1].item()
        total_loss_3 += losses[2].item()
        # 调整学习率
    scheduler.step()
    # 打印每个模态的平均损失
    print("RNP_Training finished.")
    return total_loss_1 / len_dataloader, total_loss_2 / len_dataloader, total_loss_3 / len_dataloader

# Training function
def train_RNP_mlp(model, device, dataloader, rnp_model=MultiModalRNPModel_256,learning_rate=0.0008):
    criterion_mse = nn.MSELoss()
    print("Start RNP_training ... ")
    n_classes = 2
    len_dataloader = len(dataloader)
    window_size = 40
    window_stride = 20
    dynamic_length = 200
    optimizer = optim.Adam(rnp_model.parameters(), lr=learning_rate)  # 使用 Adam 优化器
    rnp_model.train()  # 切换模型到训练模式
    model.eval()
    total_loss_1, total_loss_2, total_loss_3 = 0.0, 0.0, 0.0  # 初始化每个模态的总损失
    for step, data_packet in enumerate(dataloader):
        # 根据 `args.lorb` 和 `args.modal3` 来处理输入数据
        # 获取输入数据，包括文本 token、padding 掩码、图像、音频谱图、标签和样本索引
        idx = data_packet['idx']
        subjid = data_packet['subjid:']
        fMRI = data_packet['fMRI']  # 提取 fMRI 数据
        sMRI_trad = data_packet['sMRI_trad']  # 提取 sMRI 传统特征
        sMRI_deep_brainspace = data_packet['sMRI_deep_brainspace']  # 提取 sMRI 深度特征 (brainspace)
        sMRI_deep_mnispace = data_packet['sMRI_deep_mnispace']  # 提取 sMRI 深度特征 (MNI space)
        DTI = data_packet['DTI']  # 提取 DTI 数据
        label = data_packet['label']  # 提取标签

        fMRI_data = fMRI
        sMRI_data = sMRI_deep_mnispace
        # sMRI_data = sMRI_deep_brainspace
        DTI_data = DTI
        sMRI_data = sMRI_data.unsqueeze(1)
        sMRI_data = sMRI_data / 255.0
        DTI_data = DTI_data.unsqueeze(1)
        # 将数据移动到指定设备（例如 GPU）
        dyn_fc, sampling_points = process_dynamic_fc(
            fMRI_data,  # 增加 batch 维度
            window_size=window_size,
            window_stride=window_stride,
            dynamic_length=dynamic_length
        )
        dyn_v = repeat(torch.eye(dyn_fc.shape[-1]), 'n1 n2 -> b t n1 n2', t=dyn_fc.shape[1], b=6)       
        # 如果启用了 GS 插件
        # 进行前向传播，得到音频、视觉和文本模态的输出特征
        dyn_v = dyn_v.float().to(device)
        dyn_fc = dyn_fc.float().to(device)
        DTI_data = DTI_data.float().to(device)
        sMRI_data = sMRI_data.float().to(device)
        label = label.long().to(device)
        a, v, t = model(dyn_v, dyn_fc, DTI_data, sMRI_data)
        # 通过模型的融合模块对每个模态的输出进行分类
        (uncertainty_1, random_output_1, predicted_output_1,
            uncertainty_2, random_output_2, predicted_output_2,
            uncertainty_3, random_output_3, predicted_output_3) = rnp_model(a, v, t)
        
        # 损失函数计算
        losses = []

        # 对于每个模态分别计算损失
        for random_output, predicted_output, rnp in zip(
                [random_output_1, random_output_2, random_output_3],
                [predicted_output_1, predicted_output_2, predicted_output_3],
                [rnp_model.rnp_1, rnp_model.rnp_2, rnp_model.rnp_3]):
            
            # 计算 MSE 部分的损失
            mse_loss = F.mse_loss(predicted_output, random_output)

            # 计算正则化项 R(ω) = ||ω||^2
            l2_reg = 0
            for param in rnp.parameters():
                l2_reg += torch.norm(param, 2) ** 2
            
            regularization_factor=1e-4
            # 组合 MSE 损失和正则化项
            total_loss = mse_loss + regularization_factor * l2_reg

            # 将损失加入损失列表
            losses.append(total_loss)
        #print(f"训练batch{step}:整体RNP_loss的情况{losses}")
        # 更新模型参数
        optimizer.zero_grad()  # 清空梯度
        for loss in losses:
            loss.backward(retain_graph=True)  # 反向传播每个模态的损失
        optimizer.step()  # 更新模型参数

        # # 添加梯度裁剪，避免梯度爆炸
        # max_norm = 2.0  # 设置最大范数（可以根据经验调试）
        # torch.nn.utils.clip_grad_norm_(rnp_model.parameters(), max_norm)
        # 累加每个模态的总损失
        total_loss_1 += losses[0].item()
        total_loss_2 += losses[1].item()
        total_loss_3 += losses[2].item()

    # 打印每个模态的平均损失
    print("RNP_Training finished.")
    return total_loss_1 / len_dataloader, total_loss_2 / len_dataloader, total_loss_3 / len_dataloader









import torch
import torch.nn.functional as F

def calculate_entropy(output, epsilon=1e-10):
    # 通过 softmax 将输出转换为概率分布，使用 dim=1 以适应 2D 输出
    probabilities = F.softmax(output, dim=1)

    # 计算每个概率的对数，使用 epsilon 避免 log(0)
    log_probabilities = torch.log(probabilities + epsilon)

    # 根据熵公式 H(X) = -sum(p(x) * log(p(x))) 计算熵值
    entropy = -torch.sum(probabilities * log_probabilities, dim=1).mean()

    # 返回计算出的熵
    return entropy

def calculate_gating_weights3(encoder_output_1, encoder_output_2, encoder_output_3):
    # 计算每个编码器输出的熵
    entropy_1 = calculate_entropy(encoder_output_1)
    entropy_2 = calculate_entropy(encoder_output_2)
    entropy_3 = calculate_entropy(encoder_output_3)
    
    # 找到三个熵值中最大的熵
    max_entropy = torch.max(torch.tensor([entropy_1, entropy_2, entropy_3]))
    
    # 根据最大熵与各模态熵的差值，计算每个模态的门控权重
    # 熵越低，权重越高，指数函数使得权重的变化更敏感
    gating_weight_1 = torch.exp(torch.clamp(max_entropy - entropy_1, min=-50, max=50))
    gating_weight_2 = torch.exp(torch.clamp(max_entropy - entropy_2, min=-50, max=50))
    gating_weight_3 = torch.exp(torch.clamp(max_entropy - entropy_3, min=-50, max=50))
    
    # 计算三个权重的总和，确保不会为 0
    sum_weights = gating_weight_1 + gating_weight_2 + gating_weight_3 + 1e-10
    
    # 对每个权重进行归一化，确保所有权重的和等于 1
    gating_weight_1 /= sum_weights
    gating_weight_2 /= sum_weights
    gating_weight_3 /= sum_weights
    
    # 返回三个模态的归一化门控权重
    return gating_weight_1, gating_weight_2, gating_weight_3

def valid(args, model, device, dataloader, 
          gs_flag=True):

    # 定义 softmax 激活函数，将模型输出转换为概率分布
    softmax = nn.Softmax(dim=1)

    # 设置类别数，根据数据集类型设置，这里默认是 HIVC 数据集，共有 4 个类别
    if args.dataset == "HIVC":
        n_classes = 2
    window_size = 40
    window_stride = 20
    dynamic_length = 200

    # 禁用梯度计算，用于推理或验证，以节省内存
    with torch.no_grad():
        # 设置模型为验证模式
        model.eval()
        
        # 初始化各类样本计数和准确率
        num = [0.0 for _ in range(n_classes)]  # 每个类别的样本数量
        acc = [0.0 for _ in range(n_classes)]  # 每个类别的总准确率
        acc_a = [0.0 for _ in range(n_classes)]  # fMRI模态的准确率
        acc_v = [0.0 for _ in range(n_classes)]  # DTI模态的准确率
        acc_t = [0.0 for _ in range(n_classes)]  # sMRI模态的准确率
        pred_result = []  # 预测结果

        # 遍历数据加载器中的每个 batch
        for step, data_packet in enumerate(dataloader):
            # 根据 `args.lorb` 和 `args.modal3` 来处理输入数据
            # 获取输入数据，包括文本 token、padding 掩码、图像、音频谱图、标签和样本索引
            idx = data_packet['idx']
            subjid = data_packet['subjid:']
            fMRI = data_packet['fMRI']  # 提取 fMRI 数据
            sMRI_trad = data_packet['sMRI_trad']  # 提取 sMRI 传统特征
            sMRI_deep_brainspace = data_packet['sMRI_deep_brainspace']  # 提取 sMRI 深度特征 (brainspace)
            sMRI_deep_mnispace = data_packet['sMRI_deep_mnispace']  # 提取 sMRI 深度特征 (MNI space)
            DTI = data_packet['DTI']  # 提取 DTI 数据
            label = data_packet['label']  # 提取标签

            fMRI_data = fMRI
            sMRI_data = sMRI_deep_mnispace
            # sMRI_data = sMRI_deep_brainspace
            DTI_data = DTI
            sMRI_data = sMRI_data.unsqueeze(1)
            sMRI_data = sMRI_data / 255.0
            DTI_data = DTI_data.unsqueeze(1)
            # 将数据移动到指定设备（例如 GPU）
            dyn_fc, sampling_points = process_dynamic_fc(
                fMRI_data,  # 增加 batch 维度
                window_size=window_size,
                window_stride=window_stride,
                dynamic_length=dynamic_length
            )
            dyn_v = repeat(torch.eye(dyn_fc.shape[-1]), 'n1 n2 -> b t n1 n2', t=dyn_fc.shape[1], b=6)

                    
            # 如果启用了 GS 插件
            if gs_flag:
                # 进行前向传播，得到音频、视觉和文本模态的输出特征
                dyn_v = dyn_v.float().to(device)
                dyn_fc = dyn_fc.float().to(device)
                DTI_data = DTI_data.float().to(device)
                sMRI_data = sMRI_data.float().to(device)
                label = label.long().to(device)
                a, v, t = model(dyn_v, dyn_fc, DTI_data, sMRI_data)

                # 通过模型的融合模块对每个模态的输出进行分类
                out_a = model.fusion_module.fc_out(a)  # 音频模态的输出
                out_v = model.fusion_module.fc_out(v)  # 视觉模态的输出
                out_t = model.fusion_module.fc_out(t)  # 文本模态的输出

                # print(out_a,out_v,out_t)
                audio_conf, img_conf, txt_conf = calculate_gating_weights3(out_a, out_v, out_t)

                # 根据计算出的门控权重对三个模态的输出进行加权融合
                out = (out_a * audio_conf + out_v * img_conf + out_t * txt_conf)

            # 使用 softmax 对融合后的输出进行归一化，得到预测概率
            prediction = softmax(out)
            pred_v = softmax(out_v)
            pred_a = softmax(out_a)
            pred_t = softmax(out_t)
            print(prediction,pred_a,pred_v,pred_t)
            # 对 batch 中的每个样本进行评估
            for i in range(fMRI_data.shape[0]):
                # 使用 `np.argmax` 找到概率最大的位置（即预测的类别）
                ma = np.argmax(prediction[i].cpu().data.numpy())
                v = np.argmax(pred_v[i].cpu().data.numpy())
                a = np.argmax(pred_a[i].cpu().data.numpy())
                t = np.argmax(pred_t[i].cpu().data.numpy())

                # 更新每个类别的样本数量
                num[label[i]] += 1.0

                # 更新各模态的准确率
                if np.asarray(label[i].cpu()) == ma:
                    acc[label[i]] += 1.0
                if np.asarray(label[i].cpu()) == v:
                    acc_v[label[i]] += 1.0
                if np.asarray(label[i].cpu()) == a:
                    acc_a[label[i]] += 1.0
                if np.asarray(label[i].cpu()) == t:
                    acc_t[label[i]] += 1.0
    

    # 计算各模态的总准确率并返回

    return sum(acc) / sum(num), sum(acc_a) / sum(num), sum(acc_v) / sum(num), sum(acc_t) / sum(num)

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score, confusion_matrix

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score, confusion_matrix

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score, confusion_matrix

def valid_pro(args, model, device, dataloader, gs_flag=True):
    # 定义 softmax 激活函数，将模型输出转换为概率分布
    softmax = nn.Softmax(dim=1)

    # 设置类别数，根据数据集类型设置，这里默认是 HIVC 数据集，共有 2 个类别
    if args.dataset == "HIVC":
        n_classes = 2
    window_size = 40
    window_stride = 20
    dynamic_length = 200

    # 禁用梯度计算，用于推理或验证，以节省内存
    with torch.no_grad():
        # 设置模型为验证模式
        model.eval()

        # 初始化各类样本计数和准确率
        all_labels = []
        all_predictions = []
        all_probs = []

        acc_a = [0.0 for _ in range(n_classes)]  # fMRI模态的准确率
        acc_v = [0.0 for _ in range(n_classes)]  # DTI模态的准确率
        acc_t = [0.0 for _ in range(n_classes)]  # sMRI模态的准确率

        # 遍历数据加载器中的每个 batch
        for step, data_packet in enumerate(dataloader):
            # 获取输入数据，包括 fMRI、sMRI、DTI 等
            fMRI = data_packet['fMRI']
            sMRI_deep_mnispace = data_packet['sMRI_deep_mnispace']
            DTI = data_packet['DTI']
            label = data_packet['label']

            fMRI_data = fMRI
            sMRI_data = sMRI_deep_mnispace.unsqueeze(1) / 255.0
            DTI_data = DTI.unsqueeze(1)

            # 检查输入数据是否有 NaN
            if torch.isnan(fMRI_data).any() or torch.isnan(sMRI_data).any() or torch.isnan(DTI_data).any():
                print(f"Input data contains NaN values at step {step}. Skipping this batch.")
                continue

            # 将数据移动到指定设备（例如 GPU）
            dyn_fc, sampling_points = process_dynamic_fc(fMRI_data, window_size=window_size, window_stride=window_stride, dynamic_length=dynamic_length)
            dyn_v = repeat(torch.eye(dyn_fc.shape[-1]), 'n1 n2 -> b t n1 n2', t=dyn_fc.shape[1], b=fMRI_data.shape[0])

            dyn_v = dyn_v.float().to(device)
            dyn_fc = dyn_fc.float().to(device)
            DTI_data = DTI_data.float().to(device)
            sMRI_data = sMRI_data.float().to(device)
            label = label.long().to(device)

            # 前向传播，得到各模态的输出
            a, v, t,_ = model(dyn_v, dyn_fc, DTI_data, sMRI_data)

            # 通过模型的融合模块对每个模态的输出进行分类
            out_a = model.fusion_module.fc_out(a)
            out_v = model.fusion_module.fc_out(v)
            out_t = model.fusion_module.fc_out(t)

            # print(out_a,out_v,out_t)
            # 计算权重并融合
            audio_conf, img_conf, txt_conf = calculate_gating_weights3(out_a, out_v, out_t)
            out = (out_a * audio_conf + out_v * img_conf + out_t * txt_conf)

            # 使用 softmax 对融合后的输出进行归一化，得到预测概率
            prediction_probs = softmax(out)
            pred_v_probs = softmax(out_v)
            pred_a_probs = softmax(out_a)
            pred_t_probs = softmax(out_t)



            # 记录真实标签和预测概率
            all_labels.extend(label.cpu().numpy())
            all_probs.extend(prediction_probs.cpu().numpy())
            all_predictions.extend(np.argmax(prediction_probs.cpu().numpy(), axis=1))

            # 对 batch 中的每个样本进行评估
            for i in range(fMRI_data.shape[0]):
                if label[i].item() == torch.argmax(pred_v_probs[i]).item():
                    acc_v[label[i]] += 1.0
                if label[i].item() == torch.argmax(pred_a_probs[i]).item():
                    acc_a[label[i]] += 1.0
                if label[i].item() == torch.argmax(pred_t_probs[i]).item():
                    acc_t[label[i]] += 1.0

    # 将所有的标签、预测值和预测概率转换为 NumPy 数组
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_predictions = np.array(all_predictions)


    # 计算各种指标
    acc = accuracy_score(all_labels, all_predictions)
    auc = roc_auc_score(all_labels, all_probs[:, 1]) if n_classes == 2 else roc_auc_score(all_labels, all_probs, multi_class="ovr")
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    sen = recall_score(all_labels, all_predictions, average='weighted')  # 敏感性（召回率）
    pre = precision_score(all_labels, all_predictions, average='weighted', zero_division=1)

    # 计算特异性（需要通过混淆矩阵）
    if n_classes == 2:
        tn, fp, fn, tp = confusion_matrix(all_labels, all_predictions).ravel()
        spe = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        spe = 0  # 对于多分类问题，特异性计算需要进一步处理

    # 计算各模态的准确率
    acc_a = sum(acc_a) / len(all_labels)
    acc_v = sum(acc_v) / len(all_labels)
    acc_t = sum(acc_t) / len(all_labels)

    # 返回计算的指标
    return acc, auc, f1, sen, spe, pre, acc_a, acc_v, acc_t

def rnp_valid_pro(args, model, device, dataloader,rnp_model, gs_flag=True,):
    # 定义 softmax 激活函数，将模型输出转换为概率分布
    softmax = nn.Softmax(dim=1)

    # 设置类别数，根据数据集类型设置，这里默认是 HIVC 数据集，共有 2 个类别
    if args.dataset == "HIVC":
        n_classes = 2
    window_size = 40
    window_stride = 20
    dynamic_length = 200

    # 禁用梯度计算，用于推理或验证，以节省内存
    with torch.no_grad():
        # 设置模型为验证模式
        model.eval()
        rnp_model.eval()
        # 初始化各类样本计数和准确率
        all_labels = []
        all_predictions = []
        all_probs = []

        acc_a = [0.0 for _ in range(n_classes)]  # fMRI模态的准确率
        acc_v = [0.0 for _ in range(n_classes)]  # DTI模态的准确率
        acc_t = [0.0 for _ in range(n_classes)]  # sMRI模态的准确率

        # 遍历数据加载器中的每个 batch
        for step, data_packet in enumerate(dataloader):
            # 获取输入数据，包括 fMRI、sMRI、DTI 等
            fMRI = data_packet['fMRI']
            sMRI_deep_mnispace = data_packet['sMRI_deep_mnispace']
            sMRI_deep_brainspace = data_packet['sMRI_deep_brainspace']
            DTI = data_packet['DTI']
            label = data_packet['label']

            fMRI_data = fMRI
            sMRI_data = sMRI_deep_brainspace.unsqueeze(1) / 255.0
            DTI_data = DTI.unsqueeze(1)


            # 将数据移动到指定设备（例如 GPU）
            dyn_fc, sampling_points = process_dynamic_fc(fMRI_data, window_size=window_size, window_stride=window_stride, dynamic_length=dynamic_length)
            dyn_v = repeat(torch.eye(dyn_fc.shape[-1]), 'n1 n2 -> b t n1 n2', t=dyn_fc.shape[1], b=fMRI_data.shape[0])

            dyn_v = dyn_v.float().to(device)
            dyn_fc = dyn_fc.float().to(device)
            DTI_data = DTI_data.float().to(device)
            sMRI_data = sMRI_data.float().to(device)
            label = label.long().to(device)

            # 前向传播，得到各模态的输出
            a, v, t,_ = model(dyn_v, dyn_fc, DTI_data, sMRI_data)

            # 通过模型的融合模块对每个模态的输出进行分类
            out_a = model.fusion_module.fc_out(a)
            out_v = model.fusion_module.fc_out(v)
            out_t = model.fusion_module.fc_out(t)
            (uncertainty_a, random_output_1, predicted_output_1,
                uncertainty_v, random_output_2, predicted_output_2,
                uncertainty_t, random_output_3, predicted_output_3) = rnp_model(out_a, out_v, out_t)
            # print(out_a,out_v,out_t)
            # 计算权重并融合
            #audio_conf, img_conf, txt_conf = calculate_gating_weights3(out_a, out_v, out_t)
            # print(uncertainty_a.shape)
            # print(out_a.shape)

            sum_u = (uncertainty_a+uncertainty_v+uncertainty_t)*2 + 1e-12
            u_a = uncertainty_a/sum_u
            u_v = uncertainty_v/sum_u
            u_t = uncertainty_t/sum_u
            #print(f"在batch{step}中：fMRI不确定性：{u_a},DTI不确定性{u_v},sMRI不确定性{u_t}")
            # audio_conf, img_conf, txt_conf = calculate_gating_weights3(out_a, out_v, out_t)
            # out = (out_a * audio_conf + out_v * img_conf + out_t * txt_conf)
            # out = (out_a * (u_v+u_t) * audio_conf + out_v * (u_a+u_t) * img_conf + out_t * (u_a+u_v) * txt_conf)
            out = (out_a * (u_v+u_t) + out_v * (u_a+u_t) + out_t * (u_a+u_v))
            # 使用 softmax 对融合后的输出进行归一化，得到预测概率
            prediction_probs = softmax(out)
            pred_v_probs = softmax(out_v)
            pred_a_probs = softmax(out_a)
            pred_t_probs = softmax(out_t)



            # 记录真实标签和预测概率
            all_labels.extend(label.cpu().numpy())
            all_probs.extend(prediction_probs.cpu().numpy())
            all_predictions.extend(np.argmax(prediction_probs.cpu().numpy(), axis=1))

            # 对 batch 中的每个样本进行评估
            for i in range(fMRI_data.shape[0]):
                if label[i].item() == torch.argmax(pred_v_probs[i]).item():
                    acc_v[label[i]] += 1.0
                if label[i].item() == torch.argmax(pred_a_probs[i]).item():
                    acc_a[label[i]] += 1.0
                if label[i].item() == torch.argmax(pred_t_probs[i]).item():
                    acc_t[label[i]] += 1.0

    # 将所有的标签、预测值和预测概率转换为 NumPy 数组
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_predictions = np.array(all_predictions)


    # 计算各种指标
    acc = accuracy_score(all_labels, all_predictions)
    auc = roc_auc_score(all_labels, all_probs[:, 1]) if n_classes == 2 else roc_auc_score(all_labels, all_probs, multi_class="ovr")
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    sen = recall_score(all_labels, all_predictions, average='weighted')  # 敏感性（召回率）
    pre = precision_score(all_labels, all_predictions, average='weighted', zero_division=1)

    # 计算特异性（需要通过混淆矩阵）
    if n_classes == 2:
        tn, fp, fn, tp = confusion_matrix(all_labels, all_predictions).ravel()
        spe = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        spe = 0  # 对于多分类问题，特异性计算需要进一步处理

    # 计算各模态的准确率
    acc_a = sum(acc_a) / len(all_labels)
    acc_v = sum(acc_v) / len(all_labels)
    acc_t = sum(acc_t) / len(all_labels)

    # 返回计算的指标
    return acc, auc, f1, sen, spe, pre, acc_a, acc_v, acc_t


def rnp_valid_pro_mlp(args, model, device, dataloader,rnp_model, gs_flag=True,):
    # 定义 softmax 激活函数，将模型输出转换为概率分布
    softmax = nn.Softmax(dim=1)

    # 设置类别数，根据数据集类型设置，这里默认是 HIVC 数据集，共有 2 个类别
    if args.dataset == "HIVC":
        n_classes = 2
    window_size = 40
    window_stride = 20
    dynamic_length = 200

    # 禁用梯度计算，用于推理或验证，以节省内存
    with torch.no_grad():
        # 设置模型为验证模式
        model.eval()
        rnp_model.eval()
        # 初始化各类样本计数和准确率
        all_labels = []
        all_predictions = []
        all_probs = []

        acc_a = [0.0 for _ in range(n_classes)]  # fMRI模态的准确率
        acc_v = [0.0 for _ in range(n_classes)]  # DTI模态的准确率
        acc_t = [0.0 for _ in range(n_classes)]  # sMRI模态的准确率

        # 遍历数据加载器中的每个 batch
        for step, data_packet in enumerate(dataloader):
            # 获取输入数据，包括 fMRI、sMRI、DTI 等
            fMRI = data_packet['fMRI']
            sMRI_deep_mnispace = data_packet['sMRI_deep_mnispace']
            DTI = data_packet['DTI']
            label = data_packet['label']

            fMRI_data = fMRI
            sMRI_data = sMRI_deep_mnispace.unsqueeze(1) / 255.0
            DTI_data = DTI.unsqueeze(1)


            # 将数据移动到指定设备（例如 GPU）
            dyn_fc, sampling_points = process_dynamic_fc(fMRI_data, window_size=window_size, window_stride=window_stride, dynamic_length=dynamic_length)
            dyn_v = repeat(torch.eye(dyn_fc.shape[-1]), 'n1 n2 -> b t n1 n2', t=dyn_fc.shape[1], b=fMRI_data.shape[0])

            dyn_v = dyn_v.float().to(device)
            dyn_fc = dyn_fc.float().to(device)
            DTI_data = DTI_data.float().to(device)
            sMRI_data = sMRI_data.float().to(device)
            label = label.long().to(device)

            # 前向传播，得到各模态的输出
            a, v, t = model(dyn_v, dyn_fc, DTI_data, sMRI_data)

            # 通过模型的融合模块对每个模态的输出进行分类
            out_a = model.fusion_module.fc_out(a)
            out_v = model.fusion_module.fc_out(v)
            out_t = model.fusion_module.fc_out(t)
            (uncertainty_a, random_output_1, predicted_output_1,
                uncertainty_v, random_output_2, predicted_output_2,
                uncertainty_t, random_output_3, predicted_output_3) = rnp_model(a, v, t)
            # print(out_a,out_v,out_t)
            # 计算权重并融合
            #audio_conf, img_conf, txt_conf = calculate_gating_weights3(out_a, out_v, out_t)
            # print(uncertainty_a.shape)
            # print(out_a.shape)

            sum_u = (uncertainty_a+uncertainty_v+uncertainty_t)*2 + 1e-12
            u_a = uncertainty_a/sum_u
            u_v = uncertainty_v/sum_u
            u_t = uncertainty_t/sum_u
            #print(f"在batch{step}中：fMRI不确定性：{u_a},DTI不确定性{u_v},sMRI不确定性{u_t}")
            out = (out_a * (u_v+u_t) + out_v * (u_a+u_t) + out_t * (u_a+u_v))

            # 使用 softmax 对融合后的输出进行归一化，得到预测概率
            prediction_probs = softmax(out)
            pred_v_probs = softmax(out_v)
            pred_a_probs = softmax(out_a)
            pred_t_probs = softmax(out_t)



            # 记录真实标签和预测概率
            all_labels.extend(label.cpu().numpy())
            all_probs.extend(prediction_probs.cpu().numpy())
            all_predictions.extend(np.argmax(prediction_probs.cpu().numpy(), axis=1))

            # 对 batch 中的每个样本进行评估
            for i in range(fMRI_data.shape[0]):
                if label[i].item() == torch.argmax(pred_v_probs[i]).item():
                    acc_v[label[i]] += 1.0
                if label[i].item() == torch.argmax(pred_a_probs[i]).item():
                    acc_a[label[i]] += 1.0
                if label[i].item() == torch.argmax(pred_t_probs[i]).item():
                    acc_t[label[i]] += 1.0

    # 将所有的标签、预测值和预测概率转换为 NumPy 数组
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_predictions = np.array(all_predictions)


    # 计算各种指标
    acc = accuracy_score(all_labels, all_predictions)
    auc = roc_auc_score(all_labels, all_probs[:, 1]) if n_classes == 2 else roc_auc_score(all_labels, all_probs, multi_class="ovr")
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    sen = recall_score(all_labels, all_predictions, average='weighted')  # 敏感性（召回率）
    pre = precision_score(all_labels, all_predictions, average='weighted', zero_division=1)

    # 计算特异性（需要通过混淆矩阵）
    if n_classes == 2:
        tn, fp, fn, tp = confusion_matrix(all_labels, all_predictions).ravel()
        spe = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        spe = 0  # 对于多分类问题，特异性计算需要进一步处理

    # 计算各模态的准确率
    acc_a = sum(acc_a) / len(all_labels)
    acc_v = sum(acc_v) / len(all_labels)
    acc_t = sum(acc_t) / len(all_labels)

    # 返回计算的指标
    return acc, auc, f1, sen, spe, pre, acc_a, acc_v, acc_t

def train_epoch_fMRI(args, epoch, model, device, dataloader, optimizer, scheduler,
                gs_plugin=None, writer=None, gs_flag=True, av_alpha=0.5,
                txt_history=None, img_history=None, audio_history=None):

    # 定义损失函数为交叉熵损失
    criterion = nn.CrossEntropyLoss()


    # 设置模型为训练模式
    model.train()
    print("Start training ... ")


    _loss_a = 0  # 音频模态的损失

    len_dataloader = len(dataloader)  # 数据加载器的长度
    window_size = 40
    window_stride = 20
    dynamic_length = 200
    for batch_step, data_packet in enumerate(dataloader):
        # 根据参数 args.lorb 和 args.modal3 来处理输入数据
        # 获取输入数据，包括文本 token、padding 掩码、图像、音频谱图、标签和样本索引
        idx = data_packet['idx']
        subjid = data_packet['subjid:']
        fMRI = data_packet['fMRI']  # 提取 fMRI 数据
        sMRI_trad = data_packet['sMRI_trad']  # 提取 sMRI 传统特征
        sMRI_deep_brainspace = data_packet['sMRI_deep_brainspace']  # 提取 sMRI 深度特征 (brainspace)
        sMRI_deep_mnispace = data_packet['sMRI_deep_mnispace']  # 提取 sMRI 深度特征 (MNI space)
        DTI = data_packet['DTI']  # 提取 DTI 数据
        label = data_packet['label']  # 提取标签

        fMRI_data = fMRI
        sMRI_data = sMRI_deep_mnispace
        #sMRI_data = sMRI_deep_brainspace
        DTI_data = DTI
        sMRI_data = sMRI_data.unsqueeze(1)
        sMRI_data = sMRI_data / 255.0
        DTI_data = DTI_data.unsqueeze(1)
        # 将数据移动到指定设备（例如 GPU）
        dyn_fc, sampling_points = process_dynamic_fc(
            fMRI_data,  # 增加 batch 维度
            window_size=window_size,
            window_stride=window_stride,
            dynamic_length=dynamic_length
        )
        dyn_v = repeat(torch.eye(dyn_fc.shape[-1]), 'n1 n2 -> b t n1 n2', t=dyn_fc.shape[1], b=6)



        # 在反向传播之前，先将优化器的梯度缓存清零
        optimizer.zero_grad()

        # 如果启用了 gs_plugin
        if gs_flag:
            # 进行前向传播，得到音频、视觉和文本模态的输出
            dyn_v = dyn_v.float().to(device)
            dyn_fc = dyn_fc.float().to(device)
            DTI_data = DTI_data.float().to(device)
            sMRI_data = sMRI_data.float().to(device)
            label = label.long().to(device)
            a = model(dyn_v, dyn_fc)

            # 通过模型的融合模块对音频模态进行输出
            out_a = model.fusion_module.fc_out(a.clone())
            # 计算音频模态的损失
            loss_a = criterion(out_a, label.clone())
            # 反向传播计算音频模态的梯度
            loss_a.backward(retain_graph=True)

            # 使用优化器更新模型参数
            optimizer.step()

            # 清除梯度缓存，以便下次计算
            optimizer.zero_grad()
            # 清除所有模型参数的梯度缓存，确保没有残留梯度
            for n, p in model.named_parameters():
                if p.grad is not None:
                    del p.grad

            _loss_a += loss_a.item()

        # 如果未启用 gs_plugin，则打印错误信息并退出程序
        else:
            print("MLA do not support this mode")
            exit(0)

    # 调整学习率
    scheduler.step()
    
    # 返回每个模态的平均损失
    
    return _loss_a / len_dataloader

# def train_epsoch_DTI(args, epoch, model, device, dataloader, optimizer, scheduler,
#                 gs_plugin=None, writer=None, gs_flag=True, av_alpha=0.5,
#                 txt_history=None, img_history=None, audio_history=None):

#     # 定义损失函数为交叉熵损失
#     criterion = nn.CrossEntropyLoss()


#     # 设置模型为训练模式
#     model.train()
#     print("Start training ... ")


#     _loss_v = 0
#     len_dataloader = len(dataloader)  # 数据加载器的长度

#     for batch_step, data_packet in enumerate(dataloader):
#         # 根据参数 args.lorb 和 args.modal3 来处理输入数据
#         # 获取输入数据，包括文本 token、padding 掩码、图像、音频谱图、标签和样本索引
#         idx = data_packet['idx']
#         subjid = data_packet['subjid:']
#         fMRI = data_packet['fMRI']  # 提取 fMRI 数据
#         sMRI_trad = data_packet['sMRI_trad']  # 提取 sMRI 传统特征
#         sMRI_deep_brainspace = data_packet['sMRI_deep_brainspace']  # 提取 sMRI 深度特征 (brainspace)
#         sMRI_deep_mnispace = data_packet['sMRI_deep_mnispace']  # 提取 sMRI 深度特征 (MNI space)
#         DTI = data_packet['DTI']  # 提取 DTI 数据
#         label = data_packet['label']  # 提取标签

#         fMRI_data = fMRI
#         sMRI_data = sMRI_deep_mnispace
#         #sMRI_data = sMRI_deep_brainspace
#         DTI_data = DTI
#         sMRI_data = sMRI_data.unsqueeze(1)
#         sMRI_data = sMRI_data / 255.0
#         DTI_data = DTI_data.unsqueeze(1)


#         # 在反向传播之前，先将优化器的梯度缓存清零
#         optimizer.zero_grad()

#         # 如果启用了 gs_plugin
#         if gs_flag:
#             # 进行前向传播，得到音频、视觉和文本模态的输出

#             DTI_data = DTI_data.float().to(device)
#             sMRI_data = sMRI_data.float().to(device)
#             label = label.long().to(device)
#             v = model(DTI_data)

#             # 对视觉模态进行相似的处理
#             out_v = model.fusion_module.fc_out(v.clone())
#             loss_v = criterion(out_v, label.clone())
#             loss_v.backward(retain_graph=True)

#             # 使用优化器更新模型参数
#             optimizer.step()


#             # for n, p in model.named_parameters():
#             #     if p.grad is not None:
#             #         del p.grad

#             _loss_v += loss_v.item()


#     # 调整学习率
#     scheduler.step()
    
#     # 返回每个模态的平均损失
    
#     return  _loss_v / len_dataloader

def train_epsoch_DTI(args, epoch, model, device, dataloader, optimizer, scheduler,
                gs_plugin=None, writer=None, gs_flag=True, av_alpha=0.5,
                txt_history=None, img_history=None, audio_history=None):

    # 定义损失函数为交叉熵损失
    criterion = nn.CrossEntropyLoss()

    # 设置模型为训练模式
    model.train()
    print("Start training ... ")

    _loss_v = 0
    correct_predictions = 0  # 统计正确预测数
    total_samples = 0  # 总样本数
    len_dataloader = len(dataloader)  # 数据加载器的长度

    for batch_step, data_packet in enumerate(dataloader):
        # 获取输入数据
        idx = data_packet['idx']
        subjid = data_packet['subjid:']
        fMRI = data_packet['fMRI']  # 提取 fMRI 数据
        sMRI_trad = data_packet['sMRI_trad']  # 提取 sMRI 传统特征
        sMRI_deep_brainspace = data_packet['sMRI_deep_brainspace']  # 提取 sMRI 深度特征 (brainspace)
        sMRI_deep_mnispace = data_packet['sMRI_deep_mnispace']  # 提取 sMRI 深度特征 (MNI space)
        DTI = data_packet['DTI']  # 提取 DTI 数据
        label = data_packet['label']  # 提取标签

        fMRI_data = fMRI
        sMRI_data = sMRI_deep_mnispace
        DTI_data = DTI
        sMRI_data = sMRI_data.unsqueeze(1)
        sMRI_data = sMRI_data / 255.0
        DTI_data = DTI_data.unsqueeze(1)

        # 在反向传播之前，先将优化器的梯度缓存清零
        optimizer.zero_grad()

        if gs_flag:
            # 数据移动到设备上
            DTI_data = DTI_data.float().to(device)
            sMRI_data = sMRI_data.float().to(device)
            label = label.long().to(device)

            # 前向传播
            v = model(DTI_data)
            out_v = model.fusion_module.fc_out(v.clone())

            # 计算损失
            loss_v = criterion(out_v, label.clone())
            loss_v.backward(retain_graph=True)

            # 更新模型参数
            optimizer.step()

            # 累加损失
            _loss_v += loss_v.item()

            # 计算准确率
            _, preds = torch.max(out_v, 1)  # 获取预测类别
            correct_predictions += torch.sum(preds == label).item()  # 累加正确预测数
            total_samples += label.size(0)  # 累加总样本数

    # 调整学习率
    scheduler.step()

    # 计算平均损失和准确率
    avg_loss = _loss_v / len_dataloader
    accuracy = correct_predictions / total_samples

    # print(f"Epoch {epoch}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}")
    
    # 返回每个模态的平均损失和准确率
    return avg_loss, accuracy

# def train_epoch_sMRI(args, epoch, model, device, dataloader, optimizer, scheduler,
#                      gs_plugin=None, writer=None, gs_flag=True, av_alpha=0.5,
#                      txt_history=None, img_history=None, audio_history=None):

#     # 定义损失函数为交叉熵损失
#     criterion = nn.CrossEntropyLoss()

#     # 设置模型为训练模式
#     model.train()
#     print("Start training ... ")

#     # 初始化统计变量
#     correct_predictions = 0
#     total_samples = 0
#     _loss_t = 0  # 音频模态的损失

#     len_dataloader = len(dataloader)  # 数据加载器的长度

#     for batch_step, data_packet in enumerate(dataloader):
#         # 获取输入数据
#         idx = data_packet['idx']
#         subjid = data_packet['subjid:']
#         fMRI = data_packet['fMRI']
#         sMRI_deep_mnispace = data_packet['sMRI_deep_mnispace']
#         sMRI_deep_brainspace = data_packet['sMRI_deep_brainspace']
#         label = data_packet['label']

#         # 数据预处理
#         # sMRI_data = sMRI_deep_mnispace.unsqueeze(1) / 255.0
#         sMRI_data = sMRI_deep_brainspace.unsqueeze(1) / 255.0

#         label = label.long().to(device)

#         # 清除梯度
#         optimizer.zero_grad()

#         if gs_flag:
#             # 前向传播
#             sMRI_data = sMRI_data.float().to(device)
#             t = model(sMRI_data)
#             out_t = model.fusion_module.fc_out(t)

#             # 计算损失并反向传播
#             loss_t = criterion(out_t, label)
#             loss_t.backward()
#             optimizer.step()

#             # 预测类别并统计正确率
#             _, preds = torch.max(out_t, 1)
#             correct_predictions += torch.sum(preds == label).item()
#             total_samples += label.size(0)
#             _loss_t += loss_t.item()
#         else:
#             raise ValueError("MLA does not support this mode")

#     # 调整学习率
#     scheduler.step()

#     # 计算平均损失和准确率
#     accuracy = correct_predictions / total_samples
#     loss = _loss_t / len_dataloader
#     return loss, accuracy

from torch.amp import GradScaler, autocast

def train_epoch_sMRI(args, epoch, model, device, dataloader, optimizer, scheduler,
                     gs_plugin=None, writer=None, gs_flag=True, av_alpha=0.5,
                     txt_history=None, img_history=None, audio_history=None):
    # 定义损失函数为交叉熵损失
    criterion = nn.CrossEntropyLoss()

    # 设置模型为训练模式
    model.train()
    print("Start training ... ")

    # 初始化统计变量
    correct_predictions = 0
    total_samples = 0
    _loss_t = 0  # 损失累计
    len_dataloader = len(dataloader)  # 数据加载器的长度

    # 混合精度
    scaler = GradScaler()

    for batch_step, data_packet in enumerate(dataloader):
        # 获取输入数据
        sMRI_data = data_packet['sMRI_deep_brainspace'].unsqueeze(1) / 255.0
        label = data_packet['label'].long().to(device)
        sMRI_data = sMRI_data.float().to(device)

        # 清除梯度
        optimizer.zero_grad()

        if gs_flag:
            # 前向传播
            with autocast(device_type='cuda', dtype=torch.float16):
                t = model(sMRI_data)
                out_t = model.fusion_module.fc_out(t)
                loss_t = criterion(out_t, label)

            # 反向传播
            scaler.scale(loss_t).backward()
            scaler.step(optimizer)
            scaler.update()

            # 统计准确率
            with torch.no_grad():
                _, preds = torch.max(out_t, 1)
                correct_predictions += torch.sum(preds == label).item()
                total_samples += label.size(0)

            # 累加损失
            _loss_t += loss_t.item()

            # 显式删除变量，释放显存
            del t, out_t, loss_t
        else:
            raise ValueError("MLA does not support this mode")

    # 调整学习率
    scheduler.step()

    # 计算平均损失和准确率
    accuracy = correct_predictions / total_samples
    loss = _loss_t / len_dataloader
    return loss, accuracy


class Modal_sMRI(nn.Module):
    def __init__(self, args):
        super(Modal_sMRI, self).__init__()

        # 获取融合方法类型
        fusion = 'concat'

        # 设置数据集类别数，HIVC 数据集有 2 类
        if args.dataset == 'HIVC':
            n_classes = 2

        # 根据融合方法选择不同的融合模块
        if fusion == 'sum':
            # 使用 Sum 融合方法
            self.fusion_module = SumFusion(output_dim=n_classes)
        elif fusion == 'concat':
            # 使用 Concatenation 融合方法
            if args.gs_flag:
                # 使用 ConcatFusion3 进行融合，输入维度为 768，输出维度为 n_classes
                self.fusion_module = ConcatFusion3(input_dim=256, output_dim=n_classes)
                '''
                class ConcatFusion3(nn.Module):
                    def __init__(self, input_dim=512, output_dim=100):
                        super(ConcatFusion3, self).__init__()
                        self.fc_out = nn.Linear(input_dim, output_dim)

                    def forward(self, x, y, z):
                        output = torch.cat((x, y, z), dim=1)
                        output = self.fc_out(output)
                        return x, y, z, output
                '''
        elif fusion == 'film':
            # 如果使用 FILM 融合方法，这部分还未实现
            pass
        elif fusion == 'gated':
            # 如果使用 Gated 融合方法，这部分还未实现
            pass
        else:
            # 如果给定的融合方法不正确，抛出错误
            raise NotImplementedError('Incorrect fusion method: {}!'.format(fusion))

        # 设置模型配置

        # 初始化三个模态的编码器
        self.mae_t = sMRIClassifier()  # 文本模态编码器

        self.args = args

    def forward(self,sMRI):

        # 前向传播视觉模态
        t = self.mae_t.forward(sMRI)

        # 对fMRI、DTI和文本模态进行平均池化

        return t

class Modal_fMRI(nn.Module):
    def __init__(self, args):
        super(Modal_fMRI, self).__init__()

        # 获取融合方法类型
        fusion = 'concat'

        # 设置数据集类别数，HIVC 数据集有 2 类
        if args.dataset == 'HIVC':
            n_classes = 2

        # 根据融合方法选择不同的融合模块
        if fusion == 'sum':
            # 使用 Sum 融合方法
            self.fusion_module = SumFusion(output_dim=n_classes)
        elif fusion == 'concat':
            # 使用 Concatenation 融合方法
            if args.gs_flag:
                # 使用 ConcatFusion3 进行融合，输入维度为 768，输出维度为 n_classes
                self.fusion_module = ConcatFusion3(input_dim=256, output_dim=n_classes)
                '''
                class ConcatFusion3(nn.Module):
                    def __init__(self, input_dim=512, output_dim=100):
                        super(ConcatFusion3, self).__init__()
                        self.fc_out = nn.Linear(input_dim, output_dim)

                    def forward(self, x, y, z):
                        output = torch.cat((x, y, z), dim=1)
                        output = self.fc_out(output)
                        return x, y, z, output
                '''
        elif fusion == 'film':
            # 如果使用 FILM 融合方法，这部分还未实现
            pass
        elif fusion == 'gated':
            # 如果使用 Gated 融合方法，这部分还未实现
            pass
        else:
            # 如果给定的融合方法不正确，抛出错误
            raise NotImplementedError('Incorrect fusion method: {}!'.format(fusion))

        # 初始化三个模态的编码器
        self.mae_a = fMRIClassifier(input_dim=116, hidden_dim=256, num_classes=2) # fMRI-音频模态编码器

        self.args = args

    def forward(self, dyn_v, dyn_a):

        # 前向传播fMRI模态
        a = self.mae_a.forward(dyn_v, dyn_a)

        # 对fMRI、DTI和文本模态进行平均池化

        return a   



class Modal_DTI(nn.Module):

    def __init__(self, args):
        super(Modal_DTI, self).__init__()

        # 获取融合方法类型
        fusion = 'concat'

        # 设置数据集类别数，HIVC 数据集有 2 类
        if args.dataset == 'HIVC':
            n_classes = 2

        # 根据融合方法选择不同的融合模块
        if fusion == 'sum':
            # 使用 Sum 融合方法
            self.fusion_module = SumFusion(output_dim=n_classes)
        elif fusion == 'concat':
            # 使用 Concatenation 融合方法
            if args.gs_flag:
                # 使用 ConcatFusion3 进行融合，输入维度为 768，输出维度为 n_classes
                self.fusion_module = ConcatFusion3(input_dim=256, output_dim=n_classes)
                '''
                class ConcatFusion3(nn.Module):
                    def __init__(self, input_dim=512, output_dim=100):
                        super(ConcatFusion3, self).__init__()
                        self.fc_out = nn.Linear(input_dim, output_dim)

                    def forward(self, x, y, z):
                        output = torch.cat((x, y, z), dim=1)
                        output = self.fc_out(output)
                        return x, y, z, output
                '''
        elif fusion == 'film':
            # 如果使用 FILM 融合方法，这部分还未实现
            pass
        elif fusion == 'gated':
            # 如果使用 Gated 融合方法，这部分还未实现
            pass
        else:
            # 如果给定的融合方法不正确，抛出错误
            raise NotImplementedError('Incorrect fusion method: {}!'.format(fusion))


        # 初始化三个模态的编码器
        self.mae_v = BNCNN(example=torch.zeros(1, 1, 116, 116))  # DTI模态编码器
  
        self.args = args

    def forward(self, DTI):

        # 前向传播DTI模态
        v = self.mae_v.forward(DTI.float())

        # 对fMRI、DTI和文本模态进行平均池化

        return v
    
def valid_pro_fMRI(args, model, device, dataloader, gs_flag=True):
    # 定义 softmax 激活函数，将模型输出转换为概率分布
    softmax = nn.Softmax(dim=1)

    # 设置类别数，根据数据集类型设置，这里默认是 HIVC 数据集，共有 2 个类别
    if args.dataset == "HIVC":
        n_classes = 2
    window_size = 40
    window_stride = 20
    dynamic_length = 200

    # 禁用梯度计算，用于推理或验证，以节省内存
    with torch.no_grad():
        # 设置模型为验证模式
        model.eval()

        # 初始化各类样本计数和准确率
        all_labels = []
        all_predictions = []
        all_probs = []

        acc_a = [0.0 for _ in range(n_classes)]  # fMRI模态的准确率


        # 遍历数据加载器中的每个 batch
        for step, data_packet in enumerate(dataloader):
            # 获取输入数据，包括 fMRI、sMRI、DTI 等
            fMRI = data_packet['fMRI']
            sMRI_deep_mnispace = data_packet['sMRI_deep_mnispace']
            DTI = data_packet['DTI']
            label = data_packet['label']

            fMRI_data = fMRI
            sMRI_data = sMRI_deep_mnispace.unsqueeze(1) / 255.0
            DTI_data = DTI.unsqueeze(1)

            # 检查输入数据是否有 NaN
            if torch.isnan(fMRI_data).any() or torch.isnan(sMRI_data).any() or torch.isnan(DTI_data).any():
                print(f"Input data contains NaN values at step {step}. Skipping this batch.")
                continue

            # 将数据移动到指定设备（例如 GPU）
            dyn_fc, sampling_points = process_dynamic_fc(fMRI_data, window_size=window_size, window_stride=window_stride, dynamic_length=dynamic_length)
            dyn_v = repeat(torch.eye(dyn_fc.shape[-1]), 'n1 n2 -> b t n1 n2', t=dyn_fc.shape[1], b=fMRI_data.shape[0])

            dyn_v = dyn_v.float().to(device)
            dyn_fc = dyn_fc.float().to(device)
            DTI_data = DTI_data.float().to(device)
            sMRI_data = sMRI_data.float().to(device)
            label = label.long().to(device)

            # 前向传播，得到各模态的输出
            a = model(dyn_v, dyn_fc)

            # 通过模型的融合模块对每个模态的输出进行分类
            out_a = model.fusion_module.fc_out(a)

            # print(out_a,out_v,out_t)
            # 计算权重并融合


            # 使用 softmax 对融合后的输出进行归一化，得到预测概率

            pred_a_probs = softmax(out_a)
 

            # 记录真实标签和预测概率
            all_labels.extend(label.cpu().numpy())
            all_probs.extend(pred_a_probs.cpu().numpy())
            all_predictions.extend(np.argmax(pred_a_probs.cpu().numpy(), axis=1))

            # 对 batch 中的每个样本进行评估
            for i in range(fMRI_data.shape[0]):
                if label[i].item() == torch.argmax(pred_a_probs[i]).item():
                    acc_a[label[i]] += 1.0


    # 将所有的标签、预测值和预测概率转换为 NumPy 数组
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_predictions = np.array(all_predictions)


    # 计算各种指标
    acc_a = accuracy_score(all_labels, all_predictions)
    auc = roc_auc_score(all_labels, all_probs[:, 1]) if n_classes == 2 else roc_auc_score(all_labels, all_probs, multi_class="ovr")
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    sen = recall_score(all_labels, all_predictions, average='weighted')  # 敏感性（召回率）
    pre = precision_score(all_labels, all_predictions, average='weighted', zero_division=1)

    # 计算特异性（需要通过混淆矩阵）
    if n_classes == 2:
        tn, fp, fn, tp = confusion_matrix(all_labels, all_predictions).ravel()
        spe = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        spe = 0  # 对于多分类问题，特异性计算需要进一步处理

    # # 计算各模态的准确率
    # acc_a = sum(acc_a) / len(all_labels)


    # 返回计算的指标
    return acc_a, auc, f1, sen, spe, pre

def valid_pro_DTI(args, model, device, dataloader, gs_flag=True):
    # 定义 softmax 激活函数，将模型输出转换为概率分布
    softmax = nn.Softmax(dim=1)

    # 设置类别数，根据数据集类型设置，这里默认是 HIVC 数据集，共有 2 个类别
    if args.dataset == "HIVC":
        n_classes = 2

    # 禁用梯度计算，用于推理或验证，以节省内存
    with torch.no_grad():
        # 设置模型为验证模式
        model.eval()

        # 初始化各类样本计数和准确率
        all_labels = []
        all_predictions = []
        all_probs = []

        # 遍历数据加载器中的每个 batch
        for step, data_packet in enumerate(dataloader):
            # 获取输入数据，包括 fMRI、sMRI、DTI 等
            fMRI = data_packet['fMRI']
            sMRI_deep_mnispace = data_packet['sMRI_deep_mnispace']
            DTI = data_packet['DTI']
            label = data_packet['label']

            fMRI_data = fMRI
            sMRI_data = sMRI_deep_mnispace.unsqueeze(1) / 255.0
            DTI_data = DTI.unsqueeze(1)

            # 检查输入数据是否有 NaN
            if torch.isnan(fMRI_data).any() or torch.isnan(sMRI_data).any() or torch.isnan(DTI_data).any():
                print(f"Input data contains NaN values at step {step}. Skipping this batch.")
                continue

            # 将数据移动到指定设备（例如 GPU）
            DTI_data = DTI_data.float().to(device)
            sMRI_data = sMRI_data.float().to(device)
            label = label.long().to(device)

            # 前向传播，得到各模态的输出
            v = model(DTI_data)

            # 通过模型的融合模块对每个模态的输出进行分类
            out_v = model.fusion_module.fc_out(v)
            
            # print(out_a,out_v,out_t)
            # 计算权重并融合

            # 使用 softmax 对融合后的输出进行归一化，得到预测概率
            pred_v_probs = softmax(out_v)



            # 记录真实标签和预测概率
            all_labels.extend(label.cpu().numpy())
            all_probs.extend(pred_v_probs.cpu().numpy())
            all_predictions.extend(np.argmax(pred_v_probs.cpu().numpy(), axis=1))


    # 将所有的标签、预测值和预测概率转换为 NumPy 数组
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_predictions = np.array(all_predictions)


    # 计算各种指标
    acc_v = accuracy_score(all_labels, all_predictions)
    auc = roc_auc_score(all_labels, all_probs[:, 1]) if n_classes == 2 else roc_auc_score(all_labels, all_probs, multi_class="ovr")
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    sen = recall_score(all_labels, all_predictions, average='weighted')  # 敏感性（召回率）
    pre = precision_score(all_labels, all_predictions, average='weighted', zero_division=1)

    # 计算特异性（需要通过混淆矩阵）
    if n_classes == 2:
        tn, fp, fn, tp = confusion_matrix(all_labels, all_predictions).ravel()
        spe = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        spe = 0  # 对于多分类问题，特异性计算需要进一步处理


    # 返回计算的指标
    return acc_v, auc, f1, sen, spe, pre

def valid_pro_sMRI(args, model, device, dataloader, gs_flag=True):
    # 定义 softmax 激活函数，将模型输出转换为概率分布
    softmax = nn.Softmax(dim=1)

    # 设置类别数，根据数据集类型设置，这里默认是 HIVC 数据集，共有 2 个类别
    if args.dataset == "HIVC":
        n_classes = 2


    # 禁用梯度计算，用于推理或验证，以节省内存
    with torch.no_grad():
        # 设置模型为验证模式
        model.eval()

        # 初始化各类样本计数和准确率
        all_labels = []
        all_predictions = []
        all_probs = []


        # 遍历数据加载器中的每个 batch
        for step, data_packet in enumerate(dataloader):
            # 获取输入数据，包括 fMRI、sMRI、DTI 等
            fMRI = data_packet['fMRI']
            sMRI_deep_mnispace = data_packet['sMRI_deep_mnispace']
            sMRI_deep_brainspace = data_packet['sMRI_deep_brainspace']
            DTI = data_packet['DTI']
            label = data_packet['label']

            fMRI_data = fMRI
            # sMRI_data = sMRI_deep_mnispace.unsqueeze(1) / 255.0
            sMRI_data = sMRI_deep_brainspace.unsqueeze(1) / 255.0
            DTI_data = DTI.unsqueeze(1)

            # 检查输入数据是否有 NaN
            if torch.isnan(fMRI_data).any() or torch.isnan(sMRI_data).any() or torch.isnan(DTI_data).any():
                print(f"Input data contains NaN values at step {step}. Skipping this batch.")
                continue

            # 将数据移动到指定设备（例如 GPU）

            DTI_data = DTI_data.float().to(device)
            sMRI_data = sMRI_data.float().to(device)
            label = label.long().to(device)

            # 前向传播，得到各模态的输出
            t = model(sMRI_data)

            # 通过模型的融合模块对每个模态的输出进行分类
            out_t = model.fusion_module.fc_out(t)

            # print(out_a,out_v,out_t)
            # 计算权重并融合
            # 使用 softmax 对融合后的输出进行归一化，得到预测概率
            pred_t_probs = softmax(out_t)



            # 记录真实标签和预测概率
            all_labels.extend(label.cpu().numpy())
            all_probs.extend(pred_t_probs.cpu().numpy())
            all_predictions.extend(np.argmax(pred_t_probs.cpu().numpy(), axis=1))


    # 将所有的标签、预测值和预测概率转换为 NumPy 数组
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_predictions = np.array(all_predictions)


    # 计算各种指标
    acc = accuracy_score(all_labels, all_predictions)
    auc = roc_auc_score(all_labels, all_probs[:, 1]) if n_classes == 2 else roc_auc_score(all_labels, all_probs, multi_class="ovr")
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    sen = recall_score(all_labels, all_predictions, average='weighted')  # 敏感性（召回率）
    pre = precision_score(all_labels, all_predictions, average='weighted', zero_division=1)

    # 计算特异性（需要通过混淆矩阵）
    if n_classes == 2:
        tn, fp, fn, tp = confusion_matrix(all_labels, all_predictions).ravel()
        spe = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        spe = 0  # 对于多分类问题，特异性计算需要进一步处理



    # 返回计算的指标
    return acc, auc, f1, sen, spe, pre
