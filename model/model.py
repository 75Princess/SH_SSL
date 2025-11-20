import copy
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from Models.Attention import *

from .shapelet_encoder import ShapeletsDistBlocks

def Encoder_factory(config):
    model = EEG2Rep(config, num_classes=config['num_labels'])
    return model


class EEG2Rep(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        self.config = config

        # --- 1. 定义超参数 (这是新编码器需要的) ---
        # 示例：定义一个多尺度的shapelet配置
        # key是shapelet长度，value是该长度下的shapelet数量
        # repr_dim将是所有shapelet数量的总和 (30+30+30=90)
        shapelets_size_and_len = {
            int(config['Data_shape'][2] * 0.1): 30,  # 短期
            int(config['Data_shape'][2] * 0.4): 30,  # 中期
            int(config['Data_shape'][2] * 0.7): 30,  # 长期
        }
        # 将表示层维度存入config，方便后续使用
        config['repr_dim'] = sum(shapelets_size_and_len.values())

        # --- 2. 移除旧模块，实例化新模块 ---
        # 旧的 InputEmbedding 和 Encoder 不再需要

        # ❗ 用 ShapeletsDistBlocks 替换原始的 Encoder
        # 我们使用 'mix' 模式，它会自动组合欧氏距离、余弦相似度和互相关三种度量
        self.contex_encoder = ShapeletsDistBlocks(
            shapelets_size_and_len=shapelets_size_and_len,
            in_channels=config['Data_shape'][1],
            dist_measure='mix',
            to_cuda=True  # 假设使用GPU
        )
        self.target_encoder = copy.deepcopy(self.contex_encoder)

        # ❗ 用简单的MLP替换原始的交叉注意力Predictor
        self.Predictor = nn.Sequential(
            nn.Linear(config['repr_dim'], config['repr_dim'] * 2),
            nn.GELU(),  # 使用GELU激活函数
            nn.Linear(config['repr_dim'] * 2, config['repr_dim'])
        )
        # ❗ 新增第三条线的predictor2
        self.Predictor_2 = copy.deepcopy(self.Predictor)


        # --- 3. 保留的其他模块 ---
        self.momentum = config['momentum']
        self.device = config['device']
        self.mask_ratio = config['mask_ratio']
        self.predict_head = nn.Linear(config['repr_dim'], config['num_labels'])
        # 注意：这里我们不再需要gap层，因为Shapelet编码器直接输出一个(B, repr_dim)的向量

    # 主要功能是将contex_encoder的所有参数复制到target_encoder中
    def copy_weight(self):
        with torch.no_grad():
            for (param_a, param_b) in zip(self.contex_encoder.parameters(), self.target_encoder.parameters()):
                param_b.data = param_a.data

    def momentum_update(self):
        with torch.no_grad():
            for (param_a, param_b) in zip(self.contex_encoder.parameters(), self.target_encoder.parameters()):
                param_b.data = self.momentum * param_b.data + (1 - self.momentum) * param_a.data

    def linear_prob(self, x):
        with (torch.no_grad()):
            patches = self.InputEmbedding(x)
            patches = self.Norm(patches)
            patches = patches + self.PositionalEncoding(patches)
            patches = self.Norm2(patches)
            out = self.contex_encoder(patches)
            out = out.transpose(2, 1)
            out = self.gap(out)
            return out.squeeze()

    def pretrain_forward(self, x):
        # 1. 生成学习目标 (教师路径) 第一条线
        with torch.no_grad():
            y_target = self.target_encoder(x).squeeze(1)  # Shapelet编码器直接处理原始x

        # 2. 生成掩码输入 (学生路径) 第二条线
        # 原始的SSP是作用于patch索引，现在需要一个直接在时间序列张量上操作的掩码函数
        masked_x = mask_time_series_ssp(x, self.mask_ratio)  # ssp语义子序列保留调用
        r_context = self.contex_encoder(masked_x).squeeze(1)
        y_pred = self.Predictor(r_context)

        # 3.新增学生路径 第三条线
        # 对r_context再次掩码
        r_masked = random_feature_masking(r_context, 0.5)
        y_pred_2 = self.Predictor_2(r_masked)


        # 4. 返回用于计算损失的张量 (还可加入VICReg损失所需的y_target)
        return y_target, y_pred, y_pred_2

    def forward(self, x, mode='Supervised'):
        if mode == 'Rep-Learning':
            # 在评估时，我们用contex_encoder来提取特征
            features = self.contex_encoder(x).squeeze(1)
            return features
        else:  # Supervised mode
            features = self.contex_encoder(x).squeeze(1)
            logits = self.predict_head(features)
            return logits

# 语义子序列保留函数，用于选择可见和掩码索引
def mask_time_series_ssp(x, mask_ratio, chunk_count=4):
    """
    在时间序列张量上实现语义子序列保留（SSP）掩码。
    这个函数会保留 chunk_count 个连续的块，并将其余部分置零。

    参数:
    x (torch.Tensor): 输入张量，形状为 [B, C, L].
    mask_ratio (float): 需要被遮盖掉的比例，例如 0.5.
    chunk_count (int): 要保留的连续块的数量。

    返回:
    torch.Tensor: 被掩码后的张量，形状与输入x相同。
    """
    batch_size, num_channels, seq_len = x.shape

    # 1. 计算需要保留的总长度和每个块的长度
    # (1 - mask_ratio) 是保留的比例
    visible_len = int(seq_len * (1 - mask_ratio))
    chunk_len = visible_len // chunk_count

    if chunk_len == 0:
        raise ValueError("mask_ratio is too high or chunk_count is too large, resulting in zero-length chunks.")

    # 2. 为批次中的每个样本生成一个掩码
    # 创建一个全零的掩码张量，形状为 [B, 1, L]
    # 我们在通道维度上用1，这样就可以利用广播机制将其应用到所有通道
    mask = torch.zeros(batch_size, 1, seq_len, device=x.device)

    for i in range(batch_size):
        # 3. 随机选择每个块的起始点
        start_points = []
        for _ in range(chunk_count):
            # 确保起始点能容纳一个完整的块
            start = random.randint(0, seq_len - chunk_len)
            start_points.append(start)

        # 4. 在掩码中将需要保留的区域置为1
        for start in start_points:
            end = start + chunk_len
            mask[i, :, start:end] = 1

    # 5. 将掩码应用到输入张量上
    # 利用PyTorch的广播机制，[B, 1, L]的掩码会自动扩展以匹配 [B, C, L] 的x
    masked_x = x * mask

    return masked_x


def random_feature_masking(x, mask_ratio=0.5):
    """
    对一个批次的特征向量进行随机掩码。

    参数:
    x (torch.Tensor): 输入的特征向量，形状为 [B, D_repr].
    mask_ratio (float): 需要被遮盖掉的特征维度的比例。

    返回:
    torch.Tensor: 被随机置零一部分维度后的特征向量。
    """
    if not 0.0 < mask_ratio < 1.0:
        raise ValueError("mask_ratio must be between 0 and 1")

    # 获取输入形状
    batch_size, num_features = x.shape

    # 计算需要被遮盖的特征数量
    num_to_mask = int(num_features * mask_ratio)

    # 创建一个与x形状相同的、用于存储掩码后结果的张量
    x_masked = x.clone()

    # 对批次中的每一个样本独立进行掩码
    for i in range(batch_size):
        # 随机选择要遮盖的维度的索引
        mask_indices = torch.randperm(num_features)[:num_to_mask]

        # 将这些维度的值置为0
        x_masked[i, mask_indices] = 0

    return x_masked

class Predictor(nn.Module):
    def __init__(self, d_model, attn_heads, d_ffn, enable_res_parameter, layers):
        super(Predictor, self).__init__()
        self.layers = nn.ModuleList(
            [CrossAttnTRMBlock(d_model, attn_heads, d_ffn, enable_res_parameter) for i in range(layers)])

    def forward(self, rep_visible, rep_mask_token):
        for TRM in self.layers:
            rep_mask_token = TRM(rep_visible, rep_mask_token)
        return rep_mask_token


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


