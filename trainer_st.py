# 作者:周子涵
# 2025年06月30日21时46分08秒
# 文件名: trainer_st.py

import torch
import torch.nn.functional as F
import logging
from tqdm import tqdm
import copy
import os
import numpy as np
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression

# ❗ 实现VICReg损失函数

def vicreg_loss_fn(x, config):
    """
    计算VICReg损失的方差和协方差部分。
    这个损失函数作用于一个批次的表示向量上。

    参数:
    x (torch.Tensor): 一个批次的表示向量，形状为 [B, D] (批次大小, 表示层维度)
    config (dict): 包含超参数的配置字典

    返回:
    torch.Tensor: 计算出的VICReg损失值
    """
    # 从配置中获取VICReg的超参数权重
    sim_coeff = config.get('vic_sim_coeff', 25.0)  # 不变性/重建损失的权重 (我们不在这里用)
    std_coeff = config.get('vic_std_coeff', 25.0)  # 方差损失的权重
    cov_coeff = config.get('vic_cov_coeff', 1.0)  # 协方差损失的权重

    # --- 1. 方差损失 (Variance Loss) ---
    # 计算批次中每个特征维度的标准差
    std_x = torch.sqrt(x.var(dim=0) + 1e-4)  # 加上一个小的epsilon防止出现0
    # 目标是让标准差尽可能接近1。当标准差小于1时，产生损失。
    variance_loss = torch.mean(F.relu(1 - std_x))

    # --- 2. 协方差损失 (Covariance Loss) ---
    # 将表示进行中心化
    x = x - x.mean(dim=0)
    # 计算协方差矩阵
    cov_x = (x.T @ x) / (len(x) - 1)
    # 我们希望协方差矩阵的非对角线元素都趋近于0
    # 这等价于让 (C*C - I) 的非对角线元素趋近于0
    # torch.eye(D) 会创建一个D*D的单位矩阵
    covariance_loss = (cov_x.pow(2) - torch.eye(x.shape[1], device=x.device)).sum() / x.shape[1]

    # --- 3. 组合损失 ---
    # 在我们的框架中，重建损失是单独计算的，这里只返回正则化部分
    weighted_variance_loss = std_coeff * variance_loss
    weighted_covariance_loss = cov_coeff * covariance_loss

    return weighted_variance_loss + weighted_covariance_loss

logger = logging.getLogger('__main__')


# ❗ 我们将特征提取的辅助函数也放在这个文件里，因为它只在训练器内部被调用
def make_representation_sglp(model, data_loader, config):
    model.eval()  # 确保模型处于评估模式
    all_features = []
    with torch.no_grad():
        for (batch_x,) in data_loader:
            batch_x = batch_x.to(config['device'])
            features = model(batch_x, mode='Rep-Learning')  # 调用我们设计的“特征提取”模式
            all_features.append(features.cpu().numpy())
    return np.concatenate(all_features, axis=0)


# ❗ 主训练函数现在接收Data对象，以便访问验证集
def st_train_runner(config, model, optimizer, train_loader, Data):
    """
    我们新模型SGLP的核心训练与评估循环
    """

    #调度器替换成模拟退火
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    # ❗ MODIFIED: 将学习率调度器从 CosineAnnealingLR 换成 OneCycleLR
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['lr'],
        epochs=config['epochs'],
        steps_per_epoch=len(train_loader)
    )
    reconstruction_loss_fn = F.mse_loss

    # ❗ 初始化用于追踪最佳模型的变量
    best_val_acc = 0.0
    best_model_weights = copy.deepcopy(model.state_dict())

    # 准备用于在线评估的数据加载器
    # 我们使用验证集进行周期性评估，这是一个非常重要的最佳实践，可以避免模型在训练时“偷看”到测试集
    val_loader = DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(Data['val_data']).float()),
                            batch_size=config['batch_size'])
    # 标准的线性探测需要用全部训练数据来训练临时分类器
    train_eval_loader = DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(Data['train_data']).float()),
                                   batch_size=config['batch_size'])

    for epoch in range(config['epochs']):
        model.train()
        epoch_loss = 0

        # 核心修改在这里：在tqdm中加入了enumerate 
        # 这样每次循环，都能同时获得索引i和数据batch_x
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader),
                            desc=f"Epoch {epoch + 1}/{config['epochs']}")

        for i, (batch_x,) in progress_bar:
            batch_x = batch_x.to(config['device'])

            # A. 前向传播
            y_target, y_pred, y_pred_2 = model.pretrain_forward(batch_x)

            # 在计算损失前，对目标和预测进行L2归一化   应对初始loss爆炸问题
            y_target_norm = F.normalize(y_target, p=2, dim=-1)
            y_pred_norm = F.normalize(y_pred, p=2, dim=-1)
            y_pred_2_norm = F.normalize(y_pred_2, p=2, dim=-1)

            # B. 计算损失 (假设vicreg_loss_fn在别处定义或导入)
            # 两个重建损失
            recon_loss = reconstruction_loss_fn(y_pred_norm, y_target_norm)
            recon_loss_2 = reconstruction_loss_fn(y_pred_2_norm, y_target_norm)

            div_loss = vicreg_loss_fn(y_target_norm, config)
            # 组合总损失
            loss = recon_loss + 0.5*recon_loss_2 + config.get('lambda_vic', 1.0) * div_loss

            # C. 反向传播与优化
            optimizer.zero_grad()
            loss.backward()

            # ❗❗ 新增：梯度裁剪，将所有参数的梯度总范数限制在 1.0 ❗❗
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
            optimizer.step()

            # ❗❗ 重要：OneCycleLR需要在每个step后都更新
            lr_scheduler.step()

            # D. EMA动量更新教师模型
            model.momentum_update()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_epoch_loss = epoch_loss / len(train_loader)

        # --- ❗ 新增的日志记录部分 ---
        # 1. 将每轮的训练损失写入文件
        train_log_string = f"Epoch {epoch + 1}: Average Training Loss = {avg_epoch_loss:.4f}\n"
        with open(config['log_file_path'], 'a') as f:
            f.write(train_log_string)
        logger.info(f"Epoch {epoch + 1} finished. Average Loss: {avg_epoch_loss:.4f}")

        # 2. 每5个epoch或最后一个epoch，进行在线评估并记录
        if (epoch + 1) % 5 == 0 or (epoch + 1) == config['epochs']:
            # 提取特征
            train_features = make_representation_sglp(model, train_eval_loader, config)
            val_features = make_representation_sglp(model, val_loader, config)

            # 训练临时分类器并评估
            classifier = LogisticRegression(random_state=config['seed'], max_iter=2000)
            classifier.fit(train_features, Data['train_label'])
            val_acc = classifier.score(val_features, Data['val_label'])

            # 将评估结果写入文件
            eval_log_string = f"Epoch {epoch + 1}: Validation Accuracy = {val_acc:.4f}\n"
            with open(config['log_file_path'], 'a') as f:
                f.write(eval_log_string)
            logger.info(f"--- Epoch {epoch + 1} Evaluation --- Validation Acc: {val_acc:.4f}")

            # 检查并保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_weights = copy.deepcopy(model.state_dict())
                logger.info(f"  -> New best model found. Saving checkpoint...")
                torch.save(best_model_weights, os.path.join(config['output_dir'], 'best_model.pth'))

        # ❗ REMOVED: 因为OneCycleLR在step级别更新，所以epoch级别的更新需要删除
        # lr_scheduler.step()


    print("预训练完成！")
    # 训练结束后，返回最佳模型的权重
    return best_model_weights