# 作者:周子涵
# 2025年06月30日21时00分22秒
# 文件名: running_st.py

import logging
import os
import torch
from torch.utils.data import DataLoader
import copy

# 导入修改后的模型类和新的训练器
from Models.model import EEG2Rep  # 确保这是我们修改过的、使用Shapelet的版本
from trainer_st import st_train_runner  # 我们将在下一步创建这个训练器

# 导入评估所需的模块
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
from evaluation import evaluate_classification
from evaluation import make_representation_sglp

logger = logging.getLogger('__main__')


def Rep_Learning(config, Data):
    """
    我们新模型SGLP的表征学习（预训练）主函数 (修正版)
    """
    # --- 1. 数据准备 ---
    pre_train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(Data['All_train_data']).float())
    pre_train_loader = DataLoader(dataset=pre_train_dataset, batch_size=config['batch_size'], shuffle=True,
                                  pin_memory=True)

    # --- 2. 模型和优化器初始化 ---
    logger.info("Initializing SGLP Model for Pre-Training...")
    config['num_labels'] = int(np.max(Data['All_train_label'])) + 1
    config['Data_shape'] = Data['All_train_data'].shape
    model = EEG2Rep(config, num_classes=config['num_labels']).to(config['device'])

    params_to_optimize = list(model.contex_encoder.parameters()) + list(model.Predictor.parameters())
    optimizer = torch.optim.AdamW(params_to_optimize, lr=config['lr'])

    # --- 3. 调用训练器开始训练 ---
    logger.info('Starting SGLP Self-Supervised Pre-Training...')

    # ❗ MODIFIED: 将Data对象传递给训练器，并接收返回的最佳模型权重
    best_model_weights = st_train_runner(config, model, optimizer, pre_train_loader, Data)

    # --- 4. 下游任务评估 (Linear Probing) ---
    final_metrics = {}
    if best_model_weights:
        # 加载在验证集上表现最好的模型权重
        logger.info("Loading best model for final evaluation on test set...")
        model.load_state_dict(best_model_weights)

        # 定义最佳模型的保存路径，并将其传给评估函数用于记录
        checkpoint_path = os.path.join(config['output_dir'], 'best_model.pth')

        # 调用独立的评估函数
        final_metrics = evaluate_classification(model, Data, config, checkpoint_path)
    else:
        logger.warning("No best model found. Skipping final evaluation.")

    # 返回最终的测试集评估指标
    return final_metrics, final_metrics





# def make_representation_sglp(model, data_loader, labels, config):
'''
def make_representation_sglp(model, data_loader,  config):
    """
    用于提取SGLP模型特征的辅助函数
    """
    model.eval()
    all_features = []

    with torch.no_grad():
        for (batch_x,) in data_loader:
            batch_x = batch_x.to(config['device'])

            # ❗❗ 核心修改在这里
            # 我们调用统一的forward方法，并明确传入mode='Rep-Learning'
            features = model(batch_x, mode='Rep-Learning')

            all_features.append(features.cpu().numpy())

    return np.concatenate(all_features, axis=0)
'''


def Supervised(config, Data):
    # (这部分暂时留空，我们先专注于预训练阶段的实现)
    logger.info("Supervised fine-tuning is not implemented in this version yet.")
    return {}, {}