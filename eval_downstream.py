# 作者:周子涵
# 2025年07月07日13时56分21秒

# 文件名: eval_downstream.py

import torch
import argparse
import logging
import numpy as np
import os

from utils import Data_Loader, Initialization
from Models.model import EEG2Rep  # 导入我们的模型架构
from evaluation import evaluate_classification  # 导入我们的评估函数

logger = logging.getLogger('__main__')


def main(config):
    # --- 1. 初始化和加载数据 ---
    config['device'] = Initialization(config)
    Data = Data_Loader(config)

    # --- 2. 初始化模型架构，并加载预训练权重 ---
    logger.info(f"Loading pre-trained model from: {config['model_checkpoint_path']}")

    # a. 根据数据集信息，补全config
    config['num_labels'] = int(np.max(Data['All_train_label'])) + 1
    config['Data_shape'] = Data['All_train_data'].shape

    # b. 初始化一个同样架构的“空”模型
    model = EEG2Rep(config, num_classes=config['num_labels']).to(config['device'])

    # c. 加载我们训练好的权重
    try:
        model.load_state_dict(torch.load(config['model_checkpoint_path'], map_location=config['device']))
    except Exception as e:
        logger.error(f"Failed to load model weights: {e}")
        return

    # --- 3. 调用评估函数 ---
    evaluate_classification(model, Data, config, config['model_checkpoint_path'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    '''
    Tips:每次调用现有模型参数进行下游任务时，需要修改以下三个参数：
    default_model_path
    data_dir
    dataset_name
    '''

    # ❗ ADDED: 为评估脚本定义专属的命令行参数
    default_model_path = r"C:\Users\87391\Desktop\paper\ShapeletTransformer\Results\Rep-Learning\Dataset\UCIHAR\2025-07-07_14-19\best_model.pth"
    parser.add_argument('--model_checkpoint_path', type=str, default=default_model_path,
                        help='Path to the saved model checkpoint (.pth file)')
    parser.add_argument('--data_dir', default='Dataset/UCIHAR', help='Data directory')
    parser.add_argument('--dataset_name', default='UCIHAR', help='Name of the .npy data file')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for evaluation')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index, -1 for CPU')
    # ❗ 需要从main_st.py复制一些构建模型所必需的参数，如num_scales等
    parser.add_argument('--num_scales', type=int, default=3, help='Number of different scales for shapelets')
    parser.add_argument('--num_shapelets_per_scale', type=int, default=100, help='Number of shapelets per scale')
    parser.add_argument('--seed', default=1234, type=int, help='Seed used for splitting sets')

    # 需要添加的参数
    parser.add_argument('--Pre_Training', default='In-domain', choices={'In-domain', 'Cross-domain'})
    parser.add_argument('--momentum', type=float, default=0.99, help="Beta coefficient for EMA update")
    parser.add_argument('--mask_ratio', type=float, default=0.5, help="Masking ratio for SSP")
    parser.add_argument('--lambda_vic', type=float, default=1.0, help="Weight for VICReg loss")  # 示意值



    args = parser.parse_args()
    config = vars(args)  # 将命令行参数直接转为字典
    if 'problem' not in config:
        config['problem'] = config.get('dataset_name', 'default_problem')

    main(config)  # 将解析的参数转为字典传入main函数
