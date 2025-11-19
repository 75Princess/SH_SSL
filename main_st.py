# 作者:周子涵
# 2025年06月30日20时58分25秒
import os
import numpy as np
import pandas as pd
import argparse
import logging
import copy  #  ADDED: 为了深度复制模型，需要导入copy库

#  MODIFIED: 核心的训练/评估逻辑放在 running_st.py 文件中
from running_st import Rep_Learning, Supervised

# Import Project Modules -----------------------------------------------------------------------------------------------
from utils import Setup, Initialization, Data_Loader, print_title

logger = logging.getLogger('__main__')
parser = argparse.ArgumentParser()
# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------ System --------------------------------------------------------
parser.add_argument('--gpu', type=int, default=0, help='GPU index, -1 for CPU')  # ❗ MODIFIED: default to 0 for GPU
parser.add_argument('--console', action='store_true', help="Optimize printout for console output; otherwise for file")
parser.add_argument('--seed', default=1234, type=int, help='Seed used for splitting sets')
# --------------------------------------------------- I/O --------------------------------------------------------------
#  将data_dir的默认值和选项填入处理好的HAR数据集
parser.add_argument('--data_dir', default='Dataset/mhealth', help='Data directory',
                    choices={'Dataset/mhealth', 'Dataset/UCIHAR', 'Dataset/pamap2', 'Dataset/opportunity'})
parser.add_argument('--dataset_name', default='mhealth', help='Name of the .npy data file (without extension)',
                    choices={'UCIHAR', 'mhealth', 'pamap2', 'opportunity'})
parser.add_argument('--output_dir', default='Results',
                    help='Root output directory. Time-stamped directories will be created inside.')

parser.add_argument('--print_interval', type=int, default=10, help='Print batch info every this many batches')
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------- Parameters and Hyperparameter ----------------------------------------------
parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=128, help='Training batch size')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')  # ❗ MODIFIED: 较小的学习率通常更稳定
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout regularization ratio')
parser.add_argument('--Norm', type=bool, default=False, help='Data Normalization')
parser.add_argument('--val_ratio', type=float, default=0.2, help="Proportion of the train-set to be used as validation")
parser.add_argument('--val_interval', type=int, default=2, help='Evaluate on validation every XX epochs. Must be >= 1')
parser.add_argument('--key_metric', choices={'loss', 'accuracy'}, default='loss', help='Metric used for best epoch')

# -------------------------------------------------- SGLP-HAR ---------------------------------------------
# ❗ ADDED: 为我们的Shapelet模型添加专属的超参数
parser.add_argument('--Training_mode', default='Rep-Learning', choices={'Rep-Learning', 'Supervised'})
parser.add_argument('--Pre_Training', default='In-domain', choices={'In-domain', 'Cross-domain'})
parser.add_argument('--dist_measure', default='mix', choices={'mix', 'euclidean', 'cosine', 'cross-correlation'},
                    help="Distance measure for shapelets")
parser.add_argument('--num_scales', type=int, default=3, help='Number of different scales (lengths) for shapelets')
parser.add_argument('--num_shapelets_per_scale', type=int, default=100, help='Number of shapelets per scale')

# --- 移除或忽略了原始EEG-JEPA和MAE的特定参数 ---

# -------------------------------------------------- Pre-Training Hyperparameters --------------------------------------
# ❗ ADDED: 添加一些预训练需要的参数
parser.add_argument('--momentum', type=float, default=0.99, help="Beta coefficient for EMA update")
parser.add_argument('--mask_ratio', type=float, default=0.5, help="Masking ratio for SSP")
parser.add_argument('--lambda_vic', type=float, default=1.0, help="Weight for VICReg loss")  # 示意值

# ----------------------------------------------------------------------------------------------------------------------
args = parser.parse_args()
All_Results = ['Datasets', 'FC_layer']

if __name__ == '__main__':
    config = Setup(args)  # configuration dictionary
    config['device'] = Initialization(config)

    # 定义日志文件路径并存入config
    log_file_path = os.path.join(config['output_dir'], 'training_log.txt')
    config['log_file_path'] = log_file_path

    # ❗ MODIFIED: 移除了对config['problem']的依赖，因为它未定义
    print_title(f"Task: {config['dataset_name']} - Mode: {config['Training_mode']}")
    # problem参数是影响data的load方式，这里不定义problem会使用Data_Loader的默认方式
    logger.info("Loading Data ...")
    Data = Data_Loader(config)
    # ❗ ADDED: 动态生成Shapelet配置并更新config
    # 这是将命令行参数转换为Shapelet编码器所需格式的关键步骤
    # 从Data对象中获取序列长度L和通道数C
    L = Data['max_len']
    C = Data['train_data'].shape[1]
    config['Data_shape'] = [None, C, L]  # 存储数据形状信息

    # 根据命令行参数动态计算shapelets_size_and_len字典
    shapelet_lengths = np.linspace(int(L * 0.1), int(L * 0.8), num=config['num_scales'], dtype=int)
    shapelets_size_and_len = {length: config['num_shapelets_per_scale'] for length in shapelet_lengths}

    config['shapelets_size_and_len'] = shapelets_size_and_len
    # 计算并存储总的表示层维度
    if config['dist_measure'] == 'mix':
        # 在mix模式下，每个尺度会创建3种类型的shapelet
        num_total_shapelets = config['num_scales'] * config['num_shapelets_per_scale']
    else:
        num_total_shapelets = config['num_scales'] * config['num_shapelets_per_scale']

    config['repr_dim'] = num_total_shapelets

    # ---------------------------------------------------------------

    if config['Training_mode'] == 'Rep-Learning':
        # 调用预训练函数
        best_aggr_metrics_test, all_metrics = Rep_Learning(config, Data)
    elif config['Training_mode'] == 'Supervised':
        # 调用监督学习/微调函数
        best_aggr_metrics_test, all_metrics = Supervised(config, Data)

    # 修改了结果打印和保存部分，使其更通用
    print_str = 'Best Model Test Summary: '
    if best_aggr_metrics_test:  # 确保有结果再打印
        for k, v in best_aggr_metrics_test.items():
            print_str += f'{k}: {v:.4f} | '
        print(print_str)
        # 这里可以根据需要决定如何保存最终结果
        # dic_position_results = [config['dataset_name'], all_metrics.get('total_accuracy', 0)]
        # All_Results = np.vstack((All_Results, dic_position_results))

# All_Results_df = pd.DataFrame(All_Results)
# All_Results_df.to_csv(os.path.join(config['output_dir'], config['Training_mode'] + '.csv'))