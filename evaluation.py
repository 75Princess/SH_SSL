# 作者:周子涵
# 2025年07月07日13时05分13秒

# 文件名: evaluation.py

import numpy as np
import torch
from torch.utils.data import DataLoader
import logging
import os

#    借鉴CSL论文，使用SVM作为下游任务的强大分类器
#    同时，为了计算所有指标，需要从sklearn.metrics导入必要的函数
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger('__main__')


def make_representation_sglp(model, data_loader, config):
    """
    辅助函数：使用训练好的模型提取一个数据集的全部特征表示。
    这个函数从trainer文件中移动到这里，以保持评估逻辑的内聚性。
    """
    model.eval()  # 确保模型处于评估模式
    all_features = []
    with torch.no_grad():
        for (batch_x,) in data_loader:
            batch_x = batch_x.to(config['device'])
            # 调用模型的“特征提取”模式
            features = model(batch_x, mode='Rep-Learning')
            all_features.append(features.cpu().numpy())
    return np.concatenate(all_features, axis=0)


def evaluate_classification(model, Data, config, checkpoint_path):
    """
    下游分类任务的完整评估函数。
    它会提取特征、训练分类器、进行预测并计算所有关键指标。
    """
    logger.info("--- Starting Downstream Classification Evaluation ---")

    # 1. 准备数据加载器
    # 我们使用全部训练数据来训练最终的分类器，以获得最可靠的性能
    train_loader = DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(Data['All_train_data']).float()),
                              batch_size=config['batch_size'],
                              shuffle=False)
    test_loader = DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(Data['test_data']).float()),
                             batch_size=config['batch_size'],
                             shuffle=False)

    # 2. 提取特征
    logger.info("Extracting features from the training set...")
    train_features = make_representation_sglp(model, train_loader, config)
    train_labels = Data['All_train_label']

    logger.info("Extracting features from the test set...")
    test_features = make_representation_sglp(model, test_loader, config)
    test_labels = Data['test_label']

    # 2.5 特征标准化
    logger.info("Standardizing features...")
    scaler = StandardScaler()
    # 关键：只在训练集上进行 .fit() 来学习缩放规则
    train_features_scaled = scaler.fit_transform(train_features)
    # 然后用同一个scaler来转换测试集
    test_features_scaled = scaler.transform(test_features)

    # 2.75 ❗ MODIFIED: 将标签转换为从0开始
    logger.info("Converting labels from 1-6 range to 0-5 range for sklearn compatibility...")
    train_labels_zero_indexed = train_labels - 1
    test_labels_zero_indexed = test_labels - 1


    # 3. 训练下游分类器 (我们选择SVM)
    logger.info("Training SVM classifier...")
    # 使用SVC，并设置probability=True以便后续计算AUROC
    classifier = SVC(probability=True, random_state=config['seed'])
    classifier.fit(train_features_scaled, train_labels_zero_indexed)

    # 4. 在测试集上进行预测
    logger.info("Making predictions on the test set...")
    test_predictions = classifier.predict(test_features_scaled)
    test_probabilities = classifier.predict_proba(test_features_scaled)

    # 5. 计算所有评价指标
    logger.info("Calculating evaluation metrics...")

    acc = accuracy_score(test_labels_zero_indexed, test_predictions)
    b_acc = balanced_accuracy_score(test_labels_zero_indexed, test_predictions)
    w_f1 = f1_score(test_labels_zero_indexed, test_predictions, average='weighted')

    num_classes = len(np.unique(train_labels_zero_indexed))
    if num_classes > 2:
        auroc = roc_auc_score(test_labels, test_probabilities, multi_class='ovr', average='weighted')
    else:
        auroc = roc_auc_score(test_labels, test_probabilities[:, 1])

    metrics = {
        'Accuracy': acc,
        'Balanced_Accuracy': b_acc,
        'Weighted_F1': w_f1,
        'AUROC': auroc
    }

    # ❗❗ 新增代码：计算并打印混淆矩阵 ❗❗
    logger.info("Calculating Confusion Matrix...")
    cm = confusion_matrix(test_labels, test_predictions)
    print("--- Confusion Matrix ---")
    print(cm)
    print("------------------------")

    logger.info("--- Evaluation Finished ---")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value:.4f}")

    # 日志文件将与被评估的模型保存在同一个目录下
    output_dir = os.path.dirname(checkpoint_path)
    eval_log_path = os.path.join(output_dir, 'evaluation_results.txt')

    logger.info(f"Writing evaluation metrics to: {eval_log_path}")
    with open(eval_log_path, 'a') as f:  # 使用 'w' (write) 模式创建新文件
        f.write(f"Evaluation results for model: {checkpoint_path}\n")
        f.write(f"Dataset: {config['dataset_name']}\n")
        f.write("-" * 50 + "\n")
        for key, value in metrics.items():
            f.write(f"  {key}: {value:.4f}\n")

    return metrics
