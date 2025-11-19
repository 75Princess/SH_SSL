from torch.utils.data import DataLoader
# Import Project Modules ---------------------------------------------------------
from utils import dataset_class
from Models.model import Encoder_factory, count_parameters
from Models.loss import get_loss_module
from Models.utils import load_model
from trainer import *
# --------- For Logistic Regression--------------------------------------------------
from eval import fit_lr,  make_representation
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE  # 非线性降维算法，常用于将高维数据映射到 2D 或 3D 空间，以便于可视化

####
logger = logging.getLogger('__main__')


def Rep_Learning(config, Data):
    # 定义 Rep_Learning 函数，用于进行自监督学习和下游分类任务
    # config 是配置字典，包含训练所需的各种参数
    # Data 是包含数据集的字典，包含训练、验证和测试数据
    # ---------------------------------------- Self Supervised Data -------------------------------------
    if config['Pre_Training'] =='Cross-domain':
        pre_train_dataset = dataset_class(Data['pre_train_data'], Data['pre_train_label'], config['patch_size'])
    else:
        pre_train_dataset = dataset_class(Data['All_train_data'], Data['All_train_label'], config['patch_size'])
    pre_train_loader = DataLoader(dataset=pre_train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    train_dataset = dataset_class(Data['All_train_data'], Data['All_train_label'], config['patch_size'])
    train_loader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    # For Linear Probing During the Pre-Training
    test_dataset = dataset_class(Data['test_data'], Data['test_label'], config['patch_size'])
    test_loader = DataLoader(dataset=test_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    # --------------------------------------------------------------------------------------------------------------
    # -------------------------------------------- Build Model -----------------------------------------------------
    logger.info("Pre-Training Self Supervised model ...")
    config['Data_shape'] = Data['All_train_data'].shape
    config['num_labels'] = int(max(Data['All_train_label'])) + 1
    Encoder = Encoder_factory(config)   # EEG2Rep
    logger.info("Model:\n{}".format(Encoder))
    logger.info("Total number of parameters: {}".format(count_parameters(Encoder)))
    # ---------------------------------------------- Model Initialization ----------------------------------------------
    # Specify which networks you want to optimize
    networks_to_optimize = [Encoder.contex_encoder, Encoder.InputEmbedding, Encoder.Predictor]
    # Convert parameters to tensors
    params_to_optimize = [p for net in networks_to_optimize for p in net.parameters()]
    params_not_to_optimize = [p for p in Encoder.target_encoder.parameters()]

    optim_class = get_optimizer("RAdam")
    config['optimizer'] = optim_class([{'params': params_to_optimize, 'lr': config['lr']},
                                       {'params': params_not_to_optimize, 'lr': 0.0}])
    # 创建优化器实例，设置需要优化的参数的学习率和不需要优化的参数的学习率为 0
    config['problem_type'] = 'Self-Supervised'
    config['loss_module'] = get_loss_module()

    save_path = os.path.join(config['save_dir'], config['problem'] +'model_{}.pth'.format('last'))
    Encoder.to(config['device'])
    # 将编码器模型移动到指定的设备（如 GPU 或 CPU）上
    # ------------------------------------------------- Training The Model ---------------------------------------------
    logger.info('Self-Supervised training...')
    SS_trainer = Self_Supervised_Trainer(Encoder, pre_train_loader, train_loader, test_loader, config, l2_reg=0, print_conf_mat=False)
    # 创建自监督训练器实例，传入编码器模型、预训练数据加载器、训练数据加载器、测试数据加载器、配置信息等
    SS_train_runner(config, Encoder, SS_trainer, save_path)
    # 调用 SS_train_runner 函数开始自监督训练，并传入配置信息、编码器模型、训练器和模型保存路径
    # **************************************************************************************************************** #
    # --------------------------------------------- Downstream Task (classification)   ---------------------------------
    # ---------------------- Loading the model and freezing layers except FC layer -------------------------------------
    SS_Encoder, optimizer, start_epoch = load_model(Encoder, save_path, config['optimizer'])  # Loading the model
    SS_Encoder.to(config['device'])
    train_repr, train_labels = make_representation(SS_Encoder, train_loader)
    test_repr, test_labels = make_representation(SS_Encoder, test_loader)

    clf = fit_lr(train_repr.cpu().detach().numpy(), train_labels.cpu().detach().numpy())
    y_hat = clf.predict(test_repr.cpu().detach().numpy())
    # plot_tSNE(test_repr.cpu().detach().numpy(), test_labels.cpu().detach().numpy())
    # 注释掉的代码，用于绘制 t-SNE 可视化图
    acc_test = accuracy_score(test_labels.cpu().detach().numpy(), y_hat)
    print('Test_acc:', acc_test)
    cm = confusion_matrix(test_labels.cpu().detach().numpy(), y_hat)
    print("Confusion Matrix:")
    print(cm)
    # print("Test ROC AUC:")
    # print(roc_auc_score(y_hat, test_labels.cpu().detach().numpy()))
    # 注释掉的代码，用于计算和打印测试数据的 ROC 曲线下的面积

    # --------------------------------- Load Data -------------------------------------------------------------
    train_dataset = dataset_class(Data['train_data'], Data['train_label'], config['patch_size'])
    val_dataset = dataset_class(Data['val_data'], Data['val_label'], config['patch_size'])
    test_dataset = dataset_class(Data['test_data'], Data['test_label'], config['patch_size'])

    train_loader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)


    # 下游任务fine-tuning 有监督训练
    '''
    logger.info('Starting Fine_Tuning...')
    S_trainer = SupervisedTrainer(SS_Encoder, None, train_loader, None, config, print_conf_mat=False)
    S_val_evaluator = SupervisedTrainer(SS_Encoder, None, val_loader, None, config, print_conf_mat=False)

    save_path = os.path.join(config['save_dir'], config['problem'] + '_model_{}.pth'.format('last'))
    Strain_runner(config, SS_Encoder, S_trainer, S_val_evaluator, save_path)

    best_Encoder, optimizer, start_epoch = load_model(Encoder, save_path, config['optimizer'])
    best_Encoder.to(config['device'])

    best_test_evaluator = SupervisedTrainer(best_Encoder, None, test_loader, None, config, print_conf_mat=True)
    best_aggr_metrics_test, all_metrics = best_test_evaluator.evaluate(keep_all=True)
    return best_aggr_metrics_test, all_metrics
    '''

def Supervised(config, Data):
    # -------------------------------------------- Build Model -----------------------------------------------------
    config['Data_shape'] = Data['train_data'].shape
    config['num_labels'] = int(max(Data['train_label'])) + 1
    Encoder = Encoder_factory(config)

    logger.info("Model:\n{}".format(Encoder))
    logger.info("Total number of parameters: {}".format(count_parameters(Encoder)))
    # ---------------------------------------------- Model Initialization ----------------------------------------------
    optim_class = get_optimizer("RAdam")
    config['optimizer'] = optim_class(Encoder.parameters(), lr=config['lr'], weight_decay=0)

    config['problem_type'] = 'Supervised'
    config['loss_module'] = get_loss_module()
    # tensorboard_writer = SummaryWriter('summary')
    Encoder.to(config['device'])
    # ------------------------------------------------- Training The Model ---------------------------------------------

    # --------------------------------- Load Data -------------------------------------------------------------
    train_dataset = dataset_class(Data['train_data'], Data['train_label'], config['patch_size'])
    val_dataset = dataset_class(Data['val_data'], Data['val_label'], config['patch_size'])
    test_dataset = dataset_class(Data['test_data'], Data['test_label'], config['patch_size'])

    train_loader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)

    S_trainer = SupervisedTrainer(Encoder, None, train_loader, None, config, print_conf_mat=False)
    S_val_evaluator = SupervisedTrainer(Encoder, None, val_loader, None, config, print_conf_mat=False)

    save_path = os.path.join(config['save_dir'], config['problem'] + '_2_model_{}.pth'.format('last'))
    Strain_runner(config, Encoder, S_trainer, S_val_evaluator, save_path)
    best_Encoder, optimizer, start_epoch = load_model(Encoder, save_path, config['optimizer'])
    best_Encoder.to(config['device'])

    best_test_evaluator = SupervisedTrainer(best_Encoder, None, test_loader, None, config, print_conf_mat=True)
    best_aggr_metrics_test, all_metrics = best_test_evaluator.evaluate(keep_all=True)
    return best_aggr_metrics_test, all_metrics


def plot_tSNE(data, labels):
    # 定义 plot_tSNE 函数，用于绘制 t-SNE 可视化图

    # Create a TSNE instance with 2 components (dimensions)
    tsne = TSNE(n_components=2, random_state=42)
    # Fit and transform the data using t-SNE
    embedded_data = tsne.fit_transform(data)

    # Separate data points for each class
    class_0_data = embedded_data[labels == 0]
    class_1_data = embedded_data[labels == 1]

    # Plot with plt.plot
    plt.figure(figsize=(6, 5))  # Set background color to white
    plt.plot(class_0_data[:, 0], class_0_data[:, 1], 'bo', label='Real')
    plt.plot(class_1_data[:, 0], class_1_data[:, 1], 'ro', label='Fake')
    plt.legend(fontsize='large')
    plt.grid(False)  # Remove grid
    plt.savefig('SSL.pdf', bbox_inches='tight', format='pdf')
    # plt.show()
