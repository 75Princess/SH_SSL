import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F


def get_loss_module():
    # 返回一个NoFussCrossEntropyLoss类的实例，设置reduction参数为'none'，表示返回每个批量样本的损失
        return NoFussCrossEntropyLoss(reduction='none')  # outputs loss for each batch sample


def l2_reg_loss(model):
    """Returns the squared L2 norm of output layer of given model"""

    for name, param in model.named_parameters():
        if name == 'output_layer.weight':
            return torch.sum(torch.square(param))


class NoFussCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    pytorch's CrossEntropyLoss is fussy: 1) needs Long (int64) targets only, and 2) only 1D.
    This function satisfies these requirements
    """

    def forward(self, inp, target):
        return F.cross_entropy(inp, target.long(), weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)


