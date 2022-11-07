"""
@File    :   optimizer_adapter.py
@Contact :   1129137758@qq.com

@Modify Time      @Author    @Version
------------      -------    --------
2022/9/27 16:48   WangLing      1.0

@Description : 优化器策略方案
"""

from torch import optim

from universe.runner_config.optimizer_config.ranger import Ranger
from timm.optim import AdaBelief


class OptimizerAdapter:

    @staticmethod
    def getOptimizer(optims, model, learning_rate, weight_decay):
        if optims == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optims == 'AdamW':
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optims == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
        elif optims == 'AdaBelief':
            optimizer = AdaBelief(model.parameters(), lr=learning_rate, eps=1e-12, betas=(0.9, 0.999))
        elif optims == 'Ranger':
            optimizer = Ranger(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            raise Exception("Unkown getSchedu: ", optims)
        return optimizer


def clipGradient(optimizer, grad_clip=1):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)
