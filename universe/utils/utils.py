"""
@File    :   utils.py
@Contact :   1129137758@qq.com

@Modify Time      @Author    @Version
------------      -------    --------
2022/10/31 16:48   WangLing      1.0

@Description : 深度学习，通用的一些基础函数
"""

import os
import torch
import random
import numpy as np
from torch import nn
import matplotlib.pyplot as plt

VERSION = "2.0"
EMAIL = "1129137758@qq.com"


def setRandomSeed(seed=42):
    """Reproducer for pytorch experiment.

    Parameters
    ----------
    seed: int, optional (default = 2019)
        Radnom seed.

    Example
    -------
    setRandomSeed(seed=2019).
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True


def printDash(num=50):
    print(''.join(['-'] * num))


def initConfig(cfg):
    if cfg["cfg_verbose"]:
        printDash()
        print(cfg)
        printDash()

    print("[INFO] Easy-Classification Version: " + VERSION)
    print("[INFO] Powered By : Wang Ling, Email:" + EMAIL)

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg['GPU_ID']
    setRandomSeed(cfg['random_seed'])

    if not os.path.exists(cfg['save_dir']):
        os.makedirs(cfg['save_dir'])


def npSoftmax(x):
    x_row_max = x.max(axis=-1)
    x_row_max = x_row_max.reshape(list(x.shape)[:-1] + [1])
    x = x - x_row_max
    x_exp = np.exp(x)
    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1] + [1])
    softmax = x_exp / x_exp_row_sum
    return softmax


def createFullConnectionLayer(dropout, last_channel, class_number):
    """
        构建一个全连接层，通用方法

        Args:
            dropout:为了防止过拟合，设置值，表示随机多少比例的神经元失效
            last_channel: 输入通道数
            class_number: 输出通道数，一般为实际需要分类的数值，如【0-9】数字分类，该值为10

        Returns:返回一个全连接层

        """
    head = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(last_channel, class_number),
        # nn.Softmax()
    )
    return head


def canApply(configModelName, modelName):
    if configModelName == modelName:
        return True
    return False


def pretrainedLoad(model, pretrainedPath=None):
    if pretrainedPath:
        state_dict = torch.load(pretrainedPath)
        state_dict = {k.replace('pretrain_', ''): v for k, v in state_dict.items()}
        state_dict = {k.replace('match_model.model.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
    return model


def one_hot(text):
    vector = np.zeros(4 * 62)  # (10+26+26)*4

    def char2pos(c):
        if c == '_':
            k = 62
            return k
        k = ord(c) - 48
        if k > 9:
            k = ord(c) - 55
            if k > 35:
                k = ord(c) - 61
                if k > 61:
                    raise ValueError('No Map')
        return k

    for i, c in enumerate(text):
        idx = i * 62 + char2pos(c)
        vector[idx] = 1.0
    return vector


def getFileNames(file_dir, tail_list=None):
    if tail_list is None:
        tail_list = ['.png', '.jpg', '.JPG', '.PNG']
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] in tail_list:
                L.append(os.path.join(root, file))
    return L


def modelLoad(model, model_path, data_parallel=True):
    model.load_state_dict(torch.load(model_path), strict=True)
    if data_parallel:
        model = torch.nn.DataParallel(model)
    return model


def toOnnx(model, cfg, device, save_name="model.onnx"):
    dummy_input = torch.randn(1, 3, cfg['target_img_size'][0], cfg['target_img_size'][1]).to(device)

    torch.onnx.export(model,
                      dummy_input,
                      os.path.join(cfg['save_dir'], save_name),
                      verbose=True)


def draw_loss(train_loss, test_loss, epoch, save_path):
    plt.clf()
    x = [i for i in range(epoch+1)]
    plt.plot(x, train_loss, label='train_loss')
    plt.plot(x, test_loss, label='test_loss')
    plt.legend()
    plt.title("loss goes by epoch")
    plt.xlabel('eopch')
    plt.ylabel('loss_value')
    save_path = os.path.join(save_path, 'loss.png')
    plt.savefig(save_path)


def draw_acc(train_acc, test_acc, epoch, save_path):
    plt.clf()
    x = [i for i in range(epoch+1)]
    plt.plot(x, train_acc, label='train_acc')
    plt.plot(x, test_acc, label='test_acc')
    plt.legend()
    plt.title("acc goes by epoch")
    plt.xlabel('eopch')
    plt.ylabel(save_path + 'acc_value')
    save_path = os.path.join(save_path, 'acc.png')
    plt.savefig(save_path)


def freezeBeforeLinear(model, epoch, freeze_epochs=2):
    if epoch < freeze_epochs:
        for child in list(model.children())[:-1]:
            for param in child.parameters():
                param.requires_grad = False
    elif epoch == freeze_epochs:
        for child in list(model.children())[:-1]:
            for param in child.parameters():
                param.requires_grad = True
