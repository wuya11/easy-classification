"""
@File    :  data_load_service.py
@Contact :   1129137758@qq.com

@Modify Time      @Author    @Version
------------      -------    --------
2022/9/27 16:48   WangLing      1.0

@Description : create a data_load service
"""

import os
import random

import cv2
import numpy as np
import torch
from sklearn.model_selection import KFold

from universe.utils.utils import getFileNames


def getValDataLoader(source_data, cfg, data_set):
    eval_loader = torch.utils.data.DataLoader(
        data_set(source_data, cfg),
        batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'], pin_memory=True)
    return eval_loader


def getPredictDataLoader(source_data, cfg, data_set):
    predict_loader = torch.utils.data.DataLoader(
        data_set(source_data, cfg),
        batch_size=cfg['test_batch_size'], shuffle=True, num_workers=cfg['num_workers'], pin_memory=True)
    return predict_loader


def getTrainDataLoader(source_data, cfg, data_set):
    train_loader = torch.utils.data.DataLoader(
        data_set(source_data, cfg),
        batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'], pin_memory=True)
    return train_loader


class DataLoadService:
    def __init__(self, cfg):

        self.cfg = cfg

    def getTrainValDataloader(self, data_set):

        """
        加载训练集数据和验证集数据，训练集路径必须存在

        当验证集路径存在时：验证集获取验证路径下的数据

        当验证集数据不存在时：采用K折交叉验证模式生成 训练集和验证集数据
        :return: 返回数据信息
        """
        if self.cfg['train_path'] == '':
            raise Exception("[ERROR] train_path path must not be none")

        if self.cfg['val_path'] != '':
            print("[INFO] val_path is not none, not use kflod to split train-val data ...")
            train_data = getFileNames(self.cfg['train_path'])
            train_data.sort(key=lambda x: os.path.basename(x))
            train_data = np.array(train_data)
            random.shuffle(train_data)

            val_data = getFileNames(self.cfg['val_path'])
            if self.cfg['try_to_train_items'] > 0:
                train_data = train_data[:self.cfg['try_to_train_items']]
        else:
            print("[INFO] val_path is none, use kflod to split data: k=%d start_fold=%d" % (
                self.cfg['k_flod'], self.cfg['start_fold']))
            data_names = getFileNames(self.cfg['train_path'])
            print("[INFO] Total images: ", len(data_names))

            data_names.sort(key=lambda x: os.path.basename(x))
            data_names = np.array(data_names)
            random.shuffle(data_names)

            if self.cfg['try_to_train_items'] > 0:
                data_names = data_names[:self.cfg['try_to_train_items']]

            folds = KFold(n_splits=self.cfg['k_flod'], shuffle=False)
            data_iter = folds.split(data_names)
            for fid in range(self.cfg['k_flod']):
                train_index, val_index = next(data_iter)
                if fid == self.cfg['start_fold']:
                    break

            train_data = data_names[train_index]
            val_data = data_names[val_index]

        train_loader = getTrainDataLoader(train_data, self.cfg, data_set)
        val_loader = getValDataLoader(val_data, self.cfg, data_set)

        return train_loader, val_loader

    def getTrainDataloader(self, data_set):
        if self.cfg['train_path'] == '':
            raise Exception("[ERROR] train_path path must not be none")
        train_data = getFileNames(self.cfg['train_path'])
        train_data.sort(key=lambda x: os.path.basename(x))
        train_data = np.array(train_data)
        random.shuffle(train_data)
        if self.cfg['try_to_train_items'] > 0:
            train_data = train_data[:self.cfg['try_to_train_items']]
        return getTrainDataLoader(train_data, self.cfg, data_set)

    def getValDataloader(self, data_set):
        if self.cfg['val_path'] == '':
            raise Exception("[ERROR] val_path path must not be none")
        val_data = getFileNames(self.cfg['val_path'])
        val_data.sort(key=lambda x: os.path.basename(x))
        val_data = np.array(val_data)
        random.shuffle(val_data)
        return getValDataLoader(val_data, self.cfg, data_set)

    def getPredictDataloader(self, data_set):
        data_names = getFileNames(self.cfg['test_path'])
        return getPredictDataLoader(
            data_names,
            self.cfg, data_set)
