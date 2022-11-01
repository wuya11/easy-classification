# -*- encoding: utf-8 -*-
"""
@File    :   densenet.py    
@Contact :   1129137758@qq.com

@Modify Time      @Author    @Version    
------------      -------    --------   
2022/10/18 17:45   WangLing      1.0        

@Description : densenet网络
"""
import torch
import torchvision
from torch import nn

from universe.utils.utils import pretrainedLoad, createFullConnectionLayer


class Densenet(nn.Module):

    def __init__(self, cfg):
        super(Densenet, self).__init__()
        self.cfg = cfg
        self.createModel()
        self.changeModelStructure()

    def forward(self, x):
        x = self.backbone(x)
        x = x.mean(3).mean(2)
        x = self.head(x)
        return x

    def createModel(self):
        self.model = torchvision.models.densenet121()
        # 生成model后，在根据实际任务，看是否需要预加载
        return pretrainedLoad(self.model, self.cfg['pretrained'])

    def changeModelStructure(self):
        in_features = self.model.classifier.in_features
        self.head = createFullConnectionLayer(self.cfg['dropout'], in_features, self.cfg['class_number'])
        self.backbone = self.model.features
