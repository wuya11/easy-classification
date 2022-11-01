"""
@File    :   mobilenet_v3.py
@Contact :   1129137758@qq.com

@Modify Time      @Author    @Version
------------      -------    --------
2022/9/27 16:48   WangLing      1.0

@Description : create a MobileNetV3-large model
"""

import torchvision
from torch import nn

from universe.utils.utils import pretrainedLoad, createFullConnectionLayer


class MobileNetV3(nn.Module):

    def __init__(self, cfg):
        super(MobileNetV3, self).__init__()
        self.cfg = cfg
        self.createModel()
        self.changeModelStructure()

    def forward(self, x):
        x = self.backbone(x)
        x = x.mean(3).mean(2)
        x = self.head(x)
        return x

    def createModel(self):
        self.model = torchvision.models.mobilenet_v3_large(pretrained=True)
        # 生成model后，在根据实际任务，看是否需要预加载
        return pretrainedLoad(self.model, self.cfg['pretrained'])

    def changeModelStructure(self):
        self.backbone = nn.Sequential(*list(self.model.children())[:-1])
        self.head = createFullConnectionLayer(self.cfg['dropout'], 960, self.cfg['class_number'])
