"""
@File    :   mobilenet_v2.py
@Contact :   1129137758@qq.com

@Modify Time      @Author    @Version
------------      -------    --------
2022/9/27 16:48   WangLing      1.0

@Description : create a MobileNetV2 model
"""

import torchvision
from torch import nn

from universe.utils.utils import pretrainedLoad, createFullConnectionLayer


class MobileNetV2(nn.Module):

    def __init__(self, cfg):
        super(MobileNetV2, self).__init__()
        self.cfg = cfg
        self.createModel()
        self.changeModelStructure()

    def forward(self, x):
        out = self.backbone(x)
        out = nn.functional.adaptive_avg_pool2d(out, 1).reshape(out.shape[0], -1)
        out = self.head(out)
        return out

    def createModel(self):
        self.model = torchvision.models.mobilenet_v2(pretrained=False, progress=True, width_mult=1.0)
        # 生成model后，在根据实际任务，看是否需要预加载
        return pretrainedLoad(self.model, self.cfg['pretrained'])

    def changeModelStructure(self):
        in_features = self.model.classifier[1].in_features
        self.backbone = self.model.features
        self.head = createFullConnectionLayer(self.cfg['dropout'], in_features, self.cfg['class_number'])

