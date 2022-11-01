"""
@File    :   model_service.py
@Contact :   1129137758@qq.com

@Modify Time      @Author    @Version
------------      -------    --------
2022/9/27 16:48   WangLing      1.0

@Description : 模型初始化服务
"""

import torch.nn as nn

from config.classification_model_enum import ClassificationModelEnum
from universe.model.model_category.densenet import Densenet
from universe.model.model_category.efficientnet_advprop import EfficientnetAutoAdvprop
from universe.model.model_category.efficientnet_auto_augment import EfficientnetAutoAugment
from universe.model.model_category.efficientnet_v2 import EfficientnetV2
from universe.model.model_category.mobilenet_v2 import MobileNetV2
from universe.model.model_category.mobilenet_v3 import MobileNetV3


class ModelService(nn.Module):
    """
        参考java 接口编程思维，定义神经网络模型服务

        外部应用调用时，只对接ModelService

        ModelService可根据配置，自动加载对应的model

    """

    def __init__(self, cfg):
        super(ModelService, self).__init__()
        self.match_model = None
        self.cfg = cfg
        self.createModel()

    def createModel(self):
        """
        构建一个模型对象，如mobilenetv3

        Args:

        Returns: 返回一个模型

        """
        model_type = ClassificationModelEnum.getModelType(self.cfg['model_name'])
        if model_type is ClassificationModelEnum.MOBILENET_V2:
            self.match_model = MobileNetV2(self.cfg)
        if model_type is ClassificationModelEnum.MOBILENET_V3:
            self.match_model = MobileNetV3(self.cfg)
        if model_type is ClassificationModelEnum.EFFICIENTNET_AUTO_AUGMENT:
            self.match_model = EfficientnetAutoAugment(self.cfg)
        if model_type is ClassificationModelEnum.EFFICIENTNET_ADVPROP:
            self.match_model = EfficientnetAutoAdvprop(self.cfg)
        if model_type is ClassificationModelEnum.EFFICIENTNET_V2:
            self.match_model = EfficientnetV2(self.cfg)
        if model_type is ClassificationModelEnum.DENSENET:
            self.match_model = Densenet(self.cfg)
        if self.match_model is None:
            raise Exception("[ERROR] Unknown model_name: ", self.cfg['model_name'])

    def forward(self, tensorData):
        """
        定义模型输出函数

        Args:
            tensorData: 模型入参，张量对象

        Returns:输出模型训练网络训练后的张量信息

        """
        return self.match_model.forward(tensorData)
