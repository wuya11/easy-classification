# -*- encoding: utf-8 -*-
"""
@File    :   classification_model_enum.py    
@Contact :   1129137758@qq.com

@Modify Time      @Author    @Version    
------------      -------    --------   
2022/9/28 15:25   WangLing      1.0        

@Description :编写一个枚举类，定义分类项目，所有的分类网络模型
"""
from enum import Enum


class ClassificationModelEnum(Enum):
    MOBILENET_V2 = ("mobilenetv2", "MobileNetV2")
    MOBILENET_V3 = ("mobilenetv3", "MobileNetV3")
    EFFICIENTNET_AUTO_AUGMENT = ("efficientnet_auto_augment", "efficientnet_auto_augment")
    EFFICIENTNET_ADVPROP = ("efficientnet_advprop", "efficientnet_advprop")
    EFFICIENTNET_V2 = ("efficientnet_v2", "efficientnet_v2")
    DENSENET = ("densenet", "densenet")


    # 暂时还不支持
    MICRONET = ("micronet", "micronet")
    SHUFFLE_NET_V2 = ("shufflenetv2", "shufflenetv2")
    SWIN = ("swin", "swin")
    CONV_NEXT = ("convnext", "convnext")
    RES_NEXT = ("resnext", "resnext")
    REGNET = ("RegNet", "RegNet")
    XCEPTION = ("xception", "xception")

    def __int__(self, model_key, model_name):
        self.model_key = model_key
        self.model_name = model_name

    @staticmethod
    def getModelType(model_config_name):
        match_model = None
        for model in ClassificationModelEnum:
            if model_config_name in model.value[0]:
                match_model = model
                break
        if match_model is None:
            raise Exception("[ERROR] Unknown model_config_name: ", model_config_name)
        return match_model
