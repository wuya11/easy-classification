"""
@File    :   normalize_adapter.py
@Contact :   1129137758@qq.com

@Modify Time      @Author    @Version
------------      -------    --------
2022/9/27 16:48   WangLing      1.0

@Description : 归一化信息适配类
"""

import torchvision.transforms as transforms


class NormalizeAdapter:

    @staticmethod
    def getNormalize(model_name):
        if model_name in ['mobilenetv2', 'mobilenetv3']:
            my_normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        elif model_name == 'xception':
            my_normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        elif model_name == 'efficientnet_auto_augment':
            my_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        elif model_name == 'efficientnet_advprop':
            my_normalize = transforms.Lambda(lambda img: img * 2.0 - 1.0)
        elif "resnex" in model_name or 'eff' in model_name or 'RegNet' in model_name:
            my_normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            # my_normalize = transforms.Normalize([0.4783, 0.4559, 0.4570], [0.2566, 0.2544, 0.2522])
        elif "EN-B" in model_name:
            my_normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        else:
            print("[Info] Not set normalize type! Use defalut imagenet normalization.")
            my_normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return my_normalize
