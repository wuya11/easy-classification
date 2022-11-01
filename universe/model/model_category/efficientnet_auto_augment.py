"""
@File    :   efficientnet_auto_augment.py
@Contact :   1129137758@qq.com

@Modify Time      @Author    @Version
------------      -------    --------
2022/9/27 16:48   WangLing      1.0

@Description : create a efficient_net model. you can update code,use B0-B7
              不支持对抗训练
              advprop 这个参数控制
              权重文件来自: https://github.com/lukemelas/EfficientNet-PyTorch, 模型实现地址: https://aistudio.baidu.com/aistudio/projectdetail/1536845
              入参注意：
              efficientnet-b7  为[600,600,3] 其他版本注意入参大小调整，图片大小不一样，修改配置文件config.train_img_size
"""
from torch import nn

from universe.model.model_category.myefficientnet_pytorch import EfficientNet


class EfficientnetAutoAugment(nn.Module):

    def __init__(self, cfg):
        super(EfficientnetAutoAugment, self).__init__()
        self.cfg = cfg
        self.createModel()

    def forward(self, x):
        x = self.model.forward(x)
        return x

    def createModel(self):
        self.model = EfficientNet.from_name('efficientnet-b7', num_classes=self.cfg['class_number'])
        if self.cfg['pretrained']:
            self.model= EfficientNet.from_pretrained('efficientnet-b7', weights_path=self.cfg['pretrained'],
                                                  num_classes=self.cfg['class_number'], advprop=False)
        return self.model
