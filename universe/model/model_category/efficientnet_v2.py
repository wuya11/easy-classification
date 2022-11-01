"""
@File    :   efficientnet_v2.py
@Contact :   1129137758@qq.com

@Modify Time      @Author    @Version
------------      -------    --------
2022/9/27 16:48   WangLing      1.0

@Description : create a efficientnet_v2 model. you can update code,use S,L,M,XL(21K)
               相关知识参考：https://zhuanlan.zhihu.com/p/361947957
              入参注意：
                为[224,224,3] 其他版本注意入参大小调整，图片大小不一样，修改配置文件config.train_img_size
"""
from torch import nn


from universe.model.model_category.myefficientnet_pytorch.modelV2 import  effnetv2_xl
from universe.utils.utils import pretrainedLoad


class EfficientnetV2(nn.Module):

    def __init__(self, cfg):
        super(EfficientnetV2, self).__init__()
        self.cfg = cfg
        self.createModel()

    def forward(self, x):
        x = self.model.forward(x)
        return x

    def createModel(self):
        self.model = effnetv2_xl(num_classes=self.cfg['class_number'])
        return pretrainedLoad(self.model, self.cfg['pretrained'])
        return self.model
