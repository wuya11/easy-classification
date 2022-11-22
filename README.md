# Easy-Classification-分类框架说明文档

(https://github.com/wuya11/easy-classification)

## 1. 前言
Easy-Classification是一个应用于分类任务的深度学习框架，它集成了众多成熟的分类神经网络模型，可帮助使用者简单快速的构建分类训练任务。
### 1.1 框架功能
#### 1.1.1 数据加载
* 文件夹形式
* 其它自定义形式，在项目应用中，参考案例编写DataSet自定义加载。如基于配置文件，csv,路径解析等。


#### 1.1.2 扩展网络
本框架扩展支持如下网络模型，可在classification_model_enum.py枚举类中查看具体的model。
- Resnet系列，Densenet系列，VGGnet系列等所有[pretrained-models.pytorch](https://github.com/Cadene/pretrained-models.pytorch)支持的网络
- [Mobilenetv2](https://pytorch.org/docs/stable/torchvision/models.html?highlight=mobilenet#torchvision.models.mobilenet_v2)，[Mbilenetv3](https://github.com/kuan-wang/pytorch-mobilenet-v3)
- ShuffleNetV2，[MicroNet](https://github.com/liyunsheng13/micronet)
-  [EfficientNet](https://github.com/lukemelas/EfficientNet-PyTorch)
-  [Swin Transformer](https://github.com/microsoft/Swin-Transformer)
- [ConvNeXt](https://github.com/facebookresearch/ConvNeXt)
#### 1.1.3 优化器
- Adam  
- SGD 
- AdaBelief 
- [Ranger](https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer)
-  AdamW
#### 1.1.4 学习率衰减
- ReduceLROnPlateau
- StepLR
- MultiStepLR
- SGDR
#### 1.1.5 损失函数
- 直接调用PyTorch相关的损失函数
- 交叉熵
- Focalloss
#### 1.1.6 其他
- Metric(acc, F1)
- 训练结果acc,loss过程图片保存
- 交叉验证
- 梯度裁剪
- Earlystop
-  weightdecay
- 冻结/解冻 除最后的全连接层的特征层

## 2. 框架设计
Easy-Classification是一个简单轻巧的分类框架，目前版本主要包括两大模块，框架通用模块和项目应用模块。为方便用户快速体验，框架中目前包括简单手写数字识别和验证码识别两个示例项目。
- 深度学习-训练训练流程说明：
- 框架设计方案参考文档：
- 简单手写数字识别：
- 验证码识别：

## 3. 参考文献
1. [albumentations](https://github.com/albumentations-team/albumentations)
2. [warmup](https://github.com/ildoonet/pytorch-gradual-warmup-lr)
3. [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)
4. [![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000) 易大师](https://github.com/fire717/Fire/blob/main/LICENSE)