"""
@File    :   mnist_dataset.py
@Contact :   1129137758@qq.com

@Modify Time      @Author    @Version
------------      -------    --------
2022/10/28 16:48   WangLing      1.0

@Description : 构建Dataset类，不同的任务，dataset自行编写，如基于csv，文本等加载标签，均可从cfg配置文件中读取配置信息后，自行扩展编写

编写自定义Dataset类时，初始化参数需定义为source_img, cfg。否则数据加载通用模块，data_load_service.py模块会报错。

source_img :传入的图像地址信息集合

cfg：传入的配置类信息，针对不同的任务，可能生成的label模式不同，可基于配置类指定label的加载模式，最终为训练的图像初始化label （用户自定义实现）

本例为 简单首先数字10分类（0-9）：基于文件夹名称(路径)做label
"""
import os

from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
import cv2

from universe.data_load.normalize_adapter import NormalizeAdapter
from PIL import Image




class TrainDataset(Dataset):
    """
    构建一个 加载原始图片的dataSet对象

    此函数可加载 训练集数据,基于路径识别验证码真实的label，label在转换为one-hot编码

    若 验证集逻辑与训练集逻辑一样，验证集可使用TrainDataset，不同，则需自定义一个，参考如下EvalDataset
    """

    def __init__(self, source_img, cfg):
        self.source_img = source_img
        self.cfg = cfg
        self.transform = createTransform(cfg, TrainImgDeal)
        self.label_dict = getLabels(cfg['train_path'], source_img)

    def __getitem__(self, index):
        img = cv2.imread(self.source_img[index])
        if self.transform is not None:
            img = self.transform(img)
        target = self.label_dict[self.source_img[index]]
        return img, target, self.source_img[index]

    def __len__(self):
        return len(self.source_img)


class EvalDataset(Dataset):
    """
    构建一个 加载原始图片的dataSet对象

    此函数可加载 验证集数据,基于路径识别验证码真实的label，label在转换为one-hot编码
    """

    def __init__(self, source_img, cfg):
        self.source_img = source_img
        self.cfg = cfg
        # 若验证集图片处理逻辑（增强，调整）与 训练集不同，可自定义一个EvalImgDeal
        self.transform = createTransform(cfg, TrainImgDeal)
        self.label_dict = getLabels(cfg['val_path'], source_img)

    def __getitem__(self, index):
        img = cv2.imread(self.source_img[index])
        if self.transform is not None:
            img = self.transform(img)
        target = self.label_dict[self.source_img[index]]
        return img, target, self.source_img[index]

    def __len__(self):
        return len(self.source_img)


class PredictDataset(Dataset):
    """
        构建一个 加载预测图片的dataSet对象

        此函数可加载 测试集数据，应用集数据（返回图像信息）
    """

    def __init__(self, source_img, cfg):
        self.source_img = source_img
        # 若预测集图片处理逻辑（增强，调整）与 训练集不同，可自定义一个PredictImgDeal
        self.transform = createTransform(cfg, TrainImgDeal)

    def __getitem__(self, index):
        img = cv2.imread(self.source_img[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, self.source_img[index]

    def __len__(self):
        return len(self.source_img)


class TrainImgDeal:
    def __init__(self, cfg):
        img_size = cfg['target_img_size']
        self.h = img_size[0]
        self.w = img_size[1]

    def __call__(self, img):
        img = cv2.resize(img, (self.h, self.w))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        # img = A.OneOf([A.ShiftScaleRotate(
        #                         shift_limit=0.1,
        #                         scale_limit=0.1,
        #                         rotate_limit=30,
        #                         interpolation=cv2.INTER_LINEAR,
        #                         border_mode=cv2.BORDER_CONSTANT,
        #                          value=0, mask_value=0,
        #                         p=0.5),
        #                 A.GridDistortion(num_steps=5, distort_limit=0.2,
        #                     interpolation=1, border_mode=4, p=0.4),
        #                 A.RandomGridShuffle(grid=(3, 3),  p=0.3)],
        #                 p=0.5)(image=img)['image']

        # img = A.HorizontalFlip(p=0.5)(image=img)['image']
        # img = A.VerticalFlip(p=0.4)(image=img)['image']

        # img = A.OneOf([A.RandomBrightness(limit=0.1, p=1),
        #             A.RandomContrast(limit=0.1, p=1),
        #             A.RandomGamma(gamma_limit=(50, 150),p=1),
        #             A.HueSaturationValue(hue_shift_limit=10,
        #                 sat_shift_limit=10, val_shift_limit=10,  p=1)],
        #             p=0.6)(image=img)['image']

        # img = A.Resize(self.h,self.w,cv2.INTER_LANCZOS4,p=1)(image=img)['image']
        # img = A.OneOf([A.GaussianBlur(blur_limit=3, p=0.1),
        #                 A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5),
        #                 A.GaussNoise(var_limit=(10.0, 50.0), mean=0, p=0.4)],
        #                 p=0.4)(image=img)['image']

        # img = A.CoarseDropout(max_holes=3, max_height=20, max_width=20,
        #                     p=0.8)(image=img)['image']

        return img


def getLabels(label_path, source_img):
    cate_dirs = os.listdir(label_path)
    cate_dirs.sort()
    label_dict = {}
    for i, img_path in enumerate(source_img):
        img_dirs = img_path.replace(label_path, '')
        img_dirs = img_dirs.split(os.sep)[:2]
        img_dir = img_dirs[0] if img_dirs[0] else img_dirs[1]
        y = cate_dirs.index(img_dir)
        label_dict[img_path] = y
    return label_dict


def createTransform(cfg, img_deal):
    my_normalize = NormalizeAdapter.getNormalize(cfg['model_name'])
    transform = transforms.Compose([
        img_deal(cfg),
        transforms.ToTensor(),
        my_normalize,
    ])
    return transform
