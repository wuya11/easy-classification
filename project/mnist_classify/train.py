"""
@File    :   train.py
@Contact :   1129137758@qq.com

@Modify Time      @Author    @Version
------------      -------    --------
2022/9/28 16:48   WangLing      1.0

@Description : 简单数字识别-训练集启动脚本
"""

from project.mnist_classify.service.mnist_config import cfg
from project.mnist_classify.service.mnist_dataset import TrainDataset, EvalDataset
from project.mnist_classify.service.mnist_runner_service import MnistRunnerService
from universe.data_load.data_load_service import DataLoadService
from universe.model.model_service import ModelService
from universe.utils.utils import initConfig


def main(cfg):
    initConfig(cfg)
    model = ModelService(cfg)
    data = DataLoadService(cfg)

    # train_loader, val_loader = data.getTrainValDataloader(TrainDataset)
    train_loader = data.getTrainDataloader(TrainDataset)
    val_loader = data.getValDataloader(EvalDataset)
    runner = MnistRunnerService(cfg, model)
    runner.train(train_loader, val_loader)

if __name__ == '__main__':
    main(cfg)





