"""
@File    :   train.py
@Contact :   1129137758@qq.com

@Modify Time      @Author    @Version
------------      -------    --------
2022/9/28 16:48   WangLing      1.0

@Description : 验证码-训练集启动脚本
"""
from project.captcha_discern.service.captcha_dataset import TrainDataset, EvalDataset
from project.captcha_discern.service.config_captcha import cfg
from project.captcha_discern.service.runner_captcha_service import RunnerCaptchaService
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
    runner = RunnerCaptchaService(cfg, model)
    runner.train(train_loader, val_loader)


if __name__ == '__main__':
    main(cfg)
