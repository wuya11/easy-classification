"""
@File    :   predict.py
@Contact :   1129137758@qq.com

@Modify Time      @Author    @Version
------------      -------    --------
2022/9/29 16:48   WangLing      1.0

@Description : 简单数字识别-预测/应用启动脚本
"""

import os
import pandas as pd

from project.mnist_classify.service.mnist_config import cfg
from project.mnist_classify.service.mnist_dataset import PredictDataset
from project.mnist_classify.service.mnist_runner_service import MnistRunnerService
from universe.data_load.data_load_service import DataLoadService
from universe.model.model_service import ModelService
from universe.utils.utils import initConfig, modelLoad


def predict(cfg):
    initConfig(cfg)
    model = ModelService(cfg)
    data = DataLoadService(cfg)

    test_loader = data.getPredictDataloader(PredictDataset)

    runner = MnistRunnerService(cfg, model)
    modelLoad(model, cfg['model_path'])
    res_dict = runner.predict(test_loader)
    print(len(res_dict))

    # to csv
    res_df = pd.DataFrame.from_dict(res_dict, orient='index', columns=['label'])
    res_df = res_df.reset_index().rename(columns={'index': 'image_id'})
    res_df.to_csv(os.path.join(cfg['save_dir'], 'pre.csv'),
                  index=False, header=True)


def main(cfg):
    predict(cfg)


if __name__ == '__main__':
    main(cfg)
