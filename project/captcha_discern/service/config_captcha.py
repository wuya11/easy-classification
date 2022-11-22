"""
@File    :   config.py
@Contact :   1129137758@qq.com

@Modify Time      @Author    @Version
------------      -------    --------
2022/9/27 16:48   WangLing      1.0

@Description :  验证码-分类基础配置文件对象
"""

cfg = {
    ### Global Set
    "model_name": "mobilenetv3",
    #shufflenetv2 adv-efficientnet-b2 se_resnext50_32x4d  xception 
    'GPU_ID': '',
    # 验证码的one-hot 一共输出 62*4个对象
    "class_number": 248,

    "random_seed":42,
    "cfg_verbose":True,
    "num_workers":8,


    ### Train Setting
    'train_path':"data/train",
    'val_path': "data/val",
    'test_path': "data/test",  # test without label, just show img result

    #../data/dataset/e_test
    #../data/dataset/d_trainval/v8/train/
    'label_type': 'DIR',# path or 'DIR' 
    'label_path': '',# if 'DIR' quale  train_path
     'pretrained':'', #path or ''

    'try_to_train_items': 0,   # 0 means all, or run part(200 e.g.) for bug test
    'save_best_only': True,  #only save model if better than before
    'save_one_only':True,    #only save one best model (will del model before)
    "save_dir": "output/",
    'metrics': ['acc'], # default is acc,  can add F1  ...
    "loss": 'CE', # CE, CEV2-0.5, Focalloss-1, BCE...

    'show_heatmap':False,
    'show_data':False,


    ### Train Hyperparameters
    "target_img_size": [224, 224], # [h, w]
    'learning_rate':0.001,
    'batch_size':64,
    'epochs':100,
    'optimizer':'Adam',  #Adam  SGD AdaBelief Ranger
    'scheduler':'default-0.1-3', #default  SGDR-5-2    step-4-0.8

    'warmup_epoch':0, # 
    'weight_decay' : 0,#0.0001,
    "k_flod":5,
    'start_fold':0,
    'early_stop_patient':10,

    'use_distill':0,
    'label_smooth':0,
    # 'checkpoint':None,
    'class_weight': None,#s[1.4, 0.78], # None [1, 1]
    'clip_gradient': 0,#1,       # 0
    'freeze_nonlinear_epoch':0,

    'dropout':0.5, #before last_linear

    'mixup':False,
    'cutmix':False,
    'sample_weights':None,


    ### Test
    'model_path':'output/mobilenetv3_e20_0.85280.pth',#test model
    
    'TTA':False,
    'merge':False,
    'test_batch_size': 1,
    

}
