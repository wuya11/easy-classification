"""
@File    :   runn_service.py
@Contact :   1129137758@qq.com

@Modify Time      @Author    @Version
------------      -------    --------
2022/9/27 16:48   WangLing      1.0

@Description : 模型初始化服务
"""

import time
import gc
import os
import datetime
import torch
import torch.nn as nn
import numpy as np
import cv2

from universe.runner.loss_function_adapter import LossFunctionAdapter
from universe.runner.metrics import getF1
from universe.runner.optimizer_adapter import OptimizerAdapter, clipGradient
from universe.runner.scheduler_adapter import SchedulerAdapter, GradualWarmupScheduler
from universe.utils.utils import printDash, draw_loss, draw_acc


def printCaptcha(pred):
    number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
                'v', 'w', 'x', 'y', 'z']
    ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
                'V', 'W', 'X', 'Y', 'Z']
    labels = number + ALPHABET + alphabet
    for i in pred:
        print(labels[i.item()], end='')


source = [str(i) for i in range(0, 10)]
source += [chr(i) for i in range(97, 97 + 26)]
source += [chr(i) for i in range(65, 65 + 26)]
alphabet = ''.join(source)


class RunnerCaptchaService:
    def __init__(self, cfg, model):

        self.cfg = cfg
        if self.cfg['GPU_ID'] != '':
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model = model.to(self.device)

        self.scaler = torch.cuda.amp.GradScaler()

        # loss
        self.loss_func = nn.MultiLabelSoftMarginLoss()

        # optimizer
        self.optimizer = OptimizerAdapter.getOptimizer(self.cfg['optimizer'],
                                                       self.model,
                                                       self.cfg['learning_rate'],
                                                       self.cfg['weight_decay'])

        # scheduler
        self.scheduler = SchedulerAdapter.getScheduler(self.cfg['scheduler'], self.optimizer)

        if self.cfg['warmup_epoch']:
            self.scheduler = GradualWarmupScheduler(self.optimizer,
                                                    multiplier=1,
                                                    total_epoch=self.cfg['warmup_epoch'],
                                                    after_scheduler=self.scheduler)

    def freezeBeforeLinear(self, epoch, freeze_epochs=2):
        if epoch < freeze_epochs:
            for child in list(self.model.children())[:-1]:
                for param in child.parameters():
                    param.requires_grad = False
        elif epoch == freeze_epochs:
            for child in list(self.model.children())[:-1]:
                for param in child.parameters():
                    param.requires_grad = True

    def train(self, train_loader, val_loader):
        """
        定义训练集流程处理方法，执行流程包括

        1.先定义开始时的部分全局变量--onTrainStart()

        2.循环做外层次数遍历
           2.1.冻结层过滤处理--非必须
           2.2.训练集数据处理
           2.3.验证集数据处理
           2.4.学习率调整
           2.5.保存最优权重信息
           2.6.过早结束处理
        3.结束训练，释放资源

        :param train_loader:训练集数据
        :param val_loader: 验证集数据
        """
        self.onTrainStart()
        for epoch in range(self.cfg['epochs']):
            self.freezeBeforeLinear(epoch, self.cfg['freeze_nonlinear_epoch'])
            self.onTrainStep(train_loader, epoch)
            self.onValidation(val_loader)
            self.schedulerUpdate(epoch)
            self.checkpoint(epoch)
            self.earlyStop(epoch)
            if self.earlystop:
                break
        draw_loss(self.tran_loss_history, self.val_loss_history, epoch, self.cfg['save_dir'])
        draw_acc(self.tran_acc_history, self.val_acc_history, epoch, self.cfg['save_dir'])
        self.onTrainEnd()

    def predictRaw(self, data_loader):
        self.model.eval()
        correct = 0

        res_dict = {}
        with torch.no_grad():
            # pres = []
            # labels = []
            for (data, img_names) in data_loader:
                data = data.to(self.device)

                output = self.model(data).double()

                # print(output.shape)
                pred_score = nn.Softmax(dim=1)(output)
                # print(pred_score.shape)
                pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability

                batch_pred_score = pred_score.data.cpu().numpy().tolist()
                for i in range(len(batch_pred_score)):
                    res_dict[os.path.basename(img_names[i])] = pred_score[i].cpu().numpy()

        # pres = np.array(pres)

        return res_dict

    def predict(self, data_loader):
        self.model.eval()
        correct = 0
        res_dict = {}
        with torch.no_grad():
            pres = []
            labels = []
            # pres = []
            # labels = []
            for (data, real_label, img_names) in data_loader:
                data = data.to(self.device)

                output = self.model(data).float()
                output = output.view(-1, 62)
                output = nn.functional.softmax(output, dim=1)
                output = torch.argmax(output, dim=1)
                output = output.view(-1, 4)[0]

                number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
                alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
                            's', 't', 'u',
                            'v', 'w', 'x', 'y', 'z']
                ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
                            'S', 'T', 'U',
                            'V', 'W', 'X', 'Y', 'Z']
                alphabet = number + ALPHABET + alphabet

                label = ''.join([alphabet[i] for i in output.cpu().numpy()])
                res_dict[os.path.basename(img_names[0])] = label
                count=0
                if label != real_label[0]:
                    count+=1
                    print("[INFO] Faith match , real_label: %s ---pre_label:%s" % (real_label[0], label))
            print(count)

        return res_dict

    def cleanData(self, data_loader, target_label, move_dir):
        """
        input: data, move_path
        output: None

        """
        self.model.eval()

        # predict
        # res_list = []
        count = 0
        with torch.no_grad():
            # end = time.time()
            for i, (inputs, target, img_names) in enumerate(data_loader):
                print("\r", str(i) + "/" + str(data_loader.__len__()), end="", flush=True)

                inputs = inputs.cuda()

                output = self.model(inputs)
                output = nn.Softmax(dim=1)(output)
                output = output.data.cpu().numpy()

                for i in range(output.shape[0]):

                    output_one = output[i][np.newaxis, :]

                    if np.argmax(output_one) != target_label:
                        print(output_one, target_label, img_names[i])
                        img_name = os.path.basename(img_names[i])
                        os.rename(img_names[i], os.path.join(move_dir, img_name))
                        count += 1
        print("[INFO] Total: ", count)

    def evaluate(self, data_loader):
        self.model.eval()
        correct = 0

        with torch.no_grad():
            pres = []
            labels = []
            for (data, target, img_names) in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                # print(target.shape)
                with torch.cuda.amp.autocast():
                    output = self.model(data).double()

                # print(img_names)
                # print(output)
                pred_score = nn.Softmax(dim=1)(output)
                # print(pred_score)
                # b
                # print(pred_score.shape)
                pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                if self.cfg['use_distill']:
                    target = target.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()

                batch_pred_score = pred_score.data.cpu().numpy().tolist()
                batch_label_score = target.data.cpu().numpy().tolist()
                # print(batch_pred_score)
                # print(batch_label_score)
                # b
                pres.extend(batch_pred_score)
                labels.extend(batch_label_score)

        pres = np.array(pres)
        labels = np.array(labels)
        # print(pres.shape, labels.shape)

        acc = correct / len(data_loader.dataset)

        print('[Info] acc: {:.3f}% \n'.format(100. * acc))

        if 'F1' in self.cfg['metrics']:
            precision, recall, f1_score = getF1(pres, labels)
            print('      precision: {:.5f}, recall: {:.5f}, f1_score: {:.5f}\n'.format(
                precision, recall, f1_score))

    def onTrainStart(self):

        self.early_stop_value = 0
        self.early_stop_dist = 0
        self.last_save_path = None

        self.earlystop = False
        self.best_epoch = 0

        # 记录每一次的acc,loss，便于后面画图
        self.tran_loss_history = []
        self.tran_acc_history = []
        self.val_loss_history = []
        self.val_acc_history = []

        # log
        self.log_time = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))

    def onTrainStep(self, train_loader, epoch):

        self.model.train()
        correct = 0
        count = 0
        batch_time = 0
        total_loss = 0
        train_acc = 0
        for batch_idx, (data, target, img_names) in enumerate(train_loader):
            one_batch_time_start = time.time()
            target = target.to(self.device)
            data = data.to(self.device)
            with torch.cuda.amp.autocast():
                output = self.model(data).float()
                loss = self.loss_func(output, target)

            total_loss += loss.item()
            if self.cfg['clip_gradient']:
                clipGradient(self.optimizer, self.cfg['clip_gradient'])

            self.optimizer.zero_grad()  # 把梯度置零
            # loss.backward() #计算梯度
            # self.optimizer.step() #更新参数
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            ### train acc
            count += len(data)
            train_loss = total_loss / count
            correct += self.get_correct(output, target)
            train_acc = correct / count
            # print(train_acc)
            one_batch_time = time.time() - one_batch_time_start
            batch_time += one_batch_time
            # print(batch_time/(batch_idx+1), len(train_loader), batch_idx,
            #     int(one_batch_time*(len(train_loader)-batch_idx)))
            eta = int((batch_time / (batch_idx + 1)) * (len(train_loader) - batch_idx - 1))

            print_epoch = ''.join([' '] * (4 - len(str(epoch + 1)))) + str(epoch + 1)
            print_epoch_total = str(self.cfg['epochs']) + ''.join([' '] * (4 - len(str(self.cfg['epochs']))))
            log_interval = 10
            if batch_idx % log_interval == 0:
                print('\r',
                      '{}/{} [{}/{} ({:.0f}%)] - ETA: {}, loss: {:.4f}, acc: {:.4f}  LR: {:f}'.format(
                          print_epoch, print_epoch_total, batch_idx * len(data), len(train_loader.dataset),
                                                          100. * batch_idx / len(train_loader),
                          datetime.timedelta(seconds=eta),
                          train_loss, train_acc,
                          self.optimizer.param_groups[0]["lr"]),
                      end="", flush=True)
        self.tran_loss_history.append(train_loss)
        self.tran_acc_history.append(train_acc)

    def onTrainEnd(self):

        del self.model
        gc.collect()
        torch.cuda.empty_cache()

        if self.cfg["cfg_verbose"]:
            printDash()
            print(self.cfg)
            printDash()

    def get_correct(self, output, target):
        correct = 0
        output, target = output.view(-1, 62), target.view(-1, 62)
        output = nn.functional.softmax(output, dim=1)
        output = torch.argmax(output, dim=1)
        target = torch.argmax(target, dim=1)
        output, target = output.view(-1, 4), target.view(-1, 4)
        correct_list = []
        for i, j in zip(target, output):
            if torch.equal(i, j):
                correct_list.append(1)
            else:
                correct_list.append(0)
        correct = sum(correct_list)
        return correct

    def onValidation(self, val_loader):

        self.model.eval()
        self.val_loss = 0
        self.correct_num = 0
        self.val_acc = 0
        with torch.no_grad():
            for (data, target, img_names) in val_loader:
                data, target = data.to(self.device), target.to(self.device)

                with torch.cuda.amp.autocast():
                    output = self.model(data).float()
                    self.val_loss += self.loss_func(output, target).item()  # sum up batch loss
                    self.correct_num += self.get_correct(output, target)

        self.val_loss /= len(val_loader.dataset)
        self.val_acc = self.correct_num / len(val_loader.dataset)
        self.val_loss_history.append(self.val_loss)
        self.val_acc_history.append(self.val_acc)
        self.best_score = self.val_acc
        print(' \n           [VAL] loss: {:.5f}, acc: {:.3f}% \n'.format(
            self.val_loss, 100. * self.val_acc))

    def schedulerUpdate(self, epoch):
        if self.cfg['warmup_epoch']:
            self.scheduler.step(epoch)
        else:
            if 'default' in self.cfg['scheduler']:
                # self.scheduler.step(self.val_acc)
                self.scheduler.step(self.val_acc)
            else:
                self.scheduler.step()

    def earlyStop(self, epoch):
        ### earlystop
        if self.val_acc > self.early_stop_value:
            self.early_stop_value = self.val_acc
        if self.best_score > self.early_stop_value:
            self.early_stop_value = self.best_score
            self.early_stop_dist = 0

        self.early_stop_dist += 1
        if self.early_stop_dist > self.cfg['early_stop_patient']:
            self.best_epoch = epoch - self.cfg['early_stop_patient'] + 1
            print("[INFO] Early Stop with patient %d , best is Epoch - %d :%f" % (
                self.cfg['early_stop_patient'], self.best_epoch, self.early_stop_value))
            self.earlystop = True
        if epoch + 1 == self.cfg['epochs']:
            self.best_epoch = epoch - self.early_stop_dist + 2
            print("[INFO] Finish trainging , best is Epoch - %d :%f" % (self.best_epoch, self.early_stop_value))
            self.earlystop = True

    def checkpoint(self, epoch):

        if self.val_acc <= self.early_stop_value:
            if self.best_score <= self.early_stop_value:
                if self.cfg['save_best_only']:
                    pass
                else:
                    self.saveModel(epoch)
            else:
                self.saveModel(epoch)
        else:
            self.saveModel(epoch)

    def saveModel(self, epoch):
        if self.cfg['save_one_only']:
            if self.last_save_path is not None and os.path.exists(self.last_save_path):
                os.remove(self.last_save_path)
        save_name = '%s_e%d_%.5f.pth' % (self.cfg['model_name'], epoch + 1, self.best_score)
        self.last_save_path = os.path.join(self.cfg['save_dir'], save_name)
        torch.save(self.model.state_dict(), self.last_save_path)