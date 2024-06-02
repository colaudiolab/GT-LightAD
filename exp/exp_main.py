
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from torch.optim import lr_scheduler
from metrics.metrics import *
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np
from optimizer import Lion
warnings.filterwarnings('ignore')
from data_provider.data_loader import get_loader_segment


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.args = args
        self.get_data()

    def get_data(self):
        self.train_loader = get_loader_segment(self.args.data_path, batch_size=self.args.batch_size,
                                               seq_len=self.args.seq_len,pre_len = self.args.pre_len,
                                               mode='train',
                                               dataset=self.args.dataset)
        self.vali_loader = get_loader_segment(self.args.data_path, batch_size=self.args.batch_size,
                                              seq_len=self.args.seq_len,pre_len = self.args.pre_len,
                                              mode='val',
                                              dataset=self.args.dataset)
        self.test_loader = get_loader_segment(self.args.data_path, batch_size=self.args.batch_size,
                                              seq_len=self.args.seq_len,pre_len = self.args.pre_len,
                                              mode='test',
                                              dataset=self.args.dataset)
        self.thre_loader = get_loader_segment(self.args.data_path, batch_size=self.args.batch_size,
                                              seq_len=self.args.seq_len,pre_len = self.args.pre_len,
                                              mode='thre',
                                              dataset=self.args.dataset)



    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self,vali_loader):
        vali_criterion = nn.MSELoss(reduce=False)
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            tqdm_vali_loader = tqdm(vali_loader)
            for i, (batch_x, batch_y, labels) in enumerate(tqdm_vali_loader ):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                outputs = self.model(batch_x)
                pred = outputs
                true = batch_y
                loss = torch.mean(vali_criterion(pred, true), dim=-1)
                total_loss.append(loss)
        total_loss = torch.cat(total_loss, 0)
        total_loss = torch.mean(total_loss)

        self.model.train()
        return total_loss

    def train(self, setting):

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(self.train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            tqdm_train_loader = tqdm(self.train_loader)
            for i, (batch_x, batch_y, labels) in enumerate(tqdm_train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 10000 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(self.vali_loader)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            self.test(setting)
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            #adjust_learning_rate(model_optim, epoch + 1, self.args)
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def test(self, setting):

        print('loading model')
        self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
        print('>>>>>>>start testing>>>>>>>>>>>>')
        criterion = nn.MSELoss(reduce=False)
        self.model.eval()

        test_reconstitution_error = []
        with torch.no_grad():
            tqdm_test_loader = tqdm(self.thre_loader)
            for i, (batch_x, batch_y, labels) in enumerate(tqdm_test_loader):
                input = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                output = self.model(input)
                loss = torch.mean(criterion(batch_y, output), dim=-1)
                test_reconstitution_error.append(loss.detach().cpu())
        test_reconstitution_error = torch.cat(test_reconstitution_error, 0)


        print("anormly_ratio is {} ".format(self.args.anormly_ratio))
        thresh = np.percentile(test_reconstitution_error, 100 - self.args.anormly_ratio)

        # train_reconstitution_error = []
        # with torch.no_grad():
        #     tqdm_train_loader = tqdm(self.train_loader)
        #     for i, (batch_x, batch_y, labels) in enumerate(tqdm_train_loader):
        #         input = batch_x.float().to(self.device)
        #         batch_y = batch_y.float().to(self.device)
        #         output = self.model(input)
        #         loss = torch.mean(criterion(batch_y, output), dim=-1)
        #         train_reconstitution_error.append(loss.detach().cpu())
        # train_reconstitution_error = torch.cat(train_reconstitution_error, 0)
        # reconstitution_error = torch.cat((train_reconstitution_error, test_reconstitution_error), 0)
        # thresh = np.percentile(reconstitution_error, 100 - self.args.anormly_ratio)

        # (3) evaluation on the test set
        test_labels = []
        reconstitution_error = []
        tqdm_thre_loader = tqdm(self.thre_loader)
        for i, (batch_x, batch_y, labels) in enumerate(tqdm_thre_loader):
            input = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            output = self.model(input)
            loss = torch.mean(criterion(batch_y, output), dim=-1)
            reconstitution_error.append(loss.detach().cpu())
            test_labels.append(labels.detach().cpu())

        reconstitution_error = torch.cat(reconstitution_error, 0).numpy().reshape(-1)
        test_labels = torch.cat(test_labels, 0).numpy().reshape(-1)

        pred = (reconstitution_error > thresh).astype(int)  # 预测标签
        gt = test_labels.astype(int)  # 实际标签

        # scores_simple = combine_all_evaluation_scores(pred, gt, reconstitution_error)
        # for key, value in scores_simple.items():
        #     #            matrix.append(value)
        #     print('{0:21} : {1:0.4f}'.format(key, value))

        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)

        # detection adjustment: please see this issue for more information https://github.com/thuml/Anomaly-Transformer/issues/14
        anomaly_state = False
        for i in range(len(gt)):
            if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
                for j in range(i, len(gt)):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
            elif gt[i] == 0:
                anomaly_state = False
            if anomaly_state:
                pred[i] = 1

        pred = np.array(pred)
        gt = np.array(gt)
        #        print("pred: ", pred.shape)
        #        print("gt:   ", gt.shape)

        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred,average='binary')
        print(
            "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
                accuracy, precision,
                recall, f_score))
        self.model.train()
        return accuracy, precision, recall, f_score
