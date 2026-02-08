from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy, cal_f1, evaluate_calibration
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pdb
import tqdm

warnings.filterwarnings('ignore')


class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super(Exp_Classification, self).__init__(args)

    def _build_model(self):
        # model input depends on data
        print('Loading dataset!')
        train_data, train_loader = self._get_data(flag='TRAIN')
        vali_data, vali_loader = self._get_data(flag='VAL')
        test_data, test_loader = self._get_data(flag='TEST')
        
        self.train_data = train_data
        self.train_loader = train_loader

        self.vali_data = vali_data
        self.vali_loader = vali_loader

        self.test_data = test_data
        self.test_loader = test_loader

        self.args.seq_len = max(train_data.max_seq_len, test_data.max_seq_len)
        self.args.pred_len = 0
        self.args.enc_in = train_data.feature_df.shape[1]
        self.args.num_class = max(len(train_data.class_names), len(test_data.class_names))
        # model init
        model = self.model_dict[self.args.model].Model(self.args).float()

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Trainable parameters: ", total_params)

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.RAdam(self.model.parameters(), lr=self.args.learning_rate)
        
        return model_optim

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        
        return criterion

    def train(self):
        train_data = self.train_data
        train_loader = self.train_loader

        vali_data = self.vali_data
        vali_loader = self.vali_loader

        test_data = self.test_data
        test_loader = self.test_loader

        path = os.path.join(self.args.checkpoints, self.args.root_path.split("/")[-2], self.args.model)
        if not os.path.exists(path):
            os.makedirs(path)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        epoch_time_list = []

        for epoch in tqdm.tqdm(range(self.args.train_epochs)):
            train_loss = []
            preds = []
            trues = []

            self.model.train()
            start_time = time.time()
            # for i, (batch_x, label, padding_mask) in enumerate(train_loader):
            for i, batch_data in enumerate(train_loader):
                # Handle both cases: with RUL (3 values) and without (2 values)
                if len(batch_data) == 3:
                    batch_x, label, _ = batch_data  # Ignore RUL for classification
                else:
                    batch_x, label = batch_data
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                # padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                # outputs = self.model(batch_x, padding_mask, None, None)
                outputs = self.model(batch_x, None, None, None)
                loss = criterion(outputs, label.long().squeeze(-1))
                train_loss.append(loss.item())

                preds.append(outputs.detach().cpu())
                trues.append(label.detach().cpu())

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()
            
            epoch_time_list.append((time.time() - start_time))
            
            if len(preds) != 0:
                preds = torch.cat(preds, 0)
                trues = torch.cat(trues, 0)
                probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
                predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
                trues = trues.flatten().cpu().numpy()
                train_accuracy = cal_accuracy(predictions, trues)
            else:
                train_accuracy = 0.0
            
            train_loss = np.average(train_loss)
            val_loss, val_accuracy = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_accuracy = self.vali(test_data, test_loader, criterion)

            print("Epoch: {} | Train Loss: {:.4f}, Train Acc: {:.4f}, Vali Loss: {:.4f}, Vali Acc: {:.4f}, Test Loss: {:.4f}, Test Acc: {:.4f}".format(epoch + 1, train_loss, train_accuracy, val_loss, val_accuracy, test_loss, test_accuracy))

            self.wandb.log(
                {
                    "Epoch": epoch + 1,
                    "Loss/Train": train_loss,
                    "Loss/Val": val_loss,
                    "Loss/Test": test_loss,
                    "Acc/Train": train_accuracy,
                    "Acc/Val": val_accuracy,
                    "Acc/Test": test_accuracy,
                    "Time Per Epoch": epoch_time_list[-1]
                },
                commit=True
            )
            
            early_stopping(val_accuracy, self.model, path)
            if early_stopping.early_stop:
                print("Current epoch: {}\n", epoch + 1)
                print("Early stopping")
                break

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

        print("average training time per epoch: ", np.mean(np.array(epoch_time_list)))

        return self.model

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            # for i, (batch_x, label, padding_mask) in enumerate(vali_loader):
            for i, batch_data in tqdm.tqdm(enumerate(vali_loader)):
                # Handle both cases: with RUL (3 values) and without (2 values)
                if len(batch_data) == 3:
                    batch_x, label, _ = batch_data
                else:
                    batch_x, label = batch_data
                batch_x = batch_x.float().to(self.device)
                # padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                # outputs = self.model(batch_x, padding_mask, None, None)
                outputs = self.model(batch_x, None, None, None)

                loss = criterion(outputs, label.long().squeeze().reshape(-1,))
                total_loss.append(loss.item())

                preds.append(outputs.detach().cpu())
                trues.append(label)

        total_loss = np.average(total_loss)

        if len(preds) != 0:
            preds = torch.cat(preds, 0)
            trues = torch.cat(trues, 0)
            probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
            predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
            trues = trues.flatten().cpu().numpy()
            accuracy = cal_accuracy(predictions, trues)
        else:
            accuracy = 0.0

        return total_loss, accuracy
    
    def test(self, load_model=False):
        if load_model:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints, self.args.root_path.split("/")[-2], self.args.model, 'checkpoint.pth'), map_location=self.device))

        preds = []
        trues = []

        self.model.eval()
        with torch.no_grad():
            # for i, (batch_x, label, padding_mask) in enumerate(test_loader):
            for i, batch_data in tqdm.tqdm(enumerate(self.test_loader)):
                # Handle both cases: with RUL (3 values) and without (2 values)
                if len(batch_data) == 3:
                    batch_x, label, _ = batch_data
                else:
                    batch_x, label = batch_data
                batch_x = batch_x.float().to(self.device)
                # padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                # outputs = self.model(batch_x, padding_mask, None, None)
                outputs = self.model(batch_x, None, None, None)

                preds.append(outputs.detach())
                trues.append(label)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        print('test shape:', preds.shape, trues.shape)

        probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        
        accuracy = cal_accuracy(predictions, trues)
        f1_micro, f1_macro, f1_weighted = cal_f1(predictions, trues)
        nll, ece, brier = evaluate_calibration(torch.tensor(np.nan_to_num(probs.cpu(), nan=0.0), dtype=torch.float32), torch.tensor(trues, dtype=torch.float32))

        # result save
        folder_path = './results/' + self.args.root_path.split("/")[-2] + "/" + self.args.model + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        print('accuracy:{}, f1 micro: {}, ece: {}, nll: {}, brier: {}'.format(accuracy, f1_micro, ece, nll, brier))
        file_name='result_classification.txt'
        # if os.path.exists(os.path.join(folder_path, file_name)):
        #     os.remove(os.path.join(folder_path, file_name))
        args_dict = vars(self.args)
        with open(os.path.join(folder_path, file_name), 'w') as f:
            for key, value in args_dict.items():
                f.write(f"{key}:{value}\n")
            f.write("\n")
            f.write("\n")
            f.write('accuracy:{}'.format(accuracy))
            f.write('\n')
            f.write('f1_micro:{}'.format(f1_micro))
            f.write('\n')
            f.write('f1_macro:{}'.format(f1_macro))
            f.write('\n')
            f.write('f1_weighted:{}'.format(f1_weighted))
            f.write('\n')
            f.write('ece:{}'.format(ece))
            f.write('\n')
            f.write('nll:{}'.format(nll))
            f.write('\n')
            f.write('brier:{}'.format(brier))
        
        return
