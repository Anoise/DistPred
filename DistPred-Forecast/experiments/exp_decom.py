from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics2 import metric, compute_PICP, compute_true_coverage_by_gen_QI
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from tqdm import tqdm
# from utils.crps_loss import crps_ensemble
from utils.crps_loss_v2 import crps_ensemble
warnings.filterwarnings('ignore')



class Exp_Decom(Exp_Basic):
    def __init__(self, args):
        super(Exp_Decom, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        #criterion = nn.L1Loss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        total_crps = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()           

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                # encoder - decoder
                
             
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                # outputs = outputs[:,:, f_dim:]
                loss = criterion(outputs.mean(-1), batch_y.transpose(-1,1))
                crps = crps_ensemble(batch_y.transpose(-1,1), outputs)

                total_loss.append(loss.item())
                total_crps.append(crps.item())
        mean_loss = np.average(total_loss)
        mean_crps = np.average(total_crps)
        self.model.train()
        return mean_loss, mean_crps

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
            
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                # encoder - decoder
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs[:, :, f_dim:]
                loss = crps_ensemble(batch_y.transpose(-1,1), outputs)
                    
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss, vali_crps = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_crps = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Vali CRPS: {4:.7f}  Test Loss: {5:.7f} Test CRPS: {6:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, vali_crps, test_loss, test_crps))
            
            early_stopping(vali_crps, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

            # get_cka(self.args, setting, self.model, train_loader, self.device, epoch)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputs = []
        naive_preds = []
        crps_loss = []
        qice_list= []
        picp_list = []
        
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        qice_len = 0
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(test_loader)):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                # encoder - decoder
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                # outputs = outputs[:, :, f_dim:]
                crps = crps_ensemble(batch_y.transpose(-1,1), outputs)
                crps_loss.append(crps.item())
                
                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()
                _input = batch_x.detach().cpu().numpy()
                naive_pred = batch_x[:,-1:,:].repeat(1, true.shape[1], 1).detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    outputs = test_data.inverse_transform(outputs)
                    batch_y = test_data.inverse_transform(batch_y)

                if i < qice_len:
                    qice, _ = compute_true_coverage_by_gen_QI(true, pred)
                    coverage, _, _ = compute_PICP(true, pred)
                    qice_list.append(qice)
                    picp_list.append(coverage)

                preds.append(pred.mean(-1))
                trues.append(true)
                inputs.append(_input)
                naive_preds.append(naive_pred)
                
                #print('batch = ', i, crps, qice, coverage)
                #print(pred.shape, true.shape, naive_pred.shape)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    output = pred.mean(-1).transpose(0,-1,1)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], output[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        inputs = np.array(inputs)
        naive_preds = np.array(naive_preds)
        print('test shape:', preds.shape, trues.shape, naive_preds.shape)

        preds = preds.squeeze(1)
        trues = trues.squeeze(1).transpose(0,-1,1)
        inputs = inputs.squeeze(1).transpose(0,-1,1)
        naive_preds= naive_preds.squeeze(1).transpose(0,-1,1)
        print('test shape:', preds.shape, trues.shape, inputs.shape, naive_preds.shape)
        # result save
        # folder_path = './results/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)

        # np.save(folder_path + 'preds.npy', preds)
        # np.save(folder_path + 'trues.npy', trues)
        # np.save(folder_path + 'inputs.npy', inputs)
        
        # crps = crps_ensemble(torch.tensor(trues), torch.tensor(preds))
        # qice_coverage_ratio, _ = compute_true_coverage_by_gen_QI(trues, preds)
        # coverage, low, high = compute_PICP(trues, preds)
        
        print(np.mean(crps_loss), np.mean(qice_list)*100, np.mean(picp_list)*100, 'xxxxxx')
        
        mae, mse, rmse, rmdspe, mape, _smape, _mase, q50, q25, q75 = metric(preds, trues, naive_preds)
        
        print('crps:{:.3f}, qice:{:.3f}, picp:{:.3f}, mse:{}, mae:{}, rmse:{}, rmdspe:{}, mape:{}, smape:{}, mase:{}, Q50:{}, Q25:{}, Q75:{}'.format(np.mean(crps_loss), np.mean(qice_list)*100, np.mean(picp_list)*100, mse, mae, rmse, rmdspe, mape, _smape, _mase, q50, q25, q75))

        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('crps:{:.3f}, qice:{:.3f}, picp:{:.3f}, mse:{}, mae:{}, rmse:{}, rmdspe:{}, mape:{}, smape:{}, mase:{}, Q50:{}, Q25:{}, Q75:{}'.format(np.mean(crps_loss), np.mean(qice_list)*100, np.mean(picp_list)*100, mse, mae, rmse, rmdspe, mape, _smape, _mase, q50, q25, q75))
        f.write('\n')
        f.write('\n')
        f.close()

   

        return
