import sys
import logging
import math
import time
import gc

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from scipy.special import logsumexp
from ema import EMA
from model import *
from utils import *
from diffusion_utils import *
# from crps_loss import crps_ensemble
from crps_loss_v2 import crps_ensemble

plt.style.use('ggplot')


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        self.num_timesteps = args.timesteps
 
        self.vis_step = config.diffusion.vis_step
        self.num_figs = config.diffusion.num_figs
        self.dataset_object = None
        self.tau = None  # precision fo test NLL computation
        
        logging.info("The number of samples is {}\n".format(config.testing.n_z_samples))

        # initial prediction model as guided condition
        self.cond_pred_model = None
        if config.diffusion.conditioning_signal == "OLS":
            # use training data to compute OLS beta parameter
            _, dataset = get_dataset(args, config, test_set=False)
            train_x = dataset[:, :-config.model.y_dim]
            # concat a column of 1's for the bias term
            train_x = torch.cat((torch.ones(train_x.shape[0], 1), train_x), dim=1)
            train_y = dataset[:, -config.model.y_dim:]
            # OLS beta hat
            xtx = train_x.T.mm(train_x)
            if torch.det(xtx).cpu().detach().numpy() == 0:
                xtx_inv = torch.linalg.pinv(xtx)
                logging.info("Invert the matrix with Moore-Penrose inverse...\n")
            else:
                xtx_inv = torch.inverse(xtx)
                logging.info("Invert the invertible square matrix...\n")
            # xtx_inv = torch.linalg.pinv(xtx)
            self.cond_pred_model = (xtx_inv.mm(train_x.T.mm(train_y))).to(self.device)
            # OLS_RMSE = (train_y - train_x.mm(self.cond_pred_model.cpu())).square().mean().sqrt()
            # logging.info("Training data RMSE with OLS is {:.8f}.\n".format(OLS_RMSE))
            del dataset
            del train_x
            del train_y
            gc.collect()
        elif config.diffusion.conditioning_signal == "NN":
            self.cond_pred_model = DeterministicFeedForwardNeuralNetwork(
                dim_in=config.model.x_dim, dim_out=config.testing.n_z_samples,
                hid_layers=config.diffusion.nonlinear_guidance.hid_layers,
                use_batchnorm=config.diffusion.nonlinear_guidance.use_batchnorm,
                negative_slope=config.diffusion.nonlinear_guidance.negative_slope,
                dropout_rate=config.diffusion.nonlinear_guidance.dropout_rate).to(self.device)
            self.aux_cost_function = crps_ensemble #nn.MSELoss()
            if config.model.ema:
                self.ema_helper = EMA(mu=config.model.ema_rate)
                self.ema_helper.register(self.cond_pred_model)
            else:
                self.ema_helper = None
        else:
            pass

    # Compute guiding prediction as diffusion condition
    def compute_guiding_prediction(self, x, method="OLS"):
        """
        Compute y_0_hat, to be used as the Gaussian mean at time step T.
        """
        if method == "OLS":
            x = torch.cat((torch.ones(x.shape[0], 1).to(x.device), x), dim=1)
            y_pred = x.mm(self.cond_pred_model)
        # elif method == "ZERO":
        #     y_pred = torch.zeros(x.shape[0], 1).to(x.device)
        elif method == "NN":
            y_pred = self.cond_pred_model(x)
        else:
            y_pred = None
        return y_pred#.unsqueeze(-1)

    def evaluate_guidance_model(self, dataset_object, dataset_loader):
        """
        Evaluate guidance model by reporting train or test set unnormalized y RMSE.
        """
        y_crps_list = []
        for xy_0 in dataset_loader:
            xy_0 = xy_0.to(self.device)
            x_batch = xy_0[:, :-self.config.model.y_dim]
            y_batch = xy_0[:, -self.config.model.y_dim:]
            y_batch_pred = self.compute_guiding_prediction(
                x_batch, method=self.config.diffusion.conditioning_signal)
            crps = self.aux_cost_function(y_batch.squeeze(-1), y_batch_pred)
            y_crps_list.append(crps)
        m_crps = torch.vstack(y_crps_list).mean()
        return m_crps.item()

    def evaluate_guidance_model_on_both_train_and_test_set(self,
                                                           train_set_object, train_loader,
                                                           test_set_object, test_loader):
        y_train_crps_aux_model = self.evaluate_guidance_model(train_set_object, train_loader)
        y_test_crps_aux_model = self.evaluate_guidance_model(test_set_object, test_loader)
        logging.info(("{} guidance model un-normalized y RMSE " +
                      "\n\tof the training set and of the test set are " +
                      "\n\t{:.4f} and {:.4f}, respectively.").format(
            self.config.diffusion.conditioning_signal, y_train_crps_aux_model, y_test_crps_aux_model))

    def nonlinear_guidance_model_train_step(self, x_batch, y_batch, aux_optimizer):
        """
        One optimization step of the non-linear guidance model that predicts y_0_hat.
        """
        y_batch_pred = self.cond_pred_model(x_batch)
        aux_cost = self.aux_cost_function(y_batch.squeeze(-1), y_batch_pred)
        # update non-linear guidance model
        aux_optimizer.zero_grad()
        aux_cost.backward()
        aux_optimizer.step()
        if self.config.model.ema:
            self.ema_helper.update(self.cond_pred_model)
        y_mse = (y_batch_pred.mean(1, keepdims=True) - y_batch) ** 2
        return aux_cost.cpu().item(), y_mse.mean().cpu().item()

    def nonlinear_guidance_model_train_loop_per_epoch(self, train_batch_loader, aux_optimizer, epoch):
        loss_list=[]
        rmse_list = []
        for xy_0 in train_batch_loader:
            xy_0 = xy_0.to(self.device)
            x_batch = xy_0[:, :-self.config.model.y_dim]
            y_batch = xy_0[:, -self.config.model.y_dim:]
            aux_loss, rmse = self.nonlinear_guidance_model_train_step(x_batch, y_batch, aux_optimizer)
            loss_list.append(aux_loss)
            rmse_list.append(rmse)
        # if epoch % self.config.diffusion.nonlinear_guidance.logging_interval == 0:
        #     logging.info(f"epoch: {epoch}, non-linear guidance model pre-training loss: {aux_loss}, {rmse}")
        
        return np.mean(loss_list), np.mean(rmse_list)
    
    def obtain_true_and_pred_y_t(self, cur_t, y_seq, y_T_mean, y_0):
        y_t_p_sample = y_seq[self.num_timesteps - cur_t].detach().cpu()
        y_t_true = q_sample(y_0, y_T_mean,
                            self.alphas_bar_sqrt, self.one_minus_alphas_bar_sqrt,
                            torch.tensor([cur_t - 1])).detach().cpu()
        return y_t_p_sample, y_t_true

    def compute_unnorm_y(self, cur_y, testing):
        if testing:
            y_mean = cur_y.cpu().reshape(-1, self.config.testing.n_z_samples).mean(1).reshape(-1, 1)
        else:
            y_mean = cur_y.cpu()
        if self.config.data.normalize_y:
            y_t_unnorm = self.dataset_object.scaler_y.inverse_transform(y_mean)
        else:
            y_t_unnorm = y_mean
        return y_t_unnorm

    def make_subplot_at_timestep_t(self, cur_t, cur_y, y_i, y_0, axs, ax_idx, prior=False, testing=True):
        # kl = (y_i - cur_y).square().mean()
        # kl_y0 = (y_0.cpu() - cur_y).square().mean()
        y_0_unnorm = self.compute_unnorm_y(y_0, testing)
        y_t_unnorm = self.compute_unnorm_y(cur_y, testing)
        kl_unnorm = ((y_0_unnorm - y_t_unnorm) ** 2).mean() ** 0.5
        axs[ax_idx].plot(cur_y, '.', label='pred', c='tab:blue')
        axs[ax_idx].plot(y_i, '.', label='true', c='tab:red')
        # axs[ax_idx].set_xlabel(
        #     'KL($q(y_t)||p(y_t)$)={:.2f}\nKL($q(y_0)||p(y_t)$)={:.2f}'.format(kl, kl_y0),
        #     fontsize=20)
        kl_unnorm_str = 'Unnormed RMSE: {:.2f}'.format(kl_unnorm)
        if prior:
            axs[ax_idx].set_title('$p({y}_\mathbf{prior})$',
                                  fontsize=23)
            axs[ax_idx].set_title('$p({y}_\mathbf{prior})$\n' + kl_unnorm_str,
                                  fontsize=23)
            axs[ax_idx].legend()
        else:
            axs[ax_idx].set_title('$p(\mathbf{y}_{' + str(cur_t) + '})$',
                                  fontsize=23)
            axs[ax_idx].set_title('$p(\mathbf{y}_{' + str(cur_t) + '})$\n' + kl_unnorm_str,
                                  fontsize=23)

    def train(self):
        args = self.args
        config = self.config
        # tb_logger = self.config.tb_logger
        # # first obtain test set for pre-trained model evaluation
        # logging.info("Test set info:")
        # test_set_object, test_set = get_dataset(args, config, test_set=True)
        # test_loader = data.DataLoader(
        #     test_set,
        #     batch_size=config.testing.batch_size,
        #     num_workers=config.data.num_workers,
        # )
        
        # obtain training set
        logging.info("Training set info:")
        dataset_object, dataset = get_dataset(args, config, test_set=False)
        self.dataset_object = dataset_object
        train_loader = data.DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )
        
        if config.diffusion.conditioning_signal == "NN":
            aux_optimizer = get_optimizer(self.config.aux_optim, self.cond_pred_model.parameters())
            
            best_model_state = self.cond_pred_model.state_dict()
            min_loss = np.inf
            counter = 0
            pretrain_start_time = time.time()
            print('--------------------------------')
            for epoch in range(config.diffusion.nonlinear_guidance.n_pretrain_epochs):
                train_loss, rmse = self.nonlinear_guidance_model_train_loop_per_epoch(train_loader, aux_optimizer, epoch)
                if train_loss < min_loss:
                    best_model_state = self.cond_pred_model.state_dict()
                    min_loss = train_loss
                    counter = 0
                else:
                    counter += 1
                
                if counter >= self.config.diffusion.nonlinear_guidance.patience:
                    break
            
                print(("Epoch {}; couter = {} , train_loss = {:.4f}, min loss = {:.4f}, rmse = {:.4f}").format(
                                epoch, counter, train_loss, min_loss, rmse))
                    
            
            pretrain_end_time = time.time()
            logging.info("Pre-training of non-linear guidance model took {:.4f} minutes.".format(
                (pretrain_end_time - pretrain_start_time) / 60))
            # logging.info("\nAfter pre-training:")
            # self.evaluate_guidance_model_on_both_train_and_test_set(dataset_object, train_loader,
            #                                                         test_set_object, test_loader)
            # save auxiliary model
            aux_states = [
                best_model_state,
                aux_optimizer.state_dict(),
            ]
            if self.config.model.ema:
                aux_states.append(self.ema_helper.state_dict())
            torch.save(aux_states, os.path.join(self.args.log_path, "aux_ckpt.pth"))
            

    def test(self):
        """
        Evaluate model on regression tasks on test set.
        """

        #####################################################################################################
        ########################## local functions within the class function scope ##########################
        def compute_prediction_SE(config, dataset_object, y_batch, generated_y, return_pred_mean=False):
            """
            generated_y: has a shape of (current_batch_size, n_z_samples, dim_y)
            """
            low, high = config.testing.trimmed_mean_range
            y_true = y_batch.cpu().detach().numpy()

            y_pred_mean = None  # to be used to compute RMSE
            if low == 50 and high == 50:
                y_pred_mean = np.median(generated_y, axis=1)  # use median of samples as the mean prediction
            else:  # compute trimmed mean (i.e. discarding certain parts of the samples at both ends)
                generated_y.sort(axis=1)
                low_idx = int(low / 100 * config.testing.n_z_samples)
                high_idx = int(high / 100 * config.testing.n_z_samples)
                y_pred_mean = (generated_y[:, low_idx:high_idx]).mean(axis=1)
            if dataset_object.normalize_y:
                y_true = dataset_object.scaler_y.inverse_transform(y_true).astype(np.float32)
                y_pred_mean = dataset_object.scaler_y.inverse_transform(y_pred_mean).astype(np.float32)
            if return_pred_mean:
                return y_pred_mean
            else:
                y_se = (y_pred_mean - y_true) ** 2
                return y_se

        def compute_true_coverage_by_gen_QI(config, dataset_object, all_true_y, all_generated_y, verbose=False):
            n_bins = config.testing.n_bins
            quantile_list = np.arange(n_bins + 1) * (100 / n_bins)
            #print(all_true_y.shape, all_generated_y.shape, dataset_object.test_n_samples,'--')
            # compute generated y quantiles
            y_pred_quantiles = np.percentile(all_generated_y.squeeze(), q=quantile_list, axis=1)
            y_true = all_true_y.T
            quantile_membership_array = ((y_true - y_pred_quantiles) > 0).astype(int)
            y_true_quantile_membership = quantile_membership_array.sum(axis=0)
            # y_true_quantile_bin_count = np.bincount(y_true_quantile_membership)
            y_true_quantile_bin_count = np.array(
                [(y_true_quantile_membership == v).sum() for v in np.arange(n_bins + 2)])
            #print(y_pred_quantiles.shape, quantile_membership_array.shape, y_true_quantile_bin_count.shape)
            if verbose:
                y_true_below_0, y_true_above_100 = y_true_quantile_bin_count[0], \
                                                   y_true_quantile_bin_count[-1]
                logging.info(("We have {} true y smaller than min of generated y, " + \
                              "and {} greater than max of generated y.").format(y_true_below_0, y_true_above_100))
            # combine true y falls outside of 0-100 gen y quantile to the first and last interval
            y_true_quantile_bin_count[1] += y_true_quantile_bin_count[0]
            y_true_quantile_bin_count[-2] += y_true_quantile_bin_count[-1]
            y_true_quantile_bin_count_ = y_true_quantile_bin_count[1:-1]
            # compute true y coverage ratio for each gen y quantile interval
            y_true_ratio_by_bin = y_true_quantile_bin_count_ / dataset_object.test_n_samples
            assert np.abs(
                np.sum(y_true_ratio_by_bin) - 1) < 1e-10, "Sum of quantile coverage ratios shall be 1!"
            qice_coverage_ratio = np.absolute(np.ones(n_bins) / n_bins - y_true_ratio_by_bin).mean()
            return y_true_ratio_by_bin, qice_coverage_ratio, y_true

        def compute_PICP(config, y_true, all_gen_y, return_CI=False):
            """
            Another coverage metric.
            """
            low, high = config.testing.PICP_range
            CI_y_pred = np.percentile(all_gen_y.squeeze(), q=[low, high], axis=1)
            # compute percentage of true y in the range of credible interval
            y_in_range = (y_true >= CI_y_pred[0]) & (y_true <= CI_y_pred[1])
            coverage = y_in_range.mean()
            if return_CI:
                return coverage, CI_y_pred, low, high
            else:
                return coverage, low, high

        def store_gen_y_at_step_t(config, current_batch_size, idx, y_tile_seq):
            """
            Store generated y from a mini-batch to the array of corresponding time step.
            """
            current_t = self.num_timesteps - idx
            #print(len(y_tile_seq), idx, 'xxx', self.num_timesteps)
            gen_y = y_tile_seq[idx].reshape(current_batch_size,
                                            config.testing.n_z_samples,
                                            config.model.y_dim).cpu().numpy()
            # directly modify the dict value by concat np.array instead of append np.array gen_y to list
            # reduces a huge amount of memory consumption
            if len(gen_y_by_batch_list[current_t]) == 0:
                gen_y_by_batch_list[current_t] = gen_y
            else:
                gen_y_by_batch_list[current_t] = np.concatenate([gen_y_by_batch_list[current_t], gen_y], axis=0)
            return gen_y

        def store_y_se_at_step_t(config, idx, dataset_object, y_batch, gen_y):
            current_t = self.num_timesteps - idx
            # compute sqaured error in each batch
            y_se = compute_prediction_SE(config=config, dataset_object=dataset_object,
                                         y_batch=y_batch, generated_y=gen_y)
            if len(y_se_by_batch_list[current_t]) == 0:
                y_se_by_batch_list[current_t] = y_se
            else:
                y_se_by_batch_list[current_t] = np.concatenate([y_se_by_batch_list[current_t], y_se], axis=0)

        def set_NLL_global_precision(test_var=True):
            if test_var:
                # compute test set sample variance
                if dataset_object.normalize_y:
                    y_test_unnorm = dataset_object.scaler_y.inverse_transform(dataset_object.y_test).astype(np.float32)
                else:
                    y_test_unnorm = dataset_object.y_test
                y_test_unnorm = y_test_unnorm if type(y_test_unnorm) is torch.Tensor \
                    else torch.from_numpy(y_test_unnorm)
                self.tau = 1 / (y_test_unnorm.var(unbiased=True).item())
            else:
                self.tau = 1

        def compute_batch_NLL(config, dataset_object, y_batch, generated_y):
            """
            generated_y: has a shape of (current_batch_size, n_z_samples, dim_y)

            NLL computation implementation from MC dropout repo
                https://github.com/yaringal/DropoutUncertaintyExps/blob/master/net/net.py,
                directly from MC Dropout paper Eq. (8).
            """
            y_true = y_batch.cpu().detach().numpy()
            if dataset_object.normalize_y:
                # unnormalize true y
                y_true = dataset_object.scaler_y.inverse_transform(y_true).astype(np.float32)
                # unnormalize generated y
                batch_size = generated_y.shape[0]
                #print(y_true.shape, generated_y.shape)
                generated_y = generated_y.reshape(batch_size * config.testing.n_z_samples, config.model.y_dim)
                #print(generated_y.shape, 'gene 1...')
                generated_y = dataset_object.scaler_y.inverse_transform(generated_y).astype(np.float32).reshape(
                    batch_size, config.testing.n_z_samples, config.model.y_dim)
            #print(generated_y.shape, 'gene 2...')
            generated_y = generated_y.swapaxes(0, 1)
            #print(generated_y.shape, 'gene 3...')
            # obtain precision value and compute test batch NLL
            #print(self.tau)
            if self.tau is not None:
                tau = self.tau
            else:
                gen_y_var = torch.from_numpy(generated_y).var(dim=0, unbiased=True).numpy()
                tau = 1 / gen_y_var
            nll = -(logsumexp(-0.5 * tau * (y_true[None] - generated_y) ** 2., 0)
                    - np.log(config.testing.n_z_samples)
                    - 0.5 * np.log(2 * np.pi) + 0.5 * np.log(tau))
            return nll

        def store_nll_at_step_t(config, idx, dataset_object, y_batch, gen_y):
            current_t = self.num_timesteps - idx
            # compute negative log-likelihood in each batch
            nll = compute_batch_NLL(config=config, dataset_object=dataset_object,
                                    y_batch=y_batch, generated_y=gen_y)
            if len(nll_by_batch_list[current_t]) == 0:
                nll_by_batch_list[current_t] = nll
            else:
                nll_by_batch_list[current_t] = np.concatenate([nll_by_batch_list[current_t], nll], axis=0)

        #####################################################################################################
        #####################################################################################################

        args = self.args
        config = self.config
        log_path = os.path.join(self.args.log_path)
        dataset_object, dataset = get_dataset(args, config, test_set=True)
        test_loader = data.DataLoader(
            dataset,
            batch_size=config.testing.batch_size,
            num_workers=config.data.num_workers,
        )
        self.dataset_object = dataset_object
        # set global prevision value for NLL computation if needed
        if args.nll_global_var:
            print('NLL computation ...')
            exit()
            set_NLL_global_precision(test_var=args.nll_test_var)



        # load auxiliary model
        if config.diffusion.conditioning_signal == "NN":
            aux_states = torch.load(os.path.join(log_path, "aux_ckpt.pth"),
                                    map_location=self.device)
            self.cond_pred_model.load_state_dict(aux_states[0], strict=True)
            self.cond_pred_model.train()

        
        logging.info(("\nWe pick t={} to compute y mean metric RMSE, " +
                        "and t={} to compute true y coverage metric QICE and PICP.\n").format(
            config.testing.mean_t, config.testing.coverage_t))

        with torch.no_grad():
            true_x_by_batch_list = []
            true_x_tile_by_batch_list = []
            true_y_by_batch_list = []
            gen_y_by_batch_list = [[] for _ in range(self.num_timesteps + 1)]
            y_se_by_batch_list = [[] for _ in range(self.num_timesteps + 1)]
            nll_by_batch_list = [[] for _ in range(self.num_timesteps + 1)]
            
            
                
            for step, xy_batch in enumerate(test_loader):
                # minibatch_start = time.time()
                xy_0 = xy_batch.to(self.device)
                current_batch_size = xy_0.shape[0]
                x_batch = xy_0[:, :-config.model.y_dim]
                y_batch = xy_0[:, -config.model.y_dim:]
                # compute y_0_hat as the initial prediction to guide the reverse diffusion process
                y_tile_seq=[]
                for _idx in range(self.num_timesteps):
                    y_0_hat_batch = self.compute_guiding_prediction(x_batch,
                                                                    method=config.diffusion.conditioning_signal)
                    y_tile_seq.append(y_0_hat_batch.view(-1,1))

                true_y_by_batch_list.append(y_batch.cpu().numpy())
                if config.testing.make_plot and config.data.dataset != "uci":
                    true_x_by_batch_list.append(x_batch.cpu().numpy())
    
                for idx in range(self.num_timesteps):
                    gen_y = store_gen_y_at_step_t(config=config, current_batch_size=current_batch_size,
                                                    idx=idx, y_tile_seq=y_tile_seq)
                    store_y_se_at_step_t(config=config, idx=idx,
                                            dataset_object=dataset_object,
                                            y_batch=y_batch, gen_y=gen_y)
                    store_nll_at_step_t(config=config, idx=idx,
                                        dataset_object=dataset_object,
                                        y_batch=y_batch, gen_y=gen_y)

                

        ################## compute metrics on test set ##################
        all_true_y = np.concatenate(true_y_by_batch_list, axis=0)
        if config.testing.make_plot and config.data.dataset != "uci":
            all_true_x = np.concatenate(true_x_by_batch_list, axis=0)
        if config.testing.plot_gen:
            all_true_x_tile = np.concatenate(true_x_tile_by_batch_list, axis=0)
        y_rmse_all_steps_list = []
        y_qice_all_steps_list = []
        y_picp_all_steps_list = []
        y_nll_all_steps_list = []

        for idx in range(self.num_timesteps):
            current_t = self.num_timesteps - idx
            # compute RMSE
            y_rmse = np.sqrt(np.mean(y_se_by_batch_list[current_t]))
            y_rmse_all_steps_list.append(y_rmse)
            # compute QICE
            all_gen_y = gen_y_by_batch_list[current_t]
            y_true_ratio_by_bin, qice_coverage_ratio, y_true = compute_true_coverage_by_gen_QI(
                config=config, dataset_object=dataset_object,
                all_true_y=all_true_y, all_generated_y=all_gen_y, verbose=False)
            y_qice_all_steps_list.append(qice_coverage_ratio)
            # compute PICP
            coverage, _, _ = compute_PICP(config=config, y_true=y_true, all_gen_y=all_gen_y)
            y_picp_all_steps_list.append(coverage)
            # print(y_true_ratio_by_bin.shape, qice_coverage_ratio.shape, y_true.shape, coverage.shape)
            # compute NLL
            y_nll = np.mean(nll_by_batch_list[current_t])
            y_nll_all_steps_list.append(y_nll)
        
        # #print(config.testing.mean_t, y_se_by_batch_list[config.testing.mean_t].shape, len(y_se_by_batch_list))
        # # compute RMSE
        # y_rmse = np.sqrt(np.mean(y_se_by_batch_list[config.testing.mean_t]))
        # y_rmse_all_steps_list.append(y_rmse)
        # logging.info(f"y RMSE at all steps: {np.mean(y_rmse_all_steps_list), len(y_rmse_all_steps_list)}.\n")
       
        # # compute QICE -- a cover metric
        # all_gen_y = gen_y_by_batch_list[config.testing.coverage_t]
        # y_true_ratio_by_bin, qice_coverage_ratio, y_true = compute_true_coverage_by_gen_QI(
        #     config=config, dataset_object=dataset_object,
        #     all_true_y=all_true_y, all_generated_y=all_gen_y, verbose=True)
        # y_qice_all_steps_list.append(qice_coverage_ratio)
        # logging.info("\nWe generated {} y's given each x.".format(config.testing.n_z_samples))
        # logging.info(("\nRMSE between true mean y and the mean of generated y given each x is " +
        #                 "{:.8f};\nQICE between true y coverage ratio by each generated y " +
        #                 "quantile interval and optimal ratio is {:.8f}.").format(y_rmse, qice_coverage_ratio))
        # # compute PICP -- another coverage metric
        # coverage, low, high = compute_PICP(config=config, y_true=y_true, all_gen_y=all_gen_y)
        # y_picp_all_steps_list.append(coverage)
        # logging.info(("There are {:.4f}% of true test y in the range of " +
        #                 "the computed {:.0f}% credible interval.").format(100 * coverage, high - low))
        # # compute NLL
        # y_nll = np.mean(nll_by_batch_list[config.testing.nll_t])
        # y_nll_all_steps_list.append(y_nll)
        # logging.info("\nNegative Log-Likelihood on test set is {:.8f}.".format(y_nll))
        logging.info(f"y RMSE at all steps: {np.mean(y_rmse_all_steps_list), len(y_rmse_all_steps_list)}.\n")
        logging.info(f"y QICE at all steps: {np.mean(y_qice_all_steps_list), len(y_qice_all_steps_list)}.\n")
        logging.info(f"y PICP at all steps: {np.mean(y_picp_all_steps_list), len(y_picp_all_steps_list)}.\n\n")
        logging.info(f"y NLL at all steps: {np.mean(y_nll_all_steps_list), len(y_nll_all_steps_list)}.\n\n")
        
        # make plots for true vs. generated distribution comparison
        if config.testing.make_plot:
            assert config.data.dataset != "uci"
            all_gen_y = gen_y_by_batch_list[config.testing.vis_t]
            # compute QICE
            y_true_ratio_by_bin, qice_coverage_ratio, y_true = compute_true_coverage_by_gen_QI(
                config=config, dataset_object=dataset_object,
                all_true_y=all_true_y, all_generated_y=all_gen_y,
                verbose=False)
            logging.info(("\nQICE between true y coverage ratio by each generated y " +
                          "quantile interval and optimal ratio is {:.8f}.").format(
                qice_coverage_ratio))
            # compute PICP and RMSE
            if config.data.no_multimodality:
                if not config.data.inverse_xy:
                    coverage, CI_y_pred, low, high = compute_PICP(config=config,
                                                                  y_true=y_true,
                                                                  all_gen_y=all_gen_y,
                                                                  return_CI=True)
            # compute mean predicted y given each x
            y_pred_mean = compute_prediction_SE(config=config, dataset_object=dataset_object,
                                                y_batch=torch.from_numpy(all_true_y),
                                                generated_y=all_gen_y,
                                                return_pred_mean=True)

            # create plot
            logging.info("\nNow we start making the plot...")
            n_fig_rows, n_fig_cols = 3, 1
            fig, (ax1, ax2, ax3) = plt.subplots(n_fig_rows, n_fig_cols, clear=True)
            fig.set_figheight(config.testing.fig_size[0])
            fig.set_figwidth(config.testing.fig_size[1])
            # make individual plot to be organized into grid plot for the paper
            fig_1, ax_1 = plt.subplots(1, 1, clear=True)
            fig_1.set_figheight(config.testing.one_fig_size[0])
            fig_1.set_figwidth(config.testing.one_fig_size[1])
            fig_2, ax_2 = plt.subplots(1, 1, clear=True)
            fig_2.set_figheight(config.testing.one_fig_size[0])
            fig_2.set_figwidth(config.testing.one_fig_size[1])
            fig_3, ax_3 = plt.subplots(1, 1, clear=True)
            fig_3.set_figheight(config.testing.one_fig_size[0])
            fig_3.set_figwidth(config.testing.one_fig_size[1])
            # un-normalize y to its original scale for plotting
            if dataset_object.normalize_y:
                all_true_y = dataset_object.scaler_y.inverse_transform(all_true_y).astype(np.float32)
                all_gen_y = all_gen_y.reshape(dataset_object.test_n_samples * config.testing.n_z_samples,
                                              config.model.y_dim)
                all_gen_y = dataset_object.scaler_y.inverse_transform(all_gen_y).astype(np.float32).reshape(
                    dataset_object.test_n_samples,
                    config.testing.n_z_samples,
                    config.model.y_dim)

            ################## make first plot (only for toy data with 1D x) ##################
            if config.testing.squared_plot:
                ax1.set(aspect='equal', adjustable='box')
            if config.data.inverse_xy:
                x_noiseless_mean = compute_y_noiseless_mean(dataset_object,
                                                            torch.from_numpy(all_true_y),
                                                            config.data.true_function)
                if config.model.y_dim == 1:
                    sorted_idx_ = np.argsort(all_true_y, axis=0).squeeze()
                    ax1.plot(x_noiseless_mean[sorted_idx_], all_true_y[sorted_idx_],
                             c='orange', alpha=1, label='true-noiseless')
            if config.data.no_multimodality:
                logging.info("\nThe toy dataset doesn't contain multimodality.")
                if not config.data.inverse_xy:
                    y_rmse = np.sqrt(np.mean((y_pred_mean - all_true_y) ** 2))
                    logging.info(("\nRMSE between true y and the mean of generated y given each x is " +
                                  "{:.8f}.").format(y_rmse))
                    # obtain noiseless mean with ground truth data generation function
                    y_noiseless_mean = compute_y_noiseless_mean(dataset_object,
                                                                torch.from_numpy(all_true_x),
                                                                config.data.true_function)
                    logging.info(("\nRMSE between true expected y and the mean of generated y given each x is " +
                                  "{:.8f}.").format(np.sqrt(np.mean((y_pred_mean - y_noiseless_mean) ** 2))))
            n_true_x_for_plot_scale = 2
            if config.testing.plot_true:
                n_total_true = all_true_x.shape[0]
                true_sampled_idx = np.random.choice(
                    np.arange(n_total_true), size=n_total_true // n_true_x_for_plot_scale, replace=False)
                ax1.scatter(all_true_x[true_sampled_idx], all_true_y[true_sampled_idx],
                            s=2, c='r', marker="o", alpha=0.5, label='true')
                ax_1.scatter(all_true_x[true_sampled_idx], all_true_y[true_sampled_idx],
                             s=2, c='r', marker="o", alpha=0.5, label='true')
            if config.testing.plot_gen:
                # if sample the same idx for each test x, the sampled generated y tend to
                # follow a smooth trace instead of scattered randomly
                samp_idx = np.random.randint(low=0, high=config.testing.n_z_samples,
                                             size=(dataset_object.test_n_samples,
                                                   n_samples_gen_y_for_plot,
                                                   config.model.y_dim))
                all_gen_y_ = np.take_along_axis(all_gen_y, indices=samp_idx, axis=1)
                if len(all_gen_y_.shape) == 3:
                    all_gen_y_ = all_gen_y_.reshape(dataset_object.test_n_samples * n_samples_gen_y_for_plot,
                                                    config.model.y_dim)
                n_total_samples = all_true_x_tile.shape[0]
                gen_sampled_idx = np.random.choice(
                    np.arange(n_total_samples),
                    size=n_total_samples // (n_true_x_for_plot_scale * n_samples_gen_y_for_plot), replace=False)
                ax1.scatter(all_true_x_tile[gen_sampled_idx], all_gen_y_[gen_sampled_idx],
                            s=2, c='b', marker="^", alpha=0.5, label='generated')
                ax_1.scatter(all_true_x_tile[gen_sampled_idx], all_gen_y_[gen_sampled_idx],
                             s=2, c='b', marker="^", alpha=0.5, label='generated')
            if config.data.no_multimodality:
                if not config.data.inverse_xy:
                    logging.info("\nWe generated {} y's given each x.".format(config.testing.n_z_samples))
                    logging.info(("There are {:.4f}% of true test y in the range of " +
                                  "the computed {:.0f}% credible interval.").format(
                        100 * coverage, high - low))
                    # make sure input to x argument is properly sorted
                    if config.data.normalize_y:
                        CI_y_pred_lower = dataset_object.scaler_y.inverse_transform(
                            CI_y_pred[0].reshape(-1, 1)).flatten()
                        CI_y_pred_higher = dataset_object.scaler_y.inverse_transform(
                            CI_y_pred[1].reshape(-1, 1)).flatten()
                    else:
                        CI_y_pred_lower = CI_y_pred[0]
                        CI_y_pred_higher = CI_y_pred[1]
                    ax1.fill_between(x=all_true_x.squeeze(),
                                     y1=CI_y_pred_lower,
                                     y2=CI_y_pred_higher,
                                     facecolor='grey',
                                     alpha=0.6)
                    ax_1.fill_between(x=all_true_x.squeeze(),
                                      y1=CI_y_pred_lower,
                                      y2=CI_y_pred_higher,
                                      facecolor='grey',
                                      alpha=0.6)
            
            ax_1.legend(loc='best')
            ax1.legend(loc='best')
            ax1.set_title('True vs. Generated y Given x')
            ax_1.legend(loc='best')
            ax_1.set_xlabel('$x$', fontsize=10)
            ax_1.set_ylabel('$y$', fontsize=10)
            fig_1.savefig(os.path.join(args.im_path, 'gen_vs_true_scatter.png'), dpi=1200, bbox_inches='tight')

            ################## make second plot ##################
            n_bins = config.testing.n_bins
            optimal_ratio = 1 / n_bins
            all_bins = np.arange(n_bins) + 1

            ax2.bar(all_bins, y_true_ratio_by_bin, label='quantile coverage')
            ax2.hlines(optimal_ratio, all_bins[0] - 1, all_bins[-1] + 1, colors='r', label='optimal ratio')
            ax2.set_xticks(all_bins[::2])
            ax2.set_xlim(all_bins[0] - 1, all_bins[-1] + 1)
            ax2.set_ylim([0, 0.5])
            ax2.set_title('Ratio of True y \nin Each Quantile Interval of Generated y')
            ax2.legend(loc='best')
            ax_2.bar(all_bins, y_true_ratio_by_bin, label='quantile coverage')
            ax_2.hlines(optimal_ratio, all_bins[0] - 1, all_bins[-1] + 1, colors='r', label='optimal ratio')
            ax_2.set_xticks(all_bins[::2])
            ax_2.set_xlim(all_bins[0] - 1, all_bins[-1] + 1)
            ax_2.set_ylim([0, 0.5])
            ax_2.legend(loc='best')
            fig_2.savefig(os.path.join(args.im_path, 'quantile_interval_coverage.png'), dpi=1200)

            ################## make third plot ##################
            bins_norm_on_y = all_bins * y_true_ratio_by_bin / optimal_ratio
            ax3.set(aspect='equal', adjustable='box')
            ax3.scatter(all_bins, bins_norm_on_y)
            ax3.plot([-1, all_bins[-1] + 1], [-1, all_bins[-1] + 1], c='orange')
            ax3.set_xticks(all_bins[::2])
            ax3.set_yticks(all_bins[::2])
            ax3.set_xlim([-1, all_bins[-1] + 1])
            ax3.set_ylim([-1, all_bins[-1] + 1])
            ax3.set_title('Ratio of True y vs. Optimal Ratio \nin Each Quantile Interval of Generated y')
            ax_3.set(aspect='equal', adjustable='box')
            ax_3.scatter(all_bins, bins_norm_on_y)
            ax_3.plot([-1, all_bins[-1] + 1], [-1, all_bins[-1] + 1], c='orange')
            ax_3.set_xticks(all_bins[::2])
            ax_3.set_yticks(all_bins[::2])
            ax_3.set_xlim([-1, all_bins[-1] + 1])
            ax_3.set_ylim([-1, all_bins[-1] + 1])
            fig_3.savefig(os.path.join(args.im_path, 'quantile_interval_coverage_true_vs_optimal.png'), dpi=1200)

            fig.tight_layout()
            fig.savefig(os.path.join(args.im_path, 'gen_vs_true_distribution_vis.png'), dpi=1200)

        # clear the memory
        plt.close('all')
        del true_y_by_batch_list
        if config.testing.make_plot and config.data.dataset != "uci":
            del all_true_x
        if config.testing.plot_gen:
            del all_true_x_tile
        del gen_y_by_batch_list
        del y_se_by_batch_list
        gc.collect()

        return y_rmse_all_steps_list, y_qice_all_steps_list, y_picp_all_steps_list, y_nll_all_steps_list
