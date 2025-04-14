import h5py
import time
import torch
import scipy.io as sio
from pathlib import Path
from datasets.data import create_loaders
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from utils.util import AverageMeter
from models import GuidedNet, RIJAN, RIJAN_P, MSL_RIJAN_P
from models import UncertaintyLoss, L1Loss, CharFreqLoss
import numpy as np
from collections import defaultdict


@torch.no_grad()
def split_patch_eval(input_ms, input_pan, model, i_s=125):
    number_h = input_ms.shape[2] // i_s
    number_w = input_ms.shape[3] // i_s
    scale = input_pan.shape[2] // input_ms.shape[2]
    pan_s = int(i_s * scale)

    ms_sequence, pan_sequence = [], []
    for h in range(number_h):
        for w in range(number_w):
            ms_patch = input_ms[:, :, (h * i_s):(h * i_s + i_s), (w * i_s):(w * i_s + i_s)]
            pan_patch = input_pan[:, :, (h * pan_s):(h * pan_s + pan_s), (w * pan_s):(w * pan_s + pan_s)]
            ms_sequence.append(ms_patch)
            pan_sequence.append(pan_patch)
    predict_sequence = []
    for ms, pan in zip(ms_sequence, pan_sequence):
        predicts = model(ms, None, pan)
        predict_sequence.append(predicts[-1].cpu().detach())

    count = 0
    result = torch.ones([1, input_ms.shape[1], input_pan.shape[2], input_pan.shape[3]])
    for h in range(number_h):
        for w in range(number_w):
            result[:, :, h * pan_s:(h + 1) * pan_s, w * pan_s:(w + 1) * pan_s] = predict_sequence[count]
            count += 1
    return result


def create_model(config, device):
    model_name = config.model_name
    dataset_name = config.dataset_name
    if dataset_name in ('GF2', 'QB'):
        in_channels, out_channels = (5, 4)
    elif dataset_name in ['CAVE_x4', 'CAVE_x8', 'HARVARD_x4', 'HARVARD_x8']:
        if model_name in ['LRTN', 'DCT']:
            in_channels, out_channels = (31, 31)
        else:
            in_channels, out_channels = (34, 31)
    else:
        in_channels, out_channels = (9, 8)

    ms_cnum = out_channels
    pan_cnum = in_channels - out_channels
    max_step = config.max_step
    pool_size, gate_channel = config.pool_size, config.gate_channel
    model = None
    if model_name == 'RIJAN':
        model = RIJAN(in_channels, out_channels, 64).to(device)
    elif model_name == 'RIJAN_P':
        model = RIJAN_P(in_ms_cnum=ms_cnum, in_pan_cnum=pan_cnum).to(device)
    elif model_name == 'RIJAN_PA':
        model = RIJAN_P(in_ms_cnum=ms_cnum, in_pan_cnum=pan_cnum, adaptive=True).to(device)
    elif model_name == 'RIJAN_PD':
        model = RIJAN_P(in_ms_cnum=ms_cnum, in_pan_cnum=pan_cnum, adaptive=False, dynamic=True,
                        # dw=True,
                        pool_size=pool_size, gate_channel=gate_channel).to(device)
    elif model_name == 'MSL_RIJAN_PD':
        model = MSL_RIJAN_P(in_ms_cnum=ms_cnum, in_pan_cnum=pan_cnum, adaptive=False, dynamic=config.dynamic,
                            recursive=config.recursive, gate_method=config.gate_method, gate_input=config.gate_input,
                            retimes=config.MRD,
                            # dw=True,
                            # hidden_channels=64 if out_channels == 31 else 32,
                            # hidden_channels=128 if out_channels == 31 else 32,
                            pool_size=pool_size, gate_channel=gate_channel).to(device)
    elif model_name == 'RIJAN_PAD':
        model = RIJAN_P(in_ms_cnum=ms_cnum, in_pan_cnum=pan_cnum, adaptive=True, dynamic=True,
                        pool_size=pool_size, gate_channel=gate_channel).to(device)
    else:
        assert f'{model_name} not supported now.'
    return model


class Trainer:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.writer = None
        dataset_name = config.dataset_name
        model_name = config.model_name
        self.debug = config.debug
        if not self.debug:
            run_time = logger.handlers[0].baseFilename.split('/')[-1][:-4]
            self.run_time = run_time
            weights_save_path = Path(self.config.weights_path) / dataset_name / run_time
            weights_save_path.mkdir(exist_ok=True, parents=True)
            self.weights_save_path = weights_save_path
            tb_log_path = Path(self.config.tb_log_path) / run_time
            tb_log_path.mkdir(exist_ok=True, parents=True)
            self.writer = SummaryWriter(str(tb_log_path))

        self.epoch_num = config.epoch_num
        self.train_loader, self.val_loader = create_loaders(config)
        base_lr = float(config.base_lr)
        device = torch.device('cuda:0')
        self.device = device
        self.model = create_model(config, device)
        self.model_name = model_name

        self.criterion = nn.L1Loss().to(device)
        self.f_loss = CharFreqLoss().to(device)
        self.l1_loss = L1Loss().to(device)
        self.U_loss = UncertaintyLoss().to(device)

        # self.criterion = dual_domain_loss
        if model_name == 'DCT':
            self.optimizer = optim.Adam(self.model.parameters(), lr=base_lr)
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[100, 150, 175, 190, 195],
                                                                  gamma=0.5)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=base_lr, betas=(0.9, 0.999))
            step_size, gamma = int(config.step_size), float(config.gamma)
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        return

    def train_all(self):
        print('Start training...')
        epoch_time = AverageMeter()
        end = time.time()

        ckpt = self.config.save_epoch
        model, optimizer, device = self.model, self.optimizer, self.device
        # try:
        #     import torch._dynamo as dynamo
        #     torch._dynamo.config.verbose = True
        #     torch.backends.cudnn.benchmark = True
        #     model = torch.compile(model, mode="max-autotune", fullgraph=False)
        #     print("Model compiled set")
        # except Exception as err:
        #     print(f"Model compile not supported: {err}"http://127.0.0.1:2017/)

        for epoch in range(self.epoch_num):
            epoch += 1
            epoch_train_loss = []
            # main experiments in the paper
            offset = max(0.7 * (1 - epoch / (self.epoch_num - 500)), 0.1)
            # ablation experiments in the paper
            # offset = max(0.7 * (1 - epoch / (self.epoch_num - 250)), 0.1)
            # PanVIT_UAG
            # offset = max(0.7 * (1 - epoch / (self.epoch_num - 250)), 0.3)
            # offset = max(0.7 * (1 - epoch / (self.epoch_num - 500)), 0.2)
            # offset = max(0.7 * (1 - epoch / (self.epoch_num - 500)), 0.05)
            # offset = max(0.5 * (1 - epoch / (self.epoch_num - 800)), 0.05)
            # offset = max(0.5 * (1 - epoch / (self.epoch_num - 800)), 0.04)
            # offset = 1
            if self.config.dynamic:
                if self.model_name == 'PanVIT_UAG':
                    for block in model.blocks:
                        block.gate.offset = torch.tensor([[offset, 0]])
                    self.logger.info(f'Current offset {model.blocks[0].gate.offset}')
                elif 'RIJAN' in self.model_name:
                    for rijab in model.rijabs:
                        rijab.gate.offset = torch.tensor([[offset, 0]])
                    self.logger.info(f'Current offset {model.rijabs[0].gate.offset}')
                else:
                    pass

            model.train()
            for iteration, batch in enumerate(self.train_loader, 1):
                gt, lms, ms, pan = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device)
                optimizer.zero_grad()  # fixed

                if self.model_name == 'ARN':
                    AU, EU, hrms1, hrms2 = model(ms, lms, pan)
                    l1_loss, U_loss = self.l1_loss, self.U_loss
                    # result loss
                    loss = l1_loss(hrms2, gt, 1) * 0.1

                    # uncertainty-related loss
                    t_gt = torch.nn.functional.interpolate(gt, hrms1[0].shape[-2:], mode='bicubic').detach()
                    loss = loss + l1_loss(hrms1[0], t_gt, 1) * 0.4 + U_loss(hrms1[0], t_gt, AU[0]) * 0.04
                    t_gt = torch.nn.functional.interpolate(gt, hrms1[1].shape[-2:], mode='bicubic').detach()
                    loss = loss + l1_loss(hrms1[1], t_gt, 1) * 0.2 + U_loss(hrms1[1], t_gt, AU[1]) * 0.02
                    loss = loss + l1_loss(hrms1[2], gt, 1) * 0.1 + U_loss(hrms1[2], gt, AU[2]) * 0.01

                    # focus loss
                    U3 = 0.5 * AU[2] + 0.5 * EU[2]
                    loss = loss + (l1_loss(hrms2, gt, U3)) * 0.01
                elif self.model_name == 'RIJAN':
                    out = model(lms, pan)
                    loss = self.criterion(out, gt)
                elif self.model_name == 'MSL_RIJAN_PD':
                    # out, mid_outs = model(ms, lms, pan)
                    outs = model(ms, lms, pan)

                    t_loss = 0
                    for bi in range(len(outs) - 1):
                        t_gt = torch.nn.functional.interpolate(gt, outs[bi].shape[-2:], mode='bicubic').detach()
                        # tmp = self.criterion(outs[bi], t_gt) * 0.01
                        tmp = self.f_loss(outs[bi], t_gt) * 0.01
                        # tmp = (self.f_loss(outs[bi], t_gt) + self.grad_loss(outs[bi], t_gt) * 0.1) * 0.01
                        t_loss += tmp

                    # loss_final = self.criterion(outs[-1], gt)
                    loss_final = self.f_loss(outs[-1], gt)
                    # loss_final = self.f_loss(outs[-1], gt) + self.grad_loss(outs[-1], gt) * 0.1
                    loss = t_loss + loss_final
                elif self.model_name == 'PanVIT_UAG':
                    outs = model(pan, lms)
                    t_loss = 0
                    for bi in range(len(outs) - 1):
                        t_gt = torch.nn.functional.interpolate(gt, outs[bi].shape[-2:], mode='bicubic').detach()
                        # tmp = self.criterion(outs[bi], t_gt) * 0.01
                        tmp = self.f_loss(outs[bi], t_gt) * 0.01
                        # tmp = (self.f_loss(outs[bi], t_gt) + self.grad_loss(outs[bi], t_gt) * 0.1) * 0.01
                        t_loss += tmp

                    # loss_final = self.criterion(outs[-1], gt)
                    loss_final = self.f_loss(outs[-1], gt)
                    loss = t_loss + loss_final
                    # loss = self.criterion(outs[-1], gt)
                elif self.model_name == 'PanVIT':
                    out = model(pan, lms)
                    loss = self.criterion(out, gt)
                elif 'RIJAN_P' in self.model_name:
                    out = model(ms, lms, pan)
                    loss = self.criterion(out, gt)
                elif 'GuidedNet' in self.model_name:
                    outs = model(ms, pan)
                    loss = 0
                    weights = [4, 2, 1]
                    for bi in range(len(outs)):
                        t_gt = torch.nn.functional.interpolate(gt, outs[bi].shape[-2:], mode='bicubic').detach()
                        tmp = self.criterion(outs[bi], t_gt) * weights[bi]
                        loss += tmp
                    # out = outs[-1]
                    # loss = self.criterion(out, gt)
                elif self.model_name == 'LRTN':
                    out = model(ms, pan)
                    loss = self.criterion(out, gt)
                elif self.model_name == 'DCT':
                    out = model(ms, pan)
                    loss = self.criterion(out, gt)

                epoch_train_loss.append(loss.item())  # save all losses into a vector for one epoch
                loss.backward()
                optimizer.step()
            self.scheduler.step()

            t_loss = np.nanmean(np.array(epoch_train_loss))  # compute the mean value of all losses, as one epoch loss
            self.logger.info('Epoch: {}/{} training loss:{:.7f}'.format(epoch, self.epoch_num, t_loss))
            if self.writer:
                self.writer.add_scalar('train/loss', t_loss, epoch)  # write to tensorboard to check
            self.validate()
            if epoch % ckpt == 0 and not self.debug:
                self.save_checkpoint(epoch)
            epoch_time.update(time.time() - end)
            end = time.time()
            remain_time = self.calc_remain_time(epoch, epoch_time)
            self.logger.info(f"remain {remain_time}")
        return

    def validate(self):
        epoch_val_loss = []
        model, device = self.model, self.device

        model.eval()
        with torch.no_grad():
            for iteration, batch in enumerate(self.val_loader, 1):
                gt, lms, ms, pan = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device)
                if self.model_name == 'ARN':
                    AU, EU, hrms1, hrms2 = model(ms, lms, pan)
                    loss = self.criterion(hrms2, gt)
                elif self.model_name == 'RIJAN':
                    out = model(lms, pan)
                    loss = self.criterion(out, gt)
                elif self.model_name == 'MSL_RIJAN_PD':
                    # out, mid_outs = model(ms, lms, pan)
                    # loss = self.criterion(out, gt)
                    outs = model(ms, lms, pan)
                    loss = self.criterion(outs[-1], gt)
                elif self.model_name == 'PanVIT_UAG':
                    outs = model(pan, lms)
                    loss = self.criterion(outs[-1], gt)
                elif self.model_name == 'PanVIT':
                    out = model(pan, lms)
                    loss = self.criterion(out, gt)
                elif 'RIJAN_P' in self.model_name:
                    out = model(ms, lms, pan)
                    loss = self.criterion(out, gt)
                elif 'GuidedNet' in self.model_name:
                    outs = model(ms, pan)
                    loss = self.criterion(outs[-1], gt)
                elif self.model_name == 'LRTN':
                    out = model(ms, pan)
                    loss = self.criterion(out, gt)
                elif self.model_name == 'DCT':
                    out = model(ms, pan)
                    loss = self.criterion(out, gt)
                epoch_val_loss.append(loss.item())
        v_loss = np.nanmean(np.array(epoch_val_loss))
        # writer.add_scalar('val/loss', v_loss, epoch)
        self.logger.info('validate loss: {:.7f}'.format(v_loss))
        return

    def save_checkpoint(self, epoch):
        model_out_path = str(self.weights_save_path / f'CSNET{epoch}.pth')
        ckpt = {'state_dict': self.model.state_dict(), 'exp_timestamp': self.run_time}
        torch.save(ckpt, model_out_path)
        return

    def calc_remain_time(self, epoch, epoch_time):
        remain_time = (self.epoch_num - epoch) * epoch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))
        return remain_time


class Tester:
    def __init__(self, config):
        self.config = config
        dataset_name = config.dataset_name
        self.dataset_name = dataset_name
        self.model_name = config.model_name
        assert config.dataset_name in ('GF2', 'QB', 'WV3', 'WV2', 'CAVE_x4', 'CAVE_x8', 'HARVARD_x4', 'HARVARD_x8')
        assert config.test_mode in ('reduced', 'full')
        data_path = Path(config.data_path)
        if dataset_name in ['CAVE_x4', 'CAVE_x8', 'HARVARD_x4', 'HARVARD_x8']:
            dname = dataset_name.lower()
            test_data_path = str(data_path / dname / f'test_{dname}_rgb.h5')
            self.dataset = h5py.File(test_data_path, 'r')
        else:
            if config.test_mode == 'reduced':
                tmp = f'test data/h5/{dataset_name}/reduce_examples/test_{dataset_name.lower()}_multiExm1.h5'
            else:
                tmp = f'test data/h5/{dataset_name}/full_examples/test_{dataset_name.lower()}_OrigScale_multiExm1.h5'
            test_data_path = str(data_path / tmp)
            self.dataset = h5py.File(test_data_path, 'r')
        # rgb channel indexes for each dataset
        if dataset_name in ('GF2', 'QB'):
            self.rgb_idx = [0, 1, 2]
        elif dataset_name in ['CAVE_x4', 'CAVE_x8', 'HARVARD_x4', 'HARVARD_x8']:
            self.rgb_idx = [30, 19, 9]
        else:
            self.rgb_idx = [0, 2, 4]

        device = torch.device('cuda:0')
        # device = torch.device('cpu')
        self.device = device
        self.model = create_model(config, device)
        weight_path = config.test_weight_path
        ckpt = torch.load(weight_path, map_location=device)
        print(f"loading weight: {weight_path}")
        self.model.load_state_dict(ckpt['state_dict'])
        # sd = {}
        # for k in ckpt['param']:
        #     new_k = k.removeprefix('module.')
        #     sd[new_k] = ckpt['param'][k]
        # self.model.load_state_dict(sd, strict=True)
        # ckpt['exp_timestamp'] = '20241225-155524'
        # for rijab in self.model.rijabs:
        #     rijab.gate.offset = torch.tensor([[0., 0]])
        #     # rijab.gate.offset = torch.tensor([[0.05, 0]])
        #     # rijab.gate.offset = torch.tensor([[0.04, 0]])
        #     # rijab.gate.offset = torch.tensor([[0.1, 0]])
        # for r in self.model.rijabs:
        #     print(r.gate.offset)
        if dataset_name in ['CAVE_x4', 'CAVE_x8', 'HARVARD_x4', 'HARVARD_x8']:
            save_path = Path(config.results_path) / f"{dataset_name}/{ckpt['exp_timestamp']}"
        else:
            save_path = Path(config.results_path) / f"{dataset_name}/{config.test_mode}/{ckpt['exp_timestamp']}"
        save_path.mkdir(exist_ok=True, parents=True)
        self.save_path = save_path
        return

    def test(self, analyse_fms=False):
        features = defaultdict(list)

        def get_features(name):
            def hook(model, input, output):
                if 'UE.conv' in name:
                    features[name].append(output.detach().cpu().numpy())
                elif 'UE' in name:
                    features[name + '.fft_var'].append(output[0].detach().cpu().numpy())
                    features[name + '.fft_EU'].append(output[1].detach().cpu().numpy())
                    features[name + '.EU'].append(output[2].detach().cpu().numpy())
                    features[name + '.mean'].append(output[3].detach().cpu().numpy())
                else:
                    features[name].append(output.detach().cpu().numpy())

            return hook

        dataset, model, gt = self.dataset, self.model, None
        dev = self.device
        scale = 1.0 if self.dataset_name in ['CAVE_x4', 'CAVE_x8', 'HARVARD_x4', 'HARVARD_x8'] else 2047.0
        # analyse_fms = False
        if analyse_fms and self.model_name != 'GuidedNet':
            rijabs = model.rijabs
            for i in range(model.block_num):
                if i != model.block_num - 1:
                    continue
                cur_ln = f'rijab_{i}'
                # rijab_i = getattr(rijabs, cur_ln)
                rijab_i = rijabs[i]
                # rijab_i.register_forward_hook(get_features(cur_ln))
                rijab_i.UE.register_forward_hook(get_features(cur_ln + '.UE'))
                # rijab_i.gate.register_forward_hook(get_features(cur_ln + '.gate'))
                # rijab_i.sat1.register_forward_hook(get_features(cur_ln + '.sat1'))
                # rijab_i.sat3.register_forward_hook(get_features(cur_ln + '.sat3'))
                # rijab_i.sat5.register_forward_hook(get_features(cur_ln + '.sat5'))
                rijab_i.UE.conv.register_forward_hook(get_features(cur_ln + '.UE.conv'))
        if self.dataset_name in ['CAVE_x4', 'CAVE_x8', 'HARVARD_x4', 'HARVARD_x8']:
            ms = np.array(dataset['LRHSI'], dtype=np.float32)
            lms = np.array(dataset['HSI_up'], dtype=np.float32)
            pan = np.array(dataset['RGB'], dtype=np.float32)
            gt = np.array(dataset['GT'], dtype=np.float32)
        else:
            ms = np.array(dataset['ms'], dtype=np.float32) / scale
            lms = np.array(dataset['lms'], dtype=np.float32) / scale
            pan = np.array(dataset['pan'], dtype=np.float32) / scale
            if self.config.test_mode == 'reduced':
                gt = np.array(dataset['gt'], dtype=np.float32)

        ms = torch.from_numpy(ms).float()
        if 'x8' in self.dataset_name and self.model_name != 'GuidedNet':
            ms = torch.nn.functional.interpolate(ms, scale_factor=2, mode='bicubic').squeeze()
        lms = torch.from_numpy(lms).float()
        pan = torch.from_numpy(pan).float()
        model.eval()
        print(f"save files to {self.save_path}")
        final_outs = []

        # infer_time = AverageMeter()
        end = time.time()
        with torch.no_grad():
            # for i in range(2):
            for i in range(len(pan)):
                features = defaultdict(list)
                if self.model_name == 'RIJAN':
                    out = model(lms[i:i + 1].to(dev), pan[i:i + 1].to(dev))
                elif self.model_name == 'MSL_RIJAN_PD':
                    # out = split_patch_eval(ms[i:i + 1].to(dev), pan[i:i + 1].to(dev), model)
                    outs = model(ms[i:i + 1].to(dev), lms[i:i + 1].to(dev), pan[i:i + 1].to(dev))
                    out = outs[-1]
                elif 'RIJAN_P' in self.model_name:
                    out = model(ms[i:i + 1].to(dev), lms[i:i + 1].to(dev), pan[i:i + 1].to(dev))
                elif 'RIJAN_P' in self.model_name:
                    out = model(ms[i:i + 1].to(dev), lms[i:i + 1].to(dev), pan[i:i + 1].to(dev))

                I_SR = torch.squeeze(out * scale).cpu().detach().numpy()  # BxCxHxW
                print(f'image {i}')
                if self.dataset_name in ['CAVE_x4', 'CAVE_x8', 'HARVARD_x4', 'HARVARD_x8']:
                    final_outs.append(I_SR.transpose(1, 2, 0)[None, ...])
                else:
                    # save H, W, C
                    sio.savemat(str(self.save_path / f'output_mulExm_{i}.mat'), {'I_SR': I_SR.transpose(1, 2, 0)})
                    if analyse_fms:
                        save_path = self.save_path / 'visualize_fms'
                        save_path.mkdir(exist_ok=True, parents=True)
                        
        if self.dataset_name in ['CAVE_x4', 'CAVE_x8', 'HARVARD_x4', 'HARVARD_x8']:
            final_outs = np.concatenate(final_outs, axis=0)
            if 'CAVE' in self.dataset_name:
                path = str(self.save_path / 'cave11-Ours.mat')
            else:
                path = str(self.save_path / 'harvard10-Ours.mat')
            sio.savemat(path, {'output': final_outs})
        print('Total inference time: ', time.time() - end)
        return
