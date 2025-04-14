import torch.utils.data as data
import torch
import h5py
import cv2
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader


class Dataset_Pro(data.Dataset):
    def __init__(self, file_path):
        super(Dataset_Pro, self).__init__()
        data = h5py.File(file_path)  # NxCxHxW = 0x1x2x3

        # tensor type:
        gt1 = data["gt"][...]  # convert to np tpye for CV2.filter
        gt1 = np.array(gt1, dtype=np.float32) / 2047
        self.gt = torch.from_numpy(gt1)  # NxCxHxW:

        # large ms
        lms1 = data["lms"][...]  # convert to np tpye for CV2.filter
        lms1 = np.array(lms1, dtype=np.float32) / 2047
        self.lms = torch.from_numpy(lms1)

        ms1 = data["ms"][...]  # NxCxHxW
        ms1 = np.array(ms1, dtype=np.float32) / 2047
        self.ms = torch.from_numpy(ms1)

        pan1 = data['pan'][...]  # Nx1xHxW
        pan1 = np.array(pan1, dtype=np.float32) / 2047  # Nx1xHxW
        self.pan = torch.from_numpy(pan1)  # Nx1xHxW:

    def __getitem__(self, index):
        return self.gt[index, :, :, :].float(), self.lms[index, :, :, :].float(), \
            self.ms[index, :, :, :].float(), self.pan[index, :, :, :].float()

    def __len__(self):
        return self.gt.shape[0]


class DatasetHSI(data.Dataset):
    def __init__(self, file_path, if_up=True):
        super(DatasetHSI, self).__init__()
        self.if_up = if_up
        dataset = h5py.File(file_path, 'r')
        self.file_path = file_path
        # print(dataset.keys())
        self.gt = dataset.get("GT")
        # print(self.GT.shape)
        self.up = dataset.get("HSI_up")
        self.lr_hsi = dataset.get("LRHSI")
        self.rgb = dataset.get("RGB")

    def __getitem__(self, index):
        input_rgb = torch.from_numpy(self.rgb[index, :, :, :]).float()
        input_lr = torch.from_numpy(self.lr_hsi[index, :, :, :]).float()
        if 'x8' in self.file_path and self.if_up:
            input_lr = torch.nn.functional.interpolate(input_lr[None, ...], scale_factor=2, mode='bicubic').squeeze()
        input_lr_u = torch.from_numpy(self.up[index, :, :, :]).float()
        target = torch.from_numpy(self.gt[index, :, :, :]).float()

        return target, input_lr_u, input_lr, input_rgb

    def __len__(self):
        return self.gt.shape[0]


def create_loaders(config):
    assert config.dataset_name in ('GF2', 'QB', 'WV3', 'WV2', 'CAVE_x4', 'CAVE_x8', 'HARVARD_x4', 'HARVARD_x8')
    data_path = Path(config.data_path)
    batch_size = config.batch_size

    # if training:
    dataset_name = config.dataset_name.lower()
    if config.dataset_name in ['CAVE_x4', 'CAVE_x8', 'HARVARD_x4', 'HARVARD_x8']:
        if config.dataset_name == 'CAVE_x8':
            train_data_path = str(data_path / f'{dataset_name}/train_{dataset_name}_rgb_16.h5')
            validate_data_path = str(data_path / f'{dataset_name}/validation_{dataset_name}_rgb_12.h5')
        else:
            train_data_path = str(data_path / f'{dataset_name}/train_{dataset_name}_rgb.h5')
            validate_data_path = str(data_path / f'{dataset_name}/validation_{dataset_name}_rgb.h5')
        train_set = DatasetHSI(train_data_path, config.model_name != 'GuidedNet')
        validate_set = DatasetHSI(validate_data_path, config.model_name != 'GuidedNet')
    else:
        train_data_path = str(data_path / f'training_{dataset_name}/train_{dataset_name}.h5')
        train_set = Dataset_Pro(train_data_path)
        validate_data_path = str(data_path / f'training_{dataset_name}/valid_{dataset_name}.h5')
        validate_set = Dataset_Pro(validate_data_path)

    training_data_loader = DataLoader(dataset=train_set, num_workers=config.workers, batch_size=batch_size,
                                      shuffle=True, pin_memory=True, drop_last=True)
    print('Train set ground truth shape', train_set.gt.shape)
    validate_data_loader = DataLoader(dataset=validate_set, num_workers=0, batch_size=batch_size, shuffle=False,
                                      pin_memory=True, drop_last=True)
    print('Validate set ground truth shape', validate_set.gt.shape)
    return training_data_loader, validate_data_loader

