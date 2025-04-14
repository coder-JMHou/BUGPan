import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from torch.nn.init import calculate_gain


# -------------Initialization----------------------------------------
def init_weights(*modules):
    for module in modules:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):  ## initialization for Conv2d

                # variance_scaling_initializer(m.weight)  # method 1: initialization
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')  # method 2: initialization
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):  ## initialization for BN
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):  ## initialization for nn.Linear
                # variance_scaling_initializer(m.weight)
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)


# -------------ResNet Block (One)----------------------------------------
class EfficientResBlock(nn.Module):
    def __init__(self, channel=64):
        super(EfficientResBlock, self).__init__()
        channel = channel
        self.conv = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1,
                              bias=True)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        rs1 = self.relu(x)
        rs1 = self.conv(rs1)
        rs = torch.add(x, rs1)
        return rs


class EfficientNet(nn.Module):
    def __init__(self, blocknum=10, channel=64):
        super(EfficientNet, self).__init__()
        self.res_block = nn.ModuleList()
        self.channel = channel
        for i in range(blocknum):
            self.res_block.append(EfficientResBlock(self.channel))
        self.body = nn.Sequential(*self.res_block)

    def forward(self, x):
        rs = self.body(x)
        return rs


# -----------------------------------------------------
class GuidedNet(nn.Module):
    def __init__(self, spectral_num=8, HR_spe=1, layer_num=2, blocknum=10, channel=64):
        super(GuidedNet, self).__init__()
        self.layer_num = layer_num
        self.channel = channel
        self.blocknum = blocknum
        self.ps = nn.PixelShuffle(upscale_factor=2)
        self.spectral_num = spectral_num
        self.HR_spe = HR_spe
        self.loss_weight = [2, 1]
        self.conv0 = nn.Conv2d(in_channels=self.spectral_num, out_channels=channel, kernel_size=3, stride=1, padding=1,
                               bias=True)

        self.component0 = nn.ModuleList()
        self.component1 = nn.ModuleList()
        self.component2 = nn.ModuleList()
        self.component3 = nn.ModuleList()
        self.component4 = nn.ModuleList()
        self.down = nn.ModuleList()

        for i in range(self.layer_num):
            # down-sampling for pan
            self.down.append(
                nn.Conv2d(in_channels=self.HR_spe, out_channels=self.HR_spe, kernel_size=5, stride=2, padding=2,
                          bias=True))
            # up-sampling and pixel shuffle for ms feature
            self.component0.append(
                nn.Conv2d(in_channels=channel, out_channels=channel * 4, kernel_size=3, stride=1, padding=1, bias=True))
            # proj for concat feature of ms and pan
            self.component1.append(
                nn.Conv2d(in_channels=channel + self.HR_spe, out_channels=channel, kernel_size=3, stride=1, padding=1,
                          bias=True))
            # proj again for concat feature of ms and pan
            self.component2.append(EfficientNet(self.blocknum, self.channel))
            # up-sampling and pixel shuffle for ms image
            self.component3.append(
                nn.Conv2d(in_channels=self.spectral_num, out_channels=self.spectral_num * 4, kernel_size=3, stride=1,
                          padding=1, bias=True))
            # conv for residual construction
            self.component4.append(
                nn.Conv2d(in_channels=channel, out_channels=self.spectral_num, kernel_size=3, stride=1, padding=1,
                          bias=True))
        self.apply(init_weights)
        self.l1loss = torch.nn.L1Loss()

    def forward(self, x, y):  # x= hp of ms; y = hp of pan
        fea = self.conv0(x)
        guidance = [y]
        results = []
        image = x
        hr = y

        for i in range(self.layer_num - 1):
            hr = self.down[i](hr)
            guidance.insert(0, hr)

        for i in range(self.layer_num):
            fea = self.component0[i](fea)  # Bsx9x64x64
            fea = self.ps(fea)
            # print(fea.shape, guidance[i].shape)
            fea = torch.cat([fea, guidance[i]], dim=1)
            fea = self.component1[i](fea)
            fea = self.component2[i](fea)
            image = self.component3[i](image)
            image = self.ps(image)

            residual = self.component4[i](fea)
            image += residual
            results.append(image)

        # return results[-1]
        return results

    # def criterion(self, results, gt):
    #     gts = [gt]
    #     sr = gt
    #     loss = 0
    #     for i in range(self.layer_num-1):
    #         gti = torch.nn.functional.interpolate(sr, scale_factor=1/2, mode='bicubic',align_corners=True)
    #         gts.insert(0, gti)
    #     for i in range(self.layer_num):
    #         lossi = self.l1loss(results[i], gts[i])
    #         loss += lossi * self.loss_weight[i]
    #     return loss


# Filter normalization
class Norm(nn.Module):
    def __init__(self, in_channels, kernel_size, filter_type,
                 nonlinearity='leaky_relu', running_std=False, running_mean=False):
        assert filter_type in ('spatial', 'spectral')
        assert in_channels >= 1
        super(Norm, self).__init__()
        self.in_channels = in_channels
        self.filter_type = filter_type
        self.runing_std = running_std
        self.runing_mean = running_mean
        if kernel_size is tuple:
            kernel_size = sum(kernel_size)
        std = calculate_gain(nonlinearity) / kernel_size
        if running_std:
            self.std = nn.Parameter(
                torch.randn(in_channels * kernel_size ** 2) * std, requires_grad=True)
        else:
            self.std = std
        if running_mean:
            self.mean = nn.Parameter(
                torch.randn(in_channels * kernel_size ** 2), requires_grad=True)

    def forward(self, x):
        if self.filter_type == 'spatial':
            # calculate mean and std at kernel size dimension
            # x - [b, k**2, h, w]
            b, _, h, w = x.size()
            x = x.reshape(b, self.in_channels, -1, h, w)
            x = x - x.mean(dim=2).reshape(b, self.in_channels, 1, h, w)
            x = x / (x.std(dim=2).reshape(b, self.in_channels, 1, h, w) + 1e-10)
            x = x.reshape(b, _, h, w)
            if self.runing_std:
                x = x * self.std[None, :, None, None]
            else:
                x = x * self.std
            if self.runing_mean:
                x = x + self.mean[None, :, None, None]
        elif self.filter_type == 'spectral':
            # x - [b, c, 1, 1]
            b = x.size(0)
            c = self.in_channels
            x = x.reshape(b, c, -1)
            x = x - x.mean(dim=2).reshape(b, c, 1)
            x = x / (x.std(dim=2).reshape(b, c, 1) + 1e-10)
            x = x.reshape(b, -1)
            if self.runing_std:
                x = x * self.std[None, :]
            else:
                x = x * self.std
            if self.runing_mean:
                x = x + self.mean[None, :]
        else:
            raise RuntimeError('Unsupported filter type {}'.format(self.filter_type))
        return x


class KernelNorm(nn.Module):
    def __init__(self, in_channels, filter_type):
        super(KernelNorm, self).__init__()
        assert filter_type in ('spatial', 'spectral')
        assert in_channels >= 1
        self.in_channels = in_channels
        self.filter_type = filter_type

    def forward(self, x):
        if self.filter_type == 'spatial':
            # calculate mean and std at kernel size dimension
            # x - [b, sum(k**2), h, w]
            b, _, h, w = x.size()
            x = x.reshape(b, self.in_channels, -1, h, w)
            x = x - x.mean(dim=2).reshape(b, self.in_channels, 1, h, w)
            x = x / (x.std(dim=2).reshape(b, self.in_channels, 1, h, w) + 1e-10)
            x = x.reshape(b, _, h, w)
        elif self.filter_type == 'spectral':
            # x - [b, c, sum(k**2)]
            b = x.size(0)
            c = self.in_channels
            x = x.reshape(b, c, -1)
            x = x - x.mean(dim=2).reshape(b, c, 1)
            x = x / (x.std(dim=2).reshape(b, c, 1) + 1e-10)
        else:
            raise RuntimeError('Unsupported filter type {}'.format(self.filter_type))
        return x


class KernelGenerator(nn.Module):
    def __init__(self, in_channels, kernel_size_list=(1, 3, 5), stride=1, padding_list=(0, 1, 2), se_ratio=0.5):
        super(KernelGenerator, self).__init__()
        self.kernel_size_list = kernel_size_list
        self.padding_list = padding_list
        self.spatial_branch = nn.ModuleList()
        self.spectral_branch = nn.ModuleList()
        self.in_channels = in_channels
        assert se_ratio > 0
        mid_channels = int(in_channels * se_ratio)

        self.cross_spatial_attn = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, 1, 0),
            nn.ReLU(True),
            nn.Conv2d(in_channels, 2, 1, 1, 0),
            nn.Sigmoid(),
        )

        for i in range(len(kernel_size_list)):
            kernel_size, padding = kernel_size_list[i], padding_list[i]
            spatial_kg = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                          kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels),
                nn.Conv2d(in_channels=in_channels, out_channels=kernel_size ** 2, kernel_size=1),
                nn.Conv2d(in_channels=kernel_size ** 2, out_channels=kernel_size ** 2,
                          kernel_size=kernel_size, padding=padding, groups=kernel_size ** 2),
                nn.Conv2d(in_channels=kernel_size ** 2, out_channels=kernel_size ** 2, kernel_size=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(in_channels=kernel_size ** 2, out_channels=kernel_size ** 2,
                          kernel_size=kernel_size, padding=padding, groups=kernel_size ** 2),
                nn.Conv2d(in_channels=kernel_size ** 2, out_channels=kernel_size ** 2, kernel_size=1),
            )
            spectral_kg = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(in_channels=mid_channels, out_channels=in_channels * kernel_size ** 2, kernel_size=1),
            )
            self.spatial_branch.append(spatial_kg)
            self.spectral_branch.append(spectral_kg)
        self.spatial_norm = KernelNorm(in_channels=1, filter_type='spatial')
        self.spectral_norm = KernelNorm(in_channels=in_channels, filter_type='spectral')

    def forward(self, x, y):  # x stands for LR-MS features while y stands for PAN features
        b, c, h, w = x.shape
        attn = self.cross_spatial_attn(torch.cat([x, y], 1))
        attn1, attn2 = torch.chunk(attn, 2, 1)
        x, y = x * attn1, y * attn2

        outputs, spatial_kernels, spectral_kernels = [], [], []
        for i, k in enumerate(self.kernel_size_list):
            spatial_kernel = self.spatial_branch[i](y)  # [b, 1*k**2, h, w]
            spectral_kernel = self.spectral_branch[i](x)  # [b, c*k**2, 1, 1]
            spectral_kernel = spectral_kernel.reshape(b, self.in_channels, k ** 2)
            spatial_kernels.append(spatial_kernel)
            spectral_kernels.append(spectral_kernel)
        k_square = list(k ** 2 for k in self.kernel_size_list)
        spatial_kernels = torch.cat(spatial_kernels, dim=1)
        spatial_kernels = self.spatial_norm(spatial_kernels)
        spatial_kernels = spatial_kernels.split(k_square, dim=1)
        spectral_kernels = torch.cat(spectral_kernels, dim=-1)
        spectral_kernels = self.spectral_norm(spectral_kernels)
        spectral_kernels = spectral_kernels.split(k_square, dim=-1)

        for i, k in enumerate(self.kernel_size_list):
            spatial_kernel = spatial_kernels[i].permute(0, 2, 3, 1).reshape(b, 1, h, w, k, k)
            spectral_kernel = spectral_kernels[i].reshape(b, c, 1, 1, k, k)
            self.adaptive_kernel = torch.mul(spectral_kernel, spatial_kernel)
            output = self.adaptive_conv(x, i)
            outputs.append(output)
        return outputs

    def adaptive_conv(self, x, i):
        b, c, h, w = x.shape
        pad = self.padding_list[i]
        k = self.kernel_size_list[i]
        kernel = self.adaptive_kernel
        x_pad = torch.zeros(b, c, h + 2 * pad, w + 2 * pad, device=x.device)
        if pad > 0:
            x_pad[:, :, pad:-pad, pad:-pad] = x
        else:
            x_pad = x
        x_pad = F.unfold(x_pad, (k, k))
        x_pad = x_pad.reshape(b, c, k, k, h, w).permute(0, 1, 4, 5, 2, 3)
        return torch.sum(torch.mul(x_pad, kernel), [4, 5])


class GateUnit(nn.Module):
    def __init__(self, in_channels, hidden_channels, pool_size=1):
        super(GateUnit, self).__init__()
        self.embed0 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        self.embed1 = nn.Sequential(
            nn.Linear(pool_size ** 2 * hidden_channels, hidden_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_channels, 2),
        )
        self.offset = torch.tensor([[0.6, 0]])

    def forward(self, x):
        x = self.embed0(x)
        x = self.adaptive_pool(x)
        b, c, h, w = x.shape
        # x = self.proj(x)
        # x = x.squeeze(-1).squeeze(-1)
        x = x.reshape(b, c*h*w)
        x = self.embed1(x)
        if self.training:
            g = x.uniform_()
            g = -torch.log(-torch.log(g + 1e-20) + 1e-20)
            r = F.softmax(x + g, dim=-1)
        else:
            r = F.softmax(x, dim=-1)
        offset = self.offset.repeat(r.shape[0], 1).to(r.device)
        r = r - offset
        index = r.max(dim=-1, keepdim=True)[1]
        r_hard = torch.zeros_like(x).scatter_(-1, index, 1.0)
        r = r_hard - r.detach() + r
        return r


# Adaptive recursive module
class ARM(nn.Module):
    def __init__(self, hidden_channels, gate_channel, max_step=5, pool_size=2):
        super(ARM, self).__init__()
        self.kpn = KernelGenerator(hidden_channels)
        # self.out_conv = nn.Conv2d(in_channels=hidden_channels * 3, out_channels=hidden_channels,
        #                           kernel_size=1, stride=1, bias=True)
        # self.out_conv = nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels,
        #                           kernel_size=1, stride=1, bias=True)
        self.gate = GateUnit(hidden_channels, gate_channel, pool_size)
        self.max_step = max_step
        self.weight_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, groups=hidden_channels),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels,  kernel_size=3, padding=1, groups=hidden_channels),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.tail_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, groups=hidden_channels),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
        )
        return

    def forward(self, x, y):
        r0 = torch.zeros(x.shape).to(x.device)
        i = 0
        while i < self.max_step:
            g = self.gate(x)
            [r1, r3, r5] = self.kpn(x, y)
            r_ = (r1 + r3 + r5) / 3
            r = g[:, 0].reshape(-1, 1, 1, 1) * r0 + g[:, 1].reshape(-1, 1, 1, 1) * r_
            x = x + r
            i += 1
        x = x * self.weight_conv(x)
        x = self.tail_conv(x)
        return x


class ECALayer(nn.Module):
    def __init__(self, k_size=3):
        super(ECALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return y.expand_as(x)


# Adaptive recursive network
class ARN(nn.Module):
    def __init__(self, in_ms_cnum=8, in_pan_cnum=1, block_num=3, max_step=(1, 2, 3), hidden_channel=32,
                 pool_size=(2, 4, 8), gate_channel=(32, 16, 8)):
        super(ARN, self).__init__()
        self.in_ms_cnum = in_ms_cnum
        self.in_pan_cnum = in_pan_cnum
        self.block_num = block_num
        self.max_step = max_step
        self.pool_size = pool_size
        self.gate_channel = gate_channel
        self.hidden_channel = hidden_channel

        self.ps = nn.PixelShuffle(upscale_factor=2)
        self.proj_ms = nn.ModuleList()
        self.proj_pan = nn.ModuleList()
        self.recursive_blocks = nn.ModuleList()

        self.up_ms = nn.ModuleList()
        self.down_pan = nn.ModuleList()
        self.up_fm = nn.ModuleList()
        self.aggregate_ms = nn.ModuleList()
        for bi in range(block_num):
            self.proj_ms.append(nn.Conv2d(in_channels=self.in_ms_cnum, out_channels=self.hidden_channel,
                                          kernel_size=3, stride=1, padding=1, bias=True))
            self.proj_pan.append(nn.Conv2d(in_channels=self.in_pan_cnum, out_channels=self.hidden_channel,
                                           kernel_size=5, stride=1, padding=2, bias=True))
            self.recursive_blocks.append(ARM(self.hidden_channel, self.gate_channel[bi],
                                             self.max_step[bi], self.pool_size[bi]))
            if bi != 0:
                self.up_ms.append(
                    nn.Conv2d(in_channels=self.in_ms_cnum, out_channels=self.in_ms_cnum * 4,
                              kernel_size=3, stride=1, padding=1, bias=True)
                )
                self.down_pan.append(
                    nn.Conv2d(in_channels=self.in_pan_cnum, out_channels=self.in_pan_cnum,
                              kernel_size=5, stride=2, padding=2, bias=True)
                )
                self.up_fm.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels=self.hidden_channel, out_channels=self.hidden_channel,
                                  groups=self.hidden_channel, kernel_size=3, padding=1, bias=True),
                        nn.Conv2d(in_channels=self.hidden_channel, out_channels=self.hidden_channel * 4, kernel_size=1)
                    )

                )
                self.aggregate_ms.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels=self.hidden_channel * 2, out_channels=self.hidden_channel * 2,
                                  groups=self.hidden_channel * 2, kernel_size=3, padding=1, bias=True),
                        nn.Conv2d(in_channels=self.hidden_channel * 2, out_channels=self.hidden_channel, kernel_size=1)
                    )
                )
        self.out_conv = nn.Conv2d(in_channels=self.hidden_channel, out_channels=self.in_ms_cnum,
                                  kernel_size=3, stride=1, padding=1, bias=True)
        self.eca = ECALayer()

    def forward(self, ms, lms, pan):
        lr = ms
        hr = pan
        ms_inputs = [lr]
        pan_guidance = [hr]
        for i in range(self.block_num - 1):
            hr = self.down_pan[i](hr)
            pan_guidance.insert(0, hr)
            lr = self.up_ms[i](lr)
            lr = self.ps(lr)
            ms_inputs.append(lr)

        block_output = None
        for i in range(self.block_num):
            ms_input = self.proj_ms[i](ms_inputs[i])
            if i != 0:
                aggregate_fm = torch.cat((ms_input, block_output), dim=1)
                ms_input = self.aggregate_ms[i - 1](aggregate_fm)
            pan_input = self.proj_pan[i](pan_guidance[i])
            block_output = self.recursive_blocks[i](ms_input, pan_input)
            if i != self.block_num - 1:
                block_output = self.up_fm[i](block_output)
                block_output = self.ps(block_output)
        final_output = self.out_conv(block_output) * self.eca(lms)
        # why eca do not use self-attention?
        # final_output += lms
        return final_output


class our_criterion(nn.Module):
    def __init__(self, layer_num=2):
        super(our_criterion, self).__init__()
        self.layer_num = layer_num
        self.loss_weight = [2, 1]
        self.l1loss = torch.nn.L1Loss()

    def forward(self, results, gt):
        gts = [gt]
        sr = gt
        loss = 0
        for i in range(self.layer_num - 1):
            gti = torch.nn.functional.interpolate(sr, scale_factor=1 / 2, mode='bicubic', align_corners=True)
            gts.insert(0, gti)
        for i in range(self.layer_num):
            lossi = self.l1loss(results[i], gts[i])
            loss += lossi * self.loss_weight[i]
        return loss


def summaries(model, writer=None, grad=False):
    if grad:
        from torchsummary import summary
        summary(model, input_size=[(8, 16, 16), (1, 64, 64)], batch_size=1)
    else:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)

    if writer is not None:
        x = torch.randn(1, 64, 64, 64)
        writer.add_graph(model, (x,))


def prepare_input(resolution):
    # ms = torch.FloatTensor(1, 8, 64, 64)
    # ms = torch.FloatTensor(1, 4, 16, 16)
    # lms = torch.FloatTensor(1, 4, 64, 64)
    # pan = torch.FloatTensor(1, 1, 64, 64)
    # return dict(ms=ms, lms=lms, pan=pan)
    ms = torch.FloatTensor(1, 8, 16, 16)
    pan = torch.FloatTensor(1, 1, 64, 64)
    return dict(x=ms, y=pan)


if __name__ == '__main__':
    from ptflops import get_model_complexity_info
    from torchsummary import summary
    from fvcore.nn import FlopCountAnalysis, parameter_count_table

    N = GuidedNet()

    macs, params = get_model_complexity_info(N, input_res=(1,), input_constructor=prepare_input, as_strings=True,
                                             print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    
    # summary(N, [(8, 16, 16), (1, 64, 64)], device='cpu')
    print(parameter_count_table(N))
    ms, pan = torch.rand(1, 8, 16, 16), torch.rand(1, 1, 64, 64)
    # t = N(ms, pan)
    # print(t.shape)
    flops = FlopCountAnalysis(N, (ms, pan))
    print("FLOPs: ", flops.total())
