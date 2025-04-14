import torch
from torch import nn
from collections import OrderedDict
# import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange
from masksembles.torch import Masksembles2D
from torch.nn import init as init
import math
import torch.nn.functional as F


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
    # adaptive depth-wise separable convolution
    def __init__(self, in_channels, out_channels, kernel_size_list=(1, 3, 5),
                 stride=1, padding_list=(0, 1, 2)):
        super(KernelGenerator, self).__init__()
        self.kernel_size_list = kernel_size_list
        self.padding_list = padding_list
        self.spatial_branch = nn.ModuleList()
        self.spectral_branch = nn.ModuleList()
        self.tail_convs = nn.ModuleList()
        self.in_channels = in_channels
        self.out_channels = out_channels

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
                nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(in_channels=in_channels, out_channels=in_channels * kernel_size ** 2, kernel_size=1),
            )
            tail_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
            self.spatial_branch.append(spatial_kg)
            self.spectral_branch.append(spectral_kg)
            self.tail_convs.append(tail_conv)
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
            output = self.tail_convs[i](self.adaptive_conv(x, i))
            spatial_kernel = spatial_kernels[i].permute(0, 2, 3, 1).reshape(b, 1, h, w, k, k)
            spectral_kernel = spectral_kernels[i].reshape(b, c, 1, 1, k, k)
            self.adaptive_kernel = torch.mul(spectral_kernel, spatial_kernel)
            output = self.tail_convs[i](self.adaptive_conv(x, i))
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
        # depth-wise convolution
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
        self.offset = torch.tensor([[0.7, 0]])
        # self.offset = torch.tensor([[0.0, 0]])
        # self.offset = torch.tensor([[0.2, 0]])
        # self.proj = nn.Sequential(
        #     nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(in_channels=hidden_channels, out_channels=2, kernel_size=1)
        # )

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
        # print('Forward: ', self.offset)
        offset = self.offset.repeat(r.shape[0], 1).to(r.device)
        r = r - offset
        index = r.max(dim=-1, keepdim=True)[1]
        r_hard = torch.zeros_like(x).scatter_(-1, index, 1.0)
        r = r_hard - r.detach() + r
        return r


class ESWUnit(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(ESWUnit, self).__init__()
        self.embed0 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.embed1 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 1, 1, 0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 1, 1, 0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 1, 1, 0),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embed0(x)
        x = self.embed1(x)
        r = self.sigmoid(x)
        return r


class SSWUnit(nn.Module):
    def __init__(self, in_channels, hidden_channels, pool_size=1):
        super(SSWUnit, self).__init__()
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
            nn.Linear(hidden_channels, 1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embed0(x)
        x = self.adaptive_pool(x)
        b, c, h, w = x.shape
        x = x.reshape(b, c*h*w)
        x = self.embed1(x)
        r = self.sigmoid(x).reshape(b, 1, 1, 1)
        return r


def summaries(model, writer=None, grad=False, torchsummary=None):
    if grad:
        from torchsummary import summary
        summary(model, input_size=[(1, 64, 64), (8, 64, 64)], batch_size=1)
    else:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)

    if writer is not None:
        x = torch.randn(1, 64, 64, 64)
        writer.add_graph(model, (x,))


class Conv_Block(nn.Module):
    def __init__(self, channels, k=3, dw=False):
        super(Conv_Block, self).__init__()
        if dw:
            self.conv_block = nn.Sequential(
                nn.Conv2d(channels, channels, k, 1, k // 2, groups=channels),
                nn.Conv2d(channels, channels, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, k, 1, k // 2, groups=channels),
                nn.Conv2d(channels, channels, 1)
            )
        else:
            self.conv_block = nn.Sequential(
                nn.Conv2d(channels, channels, k, 1, k // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, k, 1, k // 2)
            )

    def forward(self, x):
        return self.conv_block(x)


class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return y.expand_as(x)


class RIJA(nn.Module):
    def __init__(self, channels, k=3):
        super(RIJA, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels * 2, 2, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(2, 2, k, 1, k // 2)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        sat = self.conv1(torch.cat([x, y], dim=1))
        sat = self.sigmoid(sat)
        return sat


class RIJAK(nn.Module):
    def __init__(self, channels, k=3, dw=False):
        super(RIJAK, self).__init__()
        self.sat = RIJA(channels, k=k)
        self.CB = Conv_Block(channels, k=k, dw=dw)

    def forward(self, x):
        resx = self.CB(x)
        satmap = self.sat(x, resx)  # b,2,H,W
        x = x * satmap[:, 0, :, :].unsqueeze(1).expand_as(x)
        resx = resx * satmap[:, 1, :, :].unsqueeze(1).expand_as(resx)
        x = resx + x
        return x


# A simple block focusing on a fixed iterative approach without additional adaptive or dynamic adjustments.
class RIJAB(nn.Module):
    def __init__(self, channels, retimes, k=3):
        super(RIJAB, self).__init__()
        self.retimes = retimes
        self.sat1 = RIJAK(channels, k=1)
        self.sat3 = RIJAK(channels, k=3)
        self.sat5 = RIJAK(channels, k=5)
        self.weight_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.Sigmoid()
        )
        self.tail_conv = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x):
        for i in range(self.retimes):
            r1 = self.sat1(x)
            r3 = self.sat3(x)
            r5 = self.sat5(x)
            r_ = (r1 + r3 + r5) / 3
            x = x + r_
        x = x * self.weight_conv(x)
        x = self.tail_conv(x)
        return x


# single-scale RIJAN which utilizes RIJAB
class RIJAN(nn.Module):
    def __init__(self, in_channels=9, out_channels=8, hidden_channels=32, block_num=3, retimes=7):
        super(RIJAN, self).__init__()
        self.block_num = block_num
        self.recur_times = retimes

        self.head_conv = nn.Conv2d(in_channels, hidden_channels, 3, 1, 1)
        self.rijabs = self._make_blocks(hidden_channels, block_num, retimes)
        self.eca = eca_layer(channel=hidden_channels)
        self.tail_conv = nn.Conv2d(hidden_channels, out_channels, 3, 1, 1)

    def forward(self, lms, pan):
        x = torch.cat([pan, lms], dim=1)  # b,9,H,W
        x = self.head_conv(x)
        x = self.rijabs(x)
        # x = x * self.eca(x)
        sr = self.tail_conv(x) * self.eca(lms)
        return sr

    def _make_blocks(self, channels, block_num, retimes):
        blocks = []
        for i in range(block_num):
            blocks.append(("rijab_{}".format(i), RIJAB(channels, retimes)))
        return nn.Sequential(OrderedDict(blocks))


# RIJAB_P extends the capabilities of RIJAB by introducing several additional (optional) features such as
# adaptive kernel generator (adaptive), dynamism (gate unit), recursive, and DW convolutions (dw)
class RIJAB_P(nn.Module):
    def __init__(self, channels, retimes, adaptive=False, dynamic=False, dw=False, recursive=True,
                 gate_channel=8, pool_size=2):
        super(RIJAB_P, self).__init__()
        self.retimes = retimes
        self.adaptive = adaptive
        self.dynamic = dynamic
        self.recursive = recursive
        if not adaptive:
            if dw:
                self.head_conv = nn.Sequential(
                    nn.Conv2d(channels * 2, channels * 2, 3, 1, 1, groups=channels * 2),
                    nn.Conv2d(channels * 2, channels, 1),
                )
            else:
                self.head_conv = nn.Conv2d(channels * 2, channels, 3, 1, 1)
            if recursive:
                self.sat1 = RIJAK(channels, k=1, dw=dw)
                self.sat3 = RIJAK(channels, k=3, dw=dw)
                self.sat5 = RIJAK(channels, k=5, dw=dw)
            else:
                self.sat1 = nn.ModuleList([RIJAK(channels, k=1, dw=dw) for _ in range(retimes)])
                self.sat3 = nn.ModuleList([RIJAK(channels, k=3, dw=dw) for _ in range(retimes)])
                self.sat5 = nn.ModuleList([RIJAK(channels, k=5, dw=dw) for _ in range(retimes)])
        else:
            self.kpn = KernelGenerator(channels, channels)
        if dynamic:
            self.gate = GateUnit(channels, gate_channel, pool_size)
        if dw:
            self.weight_conv = nn.Sequential(
                nn.Conv2d(channels, channels, 3, 1, 1, groups=channels),
                nn.Conv2d(channels, channels, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, 3, 1, 1, groups=channels),
                nn.Conv2d(channels, channels, 1),
                nn.Sigmoid()
            )
            self.tail_conv = nn.Sequential(
                nn.Conv2d(channels, channels, 3, 1, 1, groups=channels),
                nn.Conv2d(channels, channels, 1),
            )
        else:
            self.weight_conv = nn.Sequential(
                nn.Conv2d(channels, channels, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, 3, 1, 1),
                nn.Sigmoid()
            )
            self.tail_conv = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x, y):
        if not self.adaptive:
            x = self.head_conv(torch.cat([x, y], dim=1))
        for i in range(self.retimes):
            if not self.adaptive:
                if self.recursive:
                    r1 = self.sat1(x)
                    r3 = self.sat3(x)
                    r5 = self.sat5(x)
                else:
                    r1 = self.sat1[i](x)
                    r3 = self.sat3[i](x)
                    r5 = self.sat5[i](x)
            else:
                [r1, r3, r5] = self.kpn(x, y)
            r_ = (r1 + r3 + r5) / 3
            if self.dynamic:
                g = self.gate(x)
                r_ = g[:, 1].reshape(-1, 1, 1, 1) * r_
            x = x + r_
        x = x * self.weight_conv(x)
        x = self.tail_conv(x)
        return x


# RIJAN in a progressive way (multi-scale MS and PAN injection)
class RIJAN_P(nn.Module):
    def __init__(self, in_ms_cnum=8, in_pan_cnum=1, hidden_channels=32, block_num=3, retimes=5, adaptive=False,
                 recursive=True, dynamic=False, dw=False, gate_channel=(32, 16, 8), pool_size=(2, 4, 8)):
        super(RIJAN_P, self).__init__()
        self.in_ms_cnum = in_ms_cnum
        self.in_pan_cnum = in_pan_cnum
        self.hidden_channel = hidden_channels
        self.block_num = block_num
        self.recur_times = retimes
        self.adaptive = adaptive
        self.dynamic = dynamic
        self.dw = dw
        self.gate_channel = gate_channel
        self.pool_size = pool_size

        self.ps = nn.PixelShuffle(upscale_factor=2)
        self.proj_ms = nn.ModuleList()
        self.proj_pan = nn.ModuleList()
        self.up_ms = nn.ModuleList()
        self.down_pan = nn.ModuleList()
        self.up_fm = nn.ModuleList()
        self.aggregate_ms = nn.ModuleList()

        self.rijabs = self._make_blocks(hidden_channels, block_num, retimes, recursive)
        self.eca = eca_layer(channel=hidden_channels)
        self.tail_conv = nn.Conv2d(hidden_channels, in_ms_cnum, 3, 1, 1)

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

        for i in range(self.block_num):
            ms_input = self.proj_ms[i](ms_inputs[i])
            if i != 0:
                aggregate_fm = torch.cat((ms_input, block_f), dim=1)
                ms_input = self.aggregate_ms[i - 1](aggregate_fm)
            pan_input = self.proj_pan[i](pan_guidance[i])
            block_f = self.rijabs[i](ms_input, pan_input)
            if i != self.block_num - 1:
                block_f = self.up_fm[i](block_f)
                block_f = self.ps(block_f)
        sr = self.tail_conv(block_f) * self.eca(lms)
        return sr

    def _make_blocks(self, channels, block_num, retimes, recursive):
        blocks = nn.ModuleList()
        for i in range(block_num):
            blocks.append(
                RIJAB_P(channels, retimes, self.adaptive, self.dynamic, self.dw, recursive,
                        self.gate_channel[i], self.pool_size[i])
            )
            self.proj_ms.append(nn.Conv2d(in_channels=self.in_ms_cnum, out_channels=self.hidden_channel,
                                          kernel_size=3, stride=1, padding=1, bias=True))
            self.proj_pan.append(nn.Conv2d(in_channels=self.in_pan_cnum, out_channels=self.hidden_channel,
                                           kernel_size=5, stride=1, padding=2, bias=True))
            if i != 0:
                self.up_ms.append(
                    nn.Conv2d(in_channels=self.in_ms_cnum, out_channels=self.in_ms_cnum * 4,
                              kernel_size=3, stride=1, padding=1, bias=True)
                )
                self.down_pan.append(
                    nn.Conv2d(in_channels=1, out_channels=self.in_pan_cnum,
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
        return blocks


class UncertaintyEstimation(nn.Module):
    # def __init__(self, channels, T, q, ms_cnum=4):
    def __init__(self, channels, n, s, ms_cnum=4):
        super(UncertaintyEstimation, self).__init__()
        # self.T = T
        # self.q = q

        self.n = n
        self.s = s
        self.mask = Masksembles2D(channels, self.n, self.s)

        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=True, groups=channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1),
        )
        # self.main_fft = fft_bench_complex_mlp(channels, norm='backward', act_method=nn.GELU)
        # self.main_fft = FreBlockCha(channels)

        self.out = nn.Sequential(
            nn.Conv2d(channels, ms_cnum, 1, 1, 0),
        )

    def random_mask(self, x, q):
        mask = np.random.binomial(n=1, p=1 - q, size=(self.T, x.shape[1]))
        mask = torch.tensor(mask).to(x.device)
        mask = rearrange(mask, "T C -> T 1 C 1 1")
        return x * mask

    def epistemic_uncetainty(self, x, lms=None):
        # xs = x.unsqueeze(0).repeat([self.T, 1, 1, 1, 1])
        # xs = self.random_mask(xs, self.q)
        # t, b, c, h, w = xs.shape
        # xs = xs.reshape(t*b, c, h, w)
        # xs = self.out(xs).reshape(t, b, -1, h, w)

        b, c, h, w = x.shape
        xs = x.repeat([self.n, 1, 1, 1])
        xs = self.mask(xs)
        xs = self.out(xs).reshape(self.n, b, -1, h, w)
        # lms addition should add here
        EU, mean = torch.var_mean(input=xs, dim=0, unbiased=True)
        mean = mean + lms
        # return EU, mean

        fft_xs = torch.fft.fft2(xs)
        # fft_amp = fft_xs.real ** 2 + fft_xs.imag ** v
        # fft_amp = torch.sqrt(fft_amp)
        # fft_pha = torch.atan2(fft_xs.imag, fft_xs.real)
        # fft_EU = torch.cat([torch.var(fft_amp, dim=0), torch.var(fft_pha, dim=0)], dim=1)
        # t0, t1 = torch.var(fft_xs.real, dim=0), torch.var(fft_xs.imag, dim=0)
        # fft_EU = torch.cat([torch.var(fft_xs.real, dim=0), torch.var(fft_xs.imag, dim=0)], dim=1)
        fft_EU = torch.std(fft_xs, dim=0) / (torch.mean(fft_xs, dim=0) + 10e-6)
        fft_var = torch.std(fft_xs, dim=0)
        fft_EU = torch.cat([fft_EU.real, fft_EU.imag], dim=1)
        return fft_var, fft_EU, EU, mean

    def forward(self, x, lms=None):
        # r = self.conv(x)
        # r_ = self.main_fft(x)
        # x = x + r + r_
        x = self.conv(x)
        # EU, mean = self.epistemic_uncetainty(x, lms)
        # return EU, mean
        fft_var, fft_EU, EU, mean = self.epistemic_uncetainty(x, lms)
        return fft_var, fft_EU, EU, mean


class fft_bench_complex_mlp(nn.Module):
    def __init__(self, dim, dw=1, norm='backward', act_method=nn.ReLU, window_size=None, bias=False):
        super(fft_bench_complex_mlp, self).__init__()
        self.act_fft = act_method()
        self.window_size = window_size
        # dim = out_channel
        hid_dim = dim * dw
        # print(dim, hid_dim)
        self.complex_weight1_real = nn.Parameter(torch.Tensor(dim, hid_dim))
        self.complex_weight1_imag = nn.Parameter(torch.Tensor(dim, hid_dim))
        self.complex_weight2_real = nn.Parameter(torch.Tensor(hid_dim, dim))
        self.complex_weight2_imag = nn.Parameter(torch.Tensor(hid_dim, dim))
        init.kaiming_uniform_(self.complex_weight1_real, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight1_imag, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight2_real, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight2_imag, a=math.sqrt(16))
        if bias:
            self.b1_real = nn.Parameter(torch.zeros((1, 1, 1, hid_dim)), requires_grad=True)
            self.b1_imag = nn.Parameter(torch.zeros((1, 1, 1, hid_dim)), requires_grad=True)
            self.b2_real = nn.Parameter(torch.zeros((1, 1, 1, dim)), requires_grad=True)
            self.b2_imag = nn.Parameter(torch.zeros((1, 1, 1, dim)), requires_grad=True)
        self.bias = bias
        self.norm = norm

    def forward(self, x):
        _, _, H, W = x.shape
        y = torch.fft.rfft2(x, norm=self.norm)
        dim = 1
        weight1 = torch.complex(self.complex_weight1_real, self.complex_weight1_imag)
        weight2 = torch.complex(self.complex_weight2_real, self.complex_weight2_imag)
        if self.bias:
            b1 = torch.complex(self.b1_real, self.b1_imag)
            b2 = torch.complex(self.b2_real, self.b2_imag)
        y = rearrange(y, 'b c h w -> b h w c')
        y = y @ weight1
        if self.bias:
            y = y + b1
        y = torch.cat([y.real, y.imag], dim=dim)

        y = self.act_fft(y)
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        y = y @ weight2
        if self.bias:
            y = y + b2
        y = rearrange(y, 'b h w c -> b c h w')
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return y


class FreBlockCha(nn.Module):
    def __init__(self, nc):
        super(FreBlockCha, self).__init__()
        self.processreal = nn.Sequential(
            nn.Conv2d(nc, nc, kernel_size=1, padding=0, stride=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, kernel_size=1, padding=0, stride=1))
        self.processimag = nn.Sequential(
            nn.Conv2d(nc, nc, kernel_size=1, padding=0, stride=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, kernel_size=1, padding=0, stride=1))

    def forward(self, x):
        _, _, H, W = x.shape
        x = torch.fft.rfft2(x, norm='backward')
        real = self.processreal(x.real)
        imag = self.processimag(x.imag)
        x_out = torch.complex(real, imag)
        x_out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')
        return x_out


# final version used with uncertainty estimation
# adaptive kernel generator (adaptive), gate unit (dynamic), recursive, and DW convolutions (dw)
class MSL_RIJAB_P(nn.Module):
    def __init__(self, ms_cnum, channels, retimes, adaptive=False, dynamic=False, dw=False, recursive=True,
                 gate_method='gate', gate_input='all', gate_channel=8, pool_size=2):
        super(MSL_RIJAB_P, self).__init__()
        self.retimes = retimes
        self.adaptive = adaptive
        self.dynamic = dynamic
        self.recursive = recursive
        self.gate_method = gate_method
        self.gate_input = gate_input
        self.ms_cnum = ms_cnum
        if not adaptive:
            self.head_conv = nn.Conv2d(channels * 2, channels, 3, 1, 1)
            if recursive:
                self.sat1 = RIJAK(channels, k=1, dw=dw)
                self.sat3 = RIJAK(channels, k=3, dw=dw)
                self.sat5 = RIJAK(channels, k=5, dw=dw)
            else:
                self.sat1 = nn.ModuleList([RIJAK(channels, k=1, dw=dw) for _ in range(retimes)])
                self.sat3 = nn.ModuleList([RIJAK(channels, k=3, dw=dw) for _ in range(retimes)])
                self.sat5 = nn.ModuleList([RIJAK(channels, k=5, dw=dw) for _ in range(retimes)])
            # self.main_fft = FreBlockCha(channels)
        else:
            self.kpn = KernelGenerator(channels, channels)
        if dynamic:
            assert gate_method in ['gate', 'ESW', 'SSW']
            assert gate_input in ['all', 'wo_FU', 'wo_SU', 'only_F']
            if gate_method == 'gate':
                if gate_input == 'all':
                    in_channels = channels + ms_cnum * 3
                elif gate_input == 'wo_FU':
                    in_channels = channels + ms_cnum
                elif gate_input == 'wo_SU':
                    in_channels = channels + ms_cnum * 2
                else:
                    in_channels = channels
                self.gate = GateUnit(in_channels, gate_channel, pool_size)
            elif gate_method == 'ESW':
                self.gate = ESWUnit(channels + ms_cnum * 3, channels)
            else:
                self.gate = SSWUnit(channels + ms_cnum * 3, channels)
            # self.gate = GateUnit(channels if not ue else channels + ms_cnum, gate_channel, pool_size)

        self.UE = UncertaintyEstimation(channels, n=12, s=4, ms_cnum=ms_cnum)

    def forward(self, x, y, lms):
        if not self.adaptive:
            x = self.head_conv(torch.cat([x, y], dim=1))
        if lms.shape[2:] != x.shape[2:]:
            lms = torch.nn.functional.interpolate(lms, x.shape[2:], mode='bicubic')
        for i in range(self.retimes):
            if not self.adaptive:
                if self.recursive:
                    r1 = self.sat1(x)
                    r3 = self.sat3(x)
                    r5 = self.sat5(x)
                else:
                    r1 = self.sat1[i](x)
                    r3 = self.sat3[i](x)
                    r5 = self.sat5[i](x)
            else:
                [r1, r3, r5] = self.kpn(x, y)
            r_ = (r1 + r3 + r5) / 3
            # r_ = self.main_fft(x) + r_
            if self.dynamic and i != 0:
                fft_var, fft_eu, eu, _ = self.UE(x, lms)
                if self.gate_input == 'all':
                    g = self.gate(torch.cat([fft_eu, eu, x], dim=1))
                elif self.gate_input == 'wo_FU':
                    g = self.gate(torch.cat([eu, x], dim=1))
                elif self.gate_input == 'wo_SU':
                    g = self.gate(torch.cat([fft_eu, x], dim=1))
                else:
                    g = self.gate(x)
                if self.gate_method == 'gate':
                    r_ = g[:, 1].reshape(-1, 1, 1, 1) * r_
                else:
                    r_ = g * r_
            x = x + r_
        # eu, out = self.UE(x)
        fft_var, fft_eu, eu, out = self.UE(x, lms)
        return x, out

# final version of RIJAN in a progressive way with uncertainty estimation and gating
class MSL_RIJAN_P(nn.Module):
    def __init__(self, in_ms_cnum=8, in_pan_cnum=1, hidden_channels=32, block_num=3, retimes=5, adaptive=False,
                 recursive=True, dynamic=False, dw=False, gate_method='gate', gate_input='all',
                 gate_channel=(32, 16, 8), pool_size=(2, 4, 8)):
        super(MSL_RIJAN_P, self).__init__()
        self.in_ms_cnum = in_ms_cnum
        self.in_pan_cnum = in_pan_cnum
        self.hidden_channel = hidden_channels
        self.block_num = block_num
        self.recur_times = retimes
        self.adaptive = adaptive
        self.dynamic = dynamic
        self.dw = dw
        self.gate_channel = gate_channel
        self.pool_size = pool_size

        self.ps = nn.PixelShuffle(upscale_factor=2)
        self.proj_ms = nn.ModuleList()
        self.proj_pan = nn.ModuleList()
        self.proj_out = nn.ModuleList()
        self.up_ms = nn.ModuleList()
        self.down_pan = nn.ModuleList()
        self.up_out = nn.ModuleList()
        self.up_fm = nn.ModuleList()
        self.aggregate_ms = nn.ModuleList()

        self.rijabs = self._make_blocks(hidden_channels, block_num, retimes, recursive, gate_method, gate_input)

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

        outs = []
        for i in range(self.block_num):
            ms_input = self.proj_ms[i](ms_inputs[i])
            if i != 0:
                up_out = self.ps(self.up_out[i - 1](outs[i - 1]))
                proj_out = self.proj_out[i - 1](up_out)
                aggregate_fm = torch.cat((proj_out, ms_input, block_f), dim=1)
                ms_input = self.aggregate_ms[i - 1](aggregate_fm)
            pan_input = self.proj_pan[i](pan_guidance[i])
            block_f, out = self.rijabs[i](ms_input, pan_input, lms)
            outs.append(out)
            if i != self.block_num - 1:
                block_f = self.up_fm[i](block_f)
                block_f = self.ps(block_f)
        return outs

    def _make_blocks(self, channels, block_num, retimes, recursive, gate_method='gate', gate_input='all'):
        blocks = nn.ModuleList()
        for i in range(block_num):
            blocks.append(
                MSL_RIJAB_P(self.in_ms_cnum, channels, retimes, self.adaptive, self.dynamic, self.dw, recursive,
                            gate_method, gate_input, self.gate_channel[i], self.pool_size[i])
            )
            self.proj_ms.append(nn.Conv2d(in_channels=self.in_ms_cnum, out_channels=self.hidden_channel,
                                          kernel_size=3, stride=1, padding=1, bias=True))
            self.proj_pan.append(nn.Conv2d(in_channels=self.in_pan_cnum, out_channels=self.hidden_channel,
                                           kernel_size=5, stride=1, padding=2, bias=True))
            if i != 0:
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
                self.up_out.append(
                    nn.Conv2d(in_channels=self.in_ms_cnum, out_channels=self.in_ms_cnum * 4,
                              kernel_size=3, stride=1, padding=1, bias=True)
                )
                self.proj_out.append(
                    nn.Conv2d(in_channels=self.in_ms_cnum, out_channels=self.hidden_channel,
                              kernel_size=3, stride=1, padding=1, bias=True)
                )
                self.aggregate_ms.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels=self.hidden_channel * 3, out_channels=self.hidden_channel * 3,
                                  groups=self.hidden_channel * 3, kernel_size=3, padding=1, bias=True),
                        nn.Conv2d(in_channels=self.hidden_channel * 3, out_channels=self.hidden_channel, kernel_size=1)
                    )
                )
        return blocks


def prepare_input(resolution):
    ms = torch.FloatTensor(1, ms_num, 16, 16)
    lms = torch.FloatTensor(1, ms_num, 64, 64)
    pan = torch.FloatTensor(1, 1, 64, 64)

    return dict(ms=ms, lms=lms, pan=pan)


if __name__ == '__main__':
    # simulation code for masksembles
    # t = torch.tensor([[1, 2], [3, 4], [5, 6]])
    # print(t.shape)
    # tt = t.repeat(4, 1)
    # print(tt.shape)
    # b = torch.split(tt.unsqueeze(1), 12 // 4, dim=0)
    # print(b[0].shape, len(b))
    # bb = torch.cat(b, dim=1).permute([1, 0, 2])
    # # omit mask multiplication, so that the final result should be the same as original tensor
    # print(bb.shape)
    # bb_r = torch.cat(torch.split(bb, 1, dim=0), dim=1)
    # print(bb_r.shape)
    # bb_r = bb_r.squeeze(0)
    # f_r = bb_r.reshape(4, 3, 2)
    # print(f_r.float().mean(dim=0))

    # from torchsummary import summary
    from ptflops import get_model_complexity_info
    from fvcore.nn import FlopCountAnalysis, parameter_count_table

    # summary(RIJAN(9, 8), input_size=[(1, 64, 64), (8, 64, 64)], device='cpu')
    # N = RIJAN(5, 4)
    # N = RIJAN(34, 31, 64)
    # N = RIJAN_P(4, 1)
    # N = RIJAN(5, 4)
    # N = RIJAN_P(4, 1, adaptive=True)
    # N = RIJAN_P(4, 1, adaptive=True, dynamic=True)
    # N = RIJAN_P(4, 1, adaptive=False, dynamic=True)
    # N = RIJAN_P(4, 1, adaptive=False, dynamic=True, dw=True)

    ms_num = 8
    # ms_num = 4
    # Reported in the paper
    # N = MSL_RIJAN_P(ms_num, 1)
    # w/o ADA
    # N = MSL_RIJAN_P(ms_num, 1, dynamic=False)
    # w/o REC
    # N = MSL_RIJAN_P(ms_num, 1, recursive=False)
    # with ESW
    # N = MSL_RIJAN_P(ms_num, 1, gate_method='ESW')
    # with SSW
    # N = MSL_RIJAN_P(ms_num, 1, gate_method='SSW')
    # wo_FU
    # N = MSL_RIJAN_P(ms_num, 1, gate_input='wo_FU')
    # wo_SU
    N = MSL_RIJAN_P(ms_num, 1, gate_input='wo_SU')
    # with F only
    # N = MSL_RIJAN_P(ms_num, 1, adaptive=False, dynamic=True, gate_input='only_F')
    # M.R.D=1
    # N = MSL_RIJAN_P(ms_num, 1, retimes=1, gate_input='wo_SU')

    flops, params = get_model_complexity_info(N, input_res=(1,), input_constructor=prepare_input, as_strings=True,
                                              print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', flops))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    ms, lms, pan = torch.rand(1, ms_num, 16, 16), torch.rand(1, ms_num, 64, 64), torch.rand(1, 1, 64, 64)
    flops = FlopCountAnalysis(N, (ms, lms, pan))
    print(f"FLOPs: {flops.total() / 1e9}G")
    # print(f"Params: {parameter_count_table(N) / 1e6}M")
    print(f"{parameter_count_table(N)}")
    # print('Params and FLOPs are {}M and {}G'.format(params/1e6, flops/1e9))
