import torch
import torch.nn as nn
import torch.nn.functional as F


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, y1, y2, Weight):
        dis = torch.abs(y1 - y2)
        dis = dis * Weight
        return torch.mean(dis)


class UncertaintyLoss(nn.Module):
    def __init__(self):
        super(UncertaintyLoss, self).__init__()

    def forward(self, y1, y2, AU):
        dis = torch.pow(y1 - y2, 2)
        # s = s + 0.000001
        # l = dis/(2*s) + 0.5*torch.log(s)
        l = 0.5 * torch.exp(-1.0 * AU) * dis + 0.5 * AU
        return torch.mean(l)

    
class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss


class CharFreqLoss(nn.Module):
    """L1 (mean absolute error, MAE) loss of fft.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(CharFreqLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: none, mean, sum')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.l1_loss = CharbonnierLoss(loss_weight, reduction)

    def forward(self, pred, target):
        # diff = torch.abs(torch.fft.rfft2(pred)) - torch.abs(torch.fft.rfft2(target))
        # loss = torch.mean(torch.abs(diff))
        diff = torch.fft.rfft2(pred) - torch.fft.rfft2(target)
        loss = torch.mean(torch.abs(diff))
        return self.loss_weight * loss * 0.01 + self.l1_loss(pred, target)


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
    ms = torch.FloatTensor(1, 4, 16, 16)
    lms = torch.FloatTensor(1, 4, 64, 64)
    pan = torch.FloatTensor(1, 1, 64, 64)
    return dict(ms=ms, lms=lms, pan=pan)


if __name__ == '__main__':
    from ptflops import get_model_complexity_info

    N = ARN(in_ms_cnum=4, in_pan_cnum=1, max_step=(5, 5, 5))
    # N = ARN(in_ms_cnum=4, in_pan_cnum=1, max_step=(3, 3, 3))
    # N = ARN(in_ms_cnum=4, in_pan_cnum=1)

    macs, params = get_model_complexity_info(N, input_res=(1,), input_constructor=prepare_input, as_strings=True,
                                             print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
