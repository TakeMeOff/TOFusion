import torch
from pytorch_msssim import ssim
import torch.nn.functional as F
import torch.nn as nn
from tools import check_img_range
import numpy as np

class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)

def max_int(img_vi, img_if, result):
    source_max = torch.max(img_vi[:, :1, :, :], img_if[:,:1, :, :])
    return F.l1_loss(source_max, result[:, :1, :, :])

def max_grad(img_vi, img_if, mask_h, mask_l, mask_ph, mask_pl, result):
    sobelconv = Sobelxy()
    vi_grad = sobelconv(img_vi[:, :1, :, :])
    if_grad = sobelconv(img_if[:, :1, :, :])
    result_grad = sobelconv(result[:, :1, :, :])
    original_grad = torch.max(vi_grad, if_grad)
    return F.l1_loss(original_grad, result_grad)

def two_ssim(img_vi, img_if, result, ratio=0.5, range=1):
    vi_ssim = ssim(img_vi[:,:1, :, :], data_range=range, size_average=True)
    if_ssim = ssim(img_if[:,:1, :, :], data_range=range, size_average=True)
    return 1 - ratio * vi_ssim - (1 - ratio) * if_ssim


def loss_cal(img_vi_h, img_if_h, mask_h, mask_l, mask_ph, mask_pl, result):
    # Mask Loss
    cir = nn.BCEWithLogitsLoss()
    loss_mask = cir(mask_ph, mask_h) + cir(mask_pl, mask_l)


    if_mask = check_img_range(mask_ph)

    # Texture Loss for fusion
    sobelconv = Sobelxy()
    vi_grad = sobelconv(img_vi_h[:, :1, :, :])
    if_grad = sobelconv(img_if_h[:, :1, :, :])
    result_grad = sobelconv(result[:, :1, :, :])
    original_grad = torch.max(vi_grad, if_grad * if_mask)
    loss_grad = F.l1_loss(original_grad, result_grad)

    # Intensity Loss
    orginal = torch.max(img_vi_h[:, :1, :, :], img_if_h[:, :1, :, :] * if_mask)
    loss_int = F.l1_loss(result[:, :1, :, :], orginal)



    return loss_int, loss_grad, loss_mask


