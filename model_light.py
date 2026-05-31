from typing import Any, Union, List
from loss import *
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT, LRSchedulerPLType

from tools import *
import numpy as np
import torch.nn as nn
from utils.restormer import TransformerBlock, SIE, CIE, FeatureWiseAffine
import pytorch_lightning as pl
import clip


class MLP(nn.Module):
    def __init__(self, input_channels, hidden_channels, out_channels):
        super(MLP, self).__init__()
        layers = []
        current_channels = input_channels
        for middle_channels in hidden_channels:
            layers.append(nn.Conv2d(current_channels, middle_channels, 1, bias=False))
            layers.append(nn.PReLU())
            current_channels = middle_channels

        layers.append(nn.Conv2d(current_channels, out_channels, kernel_size=1, bias=False))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class MaskDecoder(nn.Module):
    def __init__(self, restormer_dim=32, ffn_factor=2.66, num_heads=8, bias=False, Layer_type='WithBias'):
        super(MaskDecoder, self).__init__()
        self.resotmer1 = TransformerBlock(restormer_dim, num_heads, ffn_factor, bias, Layer_type)
        self.conv1 = nn.Conv2d(in_channels=restormer_dim, out_channels=restormer_dim // 2, kernel_size=3,
                               stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=restormer_dim // 2, out_channels=16, kernel_size=3,
                               stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1,
                               stride=1, padding=0)
        self.action1 = nn.PReLU()
        self.action2 = nn.PReLU()

    def forward(self, x):
        x = self.resotmer1(x)
        x = self.action1(self.conv1(x))
        x = self.action2(self.conv2(x))
        x = self.conv3(x)
        return x


class MaskDecoder_two(nn.Module):
    def __init__(self):
        super(MaskDecoder_two, self).__init__()
        self.mlp1 = MLP(input_channels=64, hidden_channels=[64, 128, 256, 128, 64], out_channels=32)
        self.mlp2 = MLP(input_channels=16, hidden_channels=[16, 32, 16], out_channels=1)
        self.conv1 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3,
                               stride=1, padding=1)
        self.action = nn.PReLU()

    def forward(self, x):
        x = self.mlp1(x)
        x = self.action(self.conv1(x))
        x = self.mlp2(x)
        return x


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()
        self.prelu3 = nn.PReLU()

    def forward(self, x):
        x = self.prelu1(self.conv1(x))
        x = self.prelu2(self.conv2(x))
        x = self.prelu3(self.conv3(x))
        return x
    
class FeatureExtractor2(nn.Module):
    def __init__(self):
        super(FeatureExtractor2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()
        self.prelu3 = nn.PReLU()
        self.prelu4 = nn.PReLU()

    def forward(self, x):
        x = self.prelu1(self.conv1(x))
        x = self.prelu2(self.conv2(x))
        x = self.prelu3(self.conv3(x))
        x = self.prelu4(self.conv4(x))
        return x


class ResidualDenseBlock(nn.Module):
    def __init__(self, input_channels=64, growth_rate=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=growth_rate, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=input_channels + growth_rate, out_channels=growth_rate,
                               kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=input_channels + growth_rate * 2, out_channels=growth_rate,
                               kernel_size=3, stride=1, padding=1)
        self.action = nn.LeakyReLU(0.2)
        self.last_conv = nn.Conv2d(in_channels=input_channels + growth_rate * 3, out_channels=input_channels,
                                   kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x1 = self.action(self.conv1(x))
        x2 = self.action(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.action(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.last_conv(torch.cat((x, x1, x2, x3), 1))
        return x + x4
    


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.last = nn.Tanh()
        self.prelu2 = nn.PReLU()
        self.prelu1 = nn.PReLU()

    def forward(self, x):
        x = self.prelu1(self.conv1(x))
        x = self.prelu2(self.conv2(x))
        x = self.last(self.conv3(x))
        return x
    
class Decoder2(nn.Module):
    def __init__(self):
        super(Decoder2, self).__init__()
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.last = nn.Tanh()
        self.prelu2 = nn.PReLU()
        self.prelu1 = nn.PReLU()

    def forward(self, x):
        x = self.prelu1(self.conv1(x))
        x = self.prelu2(self.conv2(x))
        x = self.last(self.conv3(x))
        return x

# dim=64
class TOAFusion_light_two(nn.Module):
    def __init__(self, restormer_dim=32, ffn_factor=2.66, num_heads=8, bias=False, Layer_type='WithBias'):
        super(TOAFusion_light_two, self).__init__()
        self.FE_vi_H = FeatureExtractor()
        self.FE_if_H = FeatureExtractor()
        self.FE_vi_L = FeatureExtractor()
        self.FE_if_L = FeatureExtractor()
        self.RDB_vi_H = ResidualDenseBlock()
        self.RDB_vi_L = ResidualDenseBlock()
        self.RDB_if_H = ResidualDenseBlock()
        self.RDB_if_L = ResidualDenseBlock()
        self.CIE_H = CIE(restormer_dim * 2, ffn_factor, num_heads, bias)
        self.SIE_L = SIE(restormer_dim * 2, ffn_factor, num_heads, bias)
        self.SIE_H = SIE(restormer_dim * 2, ffn_factor, num_heads, bias)
        self.mask_H = MaskDecoder_two()
        self.mask_L = MaskDecoder_two()
        self.conv_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv_3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.decoder = Decoder()
        self.relu = nn.ReLU()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.text_inject_H = FeatureWiseAffine(in_channels=512, out_channels=64)
        self.text_inject_L = FeatureWiseAffine(in_channels=512, out_channels=64)

    def forward(self, img_vi_h, img_vi_l, img_if_h, img_if_l, text_feature=None):
        vi_h = self.RDB_vi_H(self.FE_vi_H(img_vi_h))
        if_h = self.RDB_if_H(self.FE_if_H(img_if_h))
        vi_l = self.RDB_vi_L(self.FE_vi_L(img_vi_l))
        if_l = self.RDB_if_L(self.FE_if_L(img_if_l))

        feature_sie_l = self.SIE_L(vi_l, if_l)
        feature_sie_h = self.SIE_H(vi_h, if_h)
        feature_cie_h = self.CIE_H(vi_h, if_h)

        feature_sie_h = self.text_inject_H(feature_sie_h, text_feature)
        feature_sie_l = self.text_inject_L(feature_sie_l, text_feature)

        mask_low = self.mask_L(feature_sie_l)
        feature_sie_l = self.up(feature_sie_l)
        feature_sie_h = self.relu(self.conv_1(torch.cat((feature_sie_l, feature_sie_h), dim=1)))
        feature_sie_h = self.relu(self.conv_2(feature_sie_h))
        mask_high = self.mask_H(feature_sie_h)

        feature_fusion = torch.cat((feature_cie_h, feature_sie_h), dim=1)
        feature_fusion = self.relu(self.conv_3(feature_fusion))
        feature_fusion = self.decoder(feature_fusion)

        return feature_fusion, mask_high, mask_low

# dim=64 Without SIE CIE   
class TOAFusion_light_three(nn.Module):
    def __init__(self, restormer_dim=32, ffn_factor=2.66, num_heads=8, bias=False, Layer_type='WithBias'):
        super(TOAFusion_light_three, self).__init__()
        self.FE_vi_H = FeatureExtractor()
        self.FE_if_H = FeatureExtractor()
        self.FE_vi_L = FeatureExtractor()
        self.FE_if_L = FeatureExtractor()
        self.RDB_vi_H = ResidualDenseBlock()
        self.RDB_vi_L = ResidualDenseBlock()
        self.RDB_if_H = ResidualDenseBlock()
        self.RDB_if_L = ResidualDenseBlock()
        # self.CIE_H = CIE(restormer_dim * 2, ffn_factor, num_heads, bias)
        # self.SIE_L = SIE(restormer_dim * 2, ffn_factor, num_heads, bias)
        # self.SIE_H = SIE(restormer_dim * 2, ffn_factor, num_heads, bias)
        self.CIE_H = nn.Conv2d(restormer_dim*4, restormer_dim*2, kernel_size=3, stride=1, padding=1)
        self.SIE_L = nn.Conv2d(restormer_dim*4, restormer_dim*2, kernel_size=3, stride=1, padding=1)
        self.SIE_H = nn.Conv2d(restormer_dim*4, restormer_dim*2, kernel_size=3, stride=1, padding=1)
        self.mask_H = MaskDecoder_two()
        self.mask_L = MaskDecoder_two()
        self.conv_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv_3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.decoder = Decoder()
        self.relu = nn.ReLU()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.text_inject_H = FeatureWiseAffine(in_channels=512, out_channels=64)
        self.text_inject_L = FeatureWiseAffine(in_channels=512, out_channels=64)

    def forward(self, img_vi_h, img_vi_l, img_if_h, img_if_l, text_feature=None):
        vi_h = self.RDB_vi_H(self.FE_vi_H(img_vi_h))
        if_h = self.RDB_if_H(self.FE_if_H(img_if_h))
        vi_l = self.RDB_vi_L(self.FE_vi_L(img_vi_l))
        if_l = self.RDB_if_L(self.FE_if_L(img_if_l))

        # feature_sie_l = self.SIE_L(vi_l, if_l)
        # feature_sie_h = self.SIE_H(vi_h, if_h)
        # feature_cie_h = self.CIE_H(vi_h, if_h)

        feature_sie_l = self.SIE_L(torch.cat((vi_l, if_l), dim=1))
        feature_sie_h = self.SIE_H(torch.cat((vi_h, if_h), dim=1))
        feature_cie_h = self.CIE_H(torch.cat((vi_h, if_h), dim=1))

        feature_sie_h = self.text_inject_H(feature_sie_h, text_feature)
        feature_sie_l = self.text_inject_L(feature_sie_l, text_feature)

        mask_low = self.mask_L(feature_sie_l)
        feature_sie_l = self.up(feature_sie_l)
        feature_sie_h = self.relu(self.conv_1(torch.cat((feature_sie_l, feature_sie_h), dim=1)))
        feature_sie_h = self.relu(self.conv_2(feature_sie_h))
        mask_high = self.mask_H(feature_sie_h)

        feature_fusion = torch.cat((feature_cie_h, feature_sie_h), dim=1)
        feature_fusion = self.relu(self.conv_3(feature_fusion))
        feature_fusion = self.decoder(feature_fusion)

        return feature_fusion, mask_high, mask_low


class TOAFusion_lightning(pl.LightningModule):
    def __init__(self, parser=None, text_path=None):
        super().__init__()
        self.TOAFusion = TOAFusion_light_three()
        self.clip, _ = clip.load("ViT-B/32")
        for param in self.clip.parameters():
            param.requires_grad = False
        self.text_path = text_path
        self.parser = parser

    def forward(self, img_vi_h, img_vi_l, img_if_h, img_if_l, text_feature=None):
        return self.TOAFusion(img_vi_h, img_vi_l, img_if_h, img_if_l, text_feature)

    def training_step(self, batch, batch_idx):
        img_vi_h, img_vi_l, img_if_h, img_if_l, mask_h, mask_l, names, type_fusions = batch
        text_lines = []

        for type_fusion in type_fusions:
            text_path_in = os.path.join(self.text_path, type_fusion + '.txt')
            with open(text_path_in, 'r', encoding='utf-8') as file:
                text_file = file.readlines()
                text_lines.append(get_text(text_file))

        text_pre = clip.tokenize(text_lines).to(self.device)
        text_feature = self.clip.encode_text(text_pre)

        result, mask_ph, mask_pl = self.TOAFusion(img_vi_h[:, :1, :, :], img_vi_l[:, :1, :, :],
                                                  img_if_h[:, :1, :, :], img_if_l[:, :1, :, :], text_feature)

        loss_int, loss_grad, loss_mask = loss_cal(img_vi_h, img_if_h, mask_h, mask_l, mask_ph, mask_pl, result)

        loss_total = loss_int * self.parser.ratio_int + loss_grad * self.parser.ratio_grad + loss_mask * self.parser.ratio_mask

        values = {"loss_total": loss_total,
                  "loss_int": loss_int.item() * self.parser.ratio_int,
                  "loss_grad": loss_grad.item() * self.parser.ratio_grad,
                  "loss_mask": loss_mask.item() * self.parser.ratio_mask,
                  "lr": self.optimizers().param_groups[0]['lr']}
        self.log_dict(values)

        return loss_total

    def predict_step(self, batch):
        img_vi_h, img_vi_l, img_if_h, img_if_l, names, text = batch
        text_lines = []
        for i in range(len(names)):
            text_lines.append(text[0])
        text_pre = clip.tokenize(text_lines).to(self.device)
        text_feature = self.clip.encode_text(text_pre)

        result, mask_ph, mask_pl = self.TOAFusion(img_vi_h[:, :1, :, :], img_vi_l[:, :1, :, :],
                                                  img_if_h[:, :1, :, :], img_if_l[:, :1, :, :], text_feature)

        result = check_img_range(result)
        result = torch.cat((result, img_vi_h[:, 1:, :, :]), dim=1)
        mask_path = os.path.join(self.parser.save_path, 'mask')
        fusion_path = os.path.join(self.parser.save_path, 'fusion')

        write_image(mask_ph, names, mask_path )
        write_image(result, names, fusion_path)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.TOAFusion.parameters(), lr=self.parser.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                               T_max=self.parser.epochs,
                                                               eta_min=self.parser.lr_end)
        return {"optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1 
                },
            }
    
    def set_par(self, par):
        self.parser=par

    def set_text(self, text):
        self.text=text


if __name__ == '__main__':
    model = TOAFusion_light_two()
    x1 = torch.rand(2, 1, 64, 64)
    x2 = torch.rand(2, 1, 64, 64)
    x3 = torch.rand(2, 1, 32, 32)
    x4 = torch.rand(2, 1, 32, 32)
    text = torch.randn(2, 512)
    out = model(x1, x3, x2, x4, text)
    print(count_parameters(model))
