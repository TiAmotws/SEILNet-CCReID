import torchvision
from torch import nn
from torch.nn import init
from models.utils import pooling
import torch
import torch.nn.functional as F
from models.classifier import Classifier


class ResNet50(nn.Module):
    def __init__(self, config, num_identities, **kwargs):
        super().__init__()

        resnet50 = torchvision.models.resnet50(pretrained=True)
        if config.MODEL.RES4_STRIDE == 1:
            resnet50.layer4[0].conv2.stride=(1, 1)
            resnet50.layer4[0].downsample[0].stride=(1, 1)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        if config.MODEL.POOLING.NAME == 'avg':
            self.globalpooling = nn.AdaptiveAvgPool2d(1)
        elif config.MODEL.POOLING.NAME == 'max':
            self.globalpooling = nn.AdaptiveMaxPool2d(1)
        elif config.MODEL.POOLING.NAME == 'gem':
            self.globalpooling = pooling.GeMPooling(p=config.MODEL.POOLING.P)
        elif config.MODEL.POOLING.NAME == 'maxavg':
            self.globalpooling = pooling.MaxAvgPooling()
        else:
            raise KeyError("Invalid pooling: '{}'".format(config.MODEL.POOLING.NAME))
        self.part_num = 7
        self.part_cls_layer = nn.Conv2d(in_channels=2048,
                                        out_channels=self.part_num,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)
        self.bn = nn.BatchNorm1d(config.MODEL.FEATURE_DIM)
        init.normal_(self.bn.weight.data, 1.0, 0.02)
        init.constant_(self.bn.bias.data, 0.0)
        self.bn1= nn.BatchNorm1d(config.MODEL.FEATURE_DIM)
        init.normal_(self.bn1.weight.data, 1.0, 0.02)
        init.constant_(self.bn1.bias.data, 0.0)
        self.bn2= nn.BatchNorm1d(config.MODEL.FEATURE_DIM)
        init.normal_(self.bn2.weight.data, 1.0, 0.02)
        init.constant_(self.bn2.bias.data, 0.0)

        self.identity_classifier = Classifier(feature_dim=config.MODEL.FEATURE_DIM, num_classes=num_identities)
        self.identity_classifier2 = Classifier(feature_dim=config.MODEL.FEATURE_DIM, num_classes=num_identities)

    def forward(self, x):
        if self.training:
            x, unclo = x
            unclo_map = self.base(unclo)
            unclo = self.globalpooling(unclo_map)
            unclo = unclo.view(unclo.size(0), -1)
            unclo_f = self.bn2(unclo)
            unclo_s = self.identity_classifier(unclo_f)
        x = self.base(x)
        N, f_h, f_w = x.size(0), x.size(2), x.size(3)
        part_cls_score = self.part_cls_layer(x)  # [16, 7, 16, 8]
        part_pred = F.softmax(part_cls_score, dim=1)  # [16, 7, 16, 8]
        clo = torch.cat((part_pred[:, 2, :, :].unsqueeze(1), part_pred[:, 3, :, :].unsqueeze(1), part_pred[:, 5, :, :].unsqueeze(1)), dim=1)
        y_noclo = self.globalpooling(x * part_pred[:, 1, :, :].view(N, 1, f_h, f_w))
        y_clo = self.globalpooling(x * torch.sum(clo, 1).view(N, 1, f_h, f_w))
        # y_fore = self.globalpooling(x * torch.sum(part_pred[:, 1:self.part_num, :, :], 1).view(N, 1, f_h, f_w))
        y_noclo = y_noclo.view(y_noclo.size(0), -1)
        y_clo = y_clo.view(y_clo.size(0), -1)
        y_fore = self.globalpooling(x)
        y_fore = y_fore.view(y_fore.size(0), -1)
        # g = y_g.view(y_g.size(0), -1)
        y_noclo = self.bn(y_noclo)
        y_clo = self.bn1(y_clo)
        y_fore = self.bn2(y_fore)
        s_fore = self.identity_classifier(y_fore)
        s_noclo = self.identity_classifier2(y_noclo)
        if self.training:
            return s_noclo, y_noclo, y_clo, s_fore, y_fore, part_cls_score, unclo_s, unclo_f
        else:
            return y_fore


class MutualCrossAttn(nn.Module):
    def __init__(self, feature_dim=2048, num_head=32):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_head = num_head

        self.layer_norm1 = nn.LayerNorm(self.feature_dim, eps=1e-6).apply(_init_vit_weights)
        self.layer_norm2 = nn.LayerNorm(self.feature_dim, eps=1e-6).apply(_init_vit_weights)

    def forward(self, f_a, f_s):
        B, N, C = f_a.shape

        q1 = f_a
        k1 = v1 = f_s
        # [B, N, C] -> [B, N, n, C//n] -> [B, n, N, C//n]
        q1 = q1.reshape(B, N, self.num_head, C // self.num_head).transpose(1, 2)
        k1 = k1.reshape(B, N, self.num_head, C // self.num_head).transpose(1, 2)
        v1 = v1.reshape(B, N, self.num_head, C // self.num_head).transpose(1, 2)

        attn1 = torch.matmul(q1 / ((C // self.num_head) ** 0.5),
                             k1.transpose(-1, -2))
        attn1 = F.softmax(attn1, dim=-1)
        output1 = torch.matmul(attn1, v1)
        # [B, n, N, C//n] -> [B, N, n, C//n] -> [B, N, C]
        output1 = output1.transpose(1, 2).contiguous().flatten(2)

        q2 = f_s
        k2 = v2 = f_a
        # [B, N, C] -> [B, N, n, C//n] -> [B, n, N, C//n]
        q2 = q2.reshape(B, N, self.num_head, C // self.num_head).transpose(1, 2)
        k2 = k2.reshape(B, N, self.num_head, C // self.num_head).transpose(1, 2)
        v2 = v2.reshape(B, N, self.num_head, C // self.num_head).transpose(1, 2)

        attn2 = torch.matmul(q2 / ((C // self.num_head) ** 0.5),
                             k2.transpose(-1, -2))  # attn2 matrix is equal to the transpose of attn1
        attn2 = F.softmax(attn2, dim=-1)
        output2 = torch.matmul(attn2, v2)
        # [B, n, N, C//n] -> [B, N, n, C//n] -> [B, N, C]
        output2 = output2.transpose(1, 2).contiguous().flatten(2)

        output1 = self.layer_norm1(output1 + f_a)
        output2 = self.layer_norm2(output2 + f_s)
        return output1, output2


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            init.constant_(m.bias, 0.0)


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


class CloNet50(nn.Module):
    def __init__(self, config, num_identities, **kwargs):
        super().__init__()

        resnet50 = torchvision.models.resnet50(pretrained=True)
        if config.MODEL.RES4_STRIDE == 1:
            resnet50.layer4[0].conv2.stride=(1, 1)
            resnet50.layer4[0].downsample[0].stride=(1, 1)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        if config.MODEL.POOLING.NAME == 'avg':
            self.globalpooling = nn.AdaptiveAvgPool2d(1)
        elif config.MODEL.POOLING.NAME == 'max':
            self.globalpooling = nn.AdaptiveMaxPool2d(1)
        elif config.MODEL.POOLING.NAME == 'gem':
            self.globalpooling = pooling.GeMPooling(p=config.MODEL.POOLING.P)
        elif config.MODEL.POOLING.NAME == 'maxavg':
            self.globalpooling = pooling.MaxAvgPooling()
        else:
            raise KeyError("Invalid pooling: '{}'".format(config.MODEL.POOLING.NAME))
        self.bn = nn.BatchNorm1d(config.MODEL.FEATURE_DIM)
        init.normal_(self.bn.weight.data, 1.0, 0.02)
        init.constant_(self.bn.bias.data, 0.0)

    def forward(self, x):
        x = self.base(x)
        y_fore = self.globalpooling(x)
        y_fore = y_fore.view(y_fore.size(0), -1)
        y_clo = self.bn(y_fore)
        return y_clo