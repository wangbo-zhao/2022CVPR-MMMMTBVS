


import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
import numpy as np

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)
        self.embedding_project = nn.Sequential(nn.Linear(768, out_channels), nn.BatchNorm1d(out_channels))

        self.project = nn.Sequential(

            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))


    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))

        res = torch.cat(res, dim=1)
        visual_feats = self.project(res)

        return visual_feats



class LangugaeEmbeddingProject(nn.Module):
    def __init__(self, in_channels, out_channels, stride, kernel_size):
        super(LangugaeEmbeddingProject, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, stride=stride, kernel_size=kernel_size)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = rearrange(x, "n tokens c -> n c tokens")
        x = self.conv(x)
        x = self.bn(x)
        x = rearrange(x, " n c tokens -> n tokens c")
        return x


class VisionFeatureProject(nn.Module):
    def __init__(self, in_channels, out_channels, stride, kernel_size):
        super(VisionFeatureProject, self).__init__()
        self.project = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=stride, kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        x = self.project(x)
        return x



class ResBlock(nn.Module):
    def __init__(self, indim, outdim=None):
        super(ResBlock, self).__init__()
        if outdim == None:
            outdim = indim
        if indim == outdim:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(indim, outdim, kernel_size=3, padding=1)

        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)

    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))

        if self.downsample is not None:
            x = self.downsample(x)

        return x + r




class UpsampleBlock(nn.Module):
    def __init__(self, v_in,  f_in, out_c, scale_factor=2, spatial_size=20):
        super().__init__()
        self.vision_f = nn.Conv2d(in_channels=v_in, out_channels=out_c, kernel_size=1)
        self.enhance_vision_f =  nn.Conv2d(in_channels=2 * out_c + 8, out_channels=out_c, kernel_size=3, padding=1, stride=1)

        self.flow_f = nn.Conv2d(in_channels=f_in, out_channels=out_c, kernel_size=1)
        self.enhance_flow_f =  nn.Conv2d(in_channels=2 * out_c + 8, out_channels=out_c, kernel_size=3, padding=1, stride=1)

        self.language_f = nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=1)

        self.vision_to_attention = nn.Conv2d(in_channels=out_c, out_channels=1, kernel_size=1)
        self.flow_to_attention = nn.Conv2d(in_channels=out_c, out_channels=1, kernel_size=1)


        self.fuse = nn.Conv2d(in_channels=2 * out_c, out_channels=out_c, kernel_size=3, padding=1, stride=1)

        self.scale_factor = scale_factor
        self.out_conv = ResBlock(out_c, out_c)
        self.register_buffer('spatial_pos',  torch.tensor(self.generate_spatial_batch(spatial_size, spatial_size)))

    def forward(self, vision_feature, flow_feature, language_feature, up_f):
        vision_feature = self.vision_f(vision_feature)
        flow_feature = self.flow_f(flow_feature)
        language_feature = self.language_f(language_feature[:, 0, :].unsqueeze(2).unsqueeze(3))


        enhanced_vision_feature_source = language_feature * self.enhance_vision_f(torch.cat((vision_feature,
                                                                                      F.interpolate(up_f, scale_factor=self.scale_factor, mode='bilinear', align_corners=False),
                                                                                      self.spatial_pos.unsqueeze(0).repeat(vision_feature.shape[0], 1, 1,1)), dim=1))

        enhanced_flow_feature_source  = language_feature * self.enhance_flow_f(torch.cat((flow_feature,
                                                                                   F.interpolate(up_f, scale_factor=self.scale_factor, mode='bilinear', align_corners=False),
                                                                                   self.spatial_pos.unsqueeze(0).repeat(flow_feature.shape[0], 1, 1,1)), dim=1))

        vision_attention = torch.sigmoid(self.vision_to_attention(enhanced_vision_feature_source))
        flow_attention = torch.sigmoid(self.flow_to_attention(enhanced_vision_feature_source))

        enhanced_vision_feature = enhanced_vision_feature_source + flow_attention * enhanced_vision_feature_source
        enhanced_flow_feature = enhanced_flow_feature_source + vision_attention * enhanced_flow_feature_source

        final = self.out_conv(self.fuse(torch.cat((enhanced_vision_feature, enhanced_flow_feature), dim=1)))

        return enhanced_vision_feature_source, enhanced_flow_feature_source, final

    def generate_spatial_batch(self, featmap_H, featmap_W):

        spatial_batch_val = np.zeros((8, featmap_H, featmap_W), dtype=np.float32)
        for h in range(featmap_H):
            for w in range(featmap_W):
                xmin = w / featmap_W * 2 - 1
                xmax = (w + 1) / featmap_W * 2 - 1
                xctr = (xmin + xmax) / 2
                ymin = h / featmap_H * 2 - 1
                ymax = (h + 1) / featmap_H * 2 - 1
                yctr = (ymin + ymax) / 2
                spatial_batch_val[:, h, w] = \
                    [xmin, ymin, xmax, ymax, xctr, yctr, 1 / featmap_W, 1 / featmap_H]
        return spatial_batch_val

