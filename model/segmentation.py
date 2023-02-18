import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from .vision_language import TransformerEncoder
from .utils import LangugaeEmbeddingProject, VisionFeatureProject, UpsampleBlock
import numpy as np
from .utils import ASPP
import random

class Contrast(nn.Module):
    def __init__(self):
        super(Contrast, self).__init__()
        self.vision_transform = nn.Linear(256, 256)
        self.language_transform = nn.Linear(256, 256)

    def forward(self, vision_feature, language_feature):
        vision_feature = rearrange(vision_feature, "n c h w -> n (h w) c")
        language_feature = language_feature[:, 0, :].unsqueeze(1) # n 1 c  here we use cls token


        vision_feature = self.vision_transform(vision_feature)
        language_feature = self.language_transform(language_feature)

        language_feature = rearrange(language_feature, "n tokens c-> n c tokens")

        map = torch.matmul(vision_feature, language_feature) # (hw tokens)

        return map




class LanguageAttention(nn.Module):
    def __init__(self):
        super(LanguageAttention, self).__init__()
        out_channels = 256
        out_channels_multimodal = 256


        self.transformer = TransformerEncoder(depth=4, dim=256, heads=4, dim_head=64, mlp_dim=128, dropout=0.1, pooling=False)



    def forward(self, vision_feature, flow_feature, language_feature, language_mask, sequence_length):
        h, w = vision_feature.shape[2:]

        vision_feature = rearrange(vision_feature, "n c h w -> n (h w) c")
        vision_mask = torch.ones(vision_feature.shape[:2], device=vision_feature.device)

        flow_feature = rearrange(flow_feature, "n c h w -> n (h w) c")
        flow_mask = torch.ones(flow_feature.shape[:2], device=flow_feature.device)


        mask = torch.cat((language_mask, vision_mask, flow_mask), dim=1)

        input_feature = torch.cat((language_feature, vision_feature, flow_feature), dim=1)

        output_feature = self.transformer(input_feature, mask, sequence_length)

        output_vision_feature = output_feature[:, 20:420, :]
        output_vision_feature = rearrange(output_vision_feature, "n (h w) c -> n c h w", h=h, w=w)

        return output_vision_feature


class FuseFeature(nn.Module):
    def __init__(self):
        super(FuseFeature, self).__init__()
        self.f_16 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=128, stride=1, kernel_size=1), nn.ReLU())
        self.f_8 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=128, stride=1, kernel_size=1), nn.ReLU())
        self.f_4 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=128, stride=1, kernel_size=1), nn.ReLU())



    def forward(self, feature_16, feature_8, feature_4):

        feature = torch.cat((F.interpolate(self.f_16(feature_16), scale_factor=4, mode='bilinear', align_corners=False),
                             F.interpolate(self.f_8(feature_8), scale_factor=2, mode='bilinear', align_corners=False),
                             self.f_4(feature_4)), dim=1)

        return feature




class LanguageGuidedSegmentation(nn.Module):
    def __init__(self):
        super(LanguageGuidedSegmentation, self).__init__()

        self.language_project = LangugaeEmbeddingProject(in_channels=768, out_channels=256, stride=1, kernel_size=1)

        self.aspp = ASPP(in_channels=2048, out_channels=256, atrous_rates=[3, 5, 7])
        self.flow_aspp = ASPP(in_channels=512, out_channels=256, atrous_rates=[3, 5, 7])

        self.up_16_16 = UpsampleBlock(v_in=1024, f_in=256, out_c=256, scale_factor=1, spatial_size=20)
        self.up_16_8 = UpsampleBlock(v_in=512, f_in=128, out_c=256, scale_factor=2, spatial_size=40)
        self.up_8_4 = UpsampleBlock(v_in=256, f_in=64, out_c=256, scale_factor=2, spatial_size=80)

        self.fuse_feature_post_rgb = nn.Sequential(nn.Conv2d(in_channels=264, out_channels=256, stride=1, kernel_size=3, padding=1),
                                              nn.ReLU(),
                                              nn.Conv2d(in_channels=256, out_channels=256, stride=1, kernel_size=1))
        self.fuse_feature_post_flow = nn.Sequential(nn.Conv2d(in_channels=264, out_channels=256, stride=1, kernel_size=3, padding=1),
                                              nn.ReLU(),
                                              nn.Conv2d(in_channels=256, out_channels=256, stride=1, kernel_size=1))

        self.fuse_rgb = FuseFeature()
        self.fuse_flow = FuseFeature()

        self.language_guided_segmentation = LanguageAttention()

        self.conv = nn.Conv2d(256, 256, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
        self.conv_2 = nn.Conv2d(256, 1, 1)
        self.register_buffer('spatial_pos',  torch.tensor(self.generate_spatial_batch(20, 20)))

    def forward(self, vision_feature, flow_feature, language_feature, language_mask, input_shape, sequence_length):
        language_feature = self.language_project(language_feature)

        flow_feature4 = self.flow_aspp(flow_feature["layer4"])
        flow_feature3 = flow_feature["layer3"]
        flow_feature2 = flow_feature["layer2"]
        flow_feature1 = flow_feature["layer1"]

        vision_feature4 = self.aspp(vision_feature["layer4"])
        vision_feature3 = vision_feature["layer3"]
        vision_feature2 = vision_feature["layer2"]
        vision_feature1 = vision_feature["layer1"]

        vision_feature = self.fuse_feature_post_rgb(
            torch.cat((vision_feature4, self.spatial_pos.unsqueeze(0).repeat(vision_feature4.shape[0], 1, 1, 1)), dim=1))

        flow_feature = self.fuse_feature_post_flow(
            torch.cat((flow_feature4, self.spatial_pos.unsqueeze(0).repeat(flow_feature4.shape[0], 1, 1, 1)), dim=1))


        feature = self.language_guided_segmentation(vision_feature, flow_feature, language_feature, language_mask, sequence_length)

        # here we can only use index=-1 to 减少计算加快训练.因为其实只有index=-1的时候有标注
        enhanced_vision_feature_16, enhanced_flow_feature_16, decoder16 = self.up_16_16(vision_feature3[2::3, :, :, :], flow_feature3[2::3, :, :, :], language_feature[2::3, :, :], feature[2::3, :, :, :])
        enhanced_vision_feature_8, enhanced_flow_feature_8, decoder8 = self.up_16_8(vision_feature2[2::3, :, :, :], flow_feature2[2::3, :, :, :], language_feature[2::3, :, :], decoder16)
        enhanced_vision_feature_4, enhanced_flow_feature_4, decoder4 = self.up_8_4(vision_feature1[2::3, :, :, :], flow_feature1[2::3, :, :, :], language_feature[2::3, :, :], decoder8)

        fused_vision = self.fuse_rgb(enhanced_vision_feature_16, enhanced_vision_feature_8, enhanced_vision_feature_4)
        fused_flow = self.fuse_flow(enhanced_flow_feature_16, enhanced_flow_feature_8, enhanced_flow_feature_4)

        res = self.conv(decoder4)
        res = self.bn(res)
        res = self.relu(res)
        res = self.conv_2(res)

        res = F.interpolate(res, size=input_shape, mode='bilinear', align_corners=False)

        return fused_vision, fused_flow, res

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

