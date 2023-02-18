import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import *
import random


class VisionLanguageAttention(nn.Module):

    def __init__(self, dim, heads=8, dim_head=64, dropout = 0., pooling=False):
        super(VisionLanguageAttention, self).__init__()

        self.heads = heads
        self.dim_head = dim_head
        self.scale = self.dim_head ** -0.5

        self.to_q = nn.Conv1d(in_channels=dim, out_channels=self.dim_head * self.heads, kernel_size=1, stride=1, bias=False)
        self.to_k = nn.Conv2d(in_channels=dim, out_channels=self.dim_head * self.heads, kernel_size=1, stride=1, bias=False)
        self.to_v = nn.Conv2d(in_channels=dim, out_channels=self.dim_head * self.heads, kernel_size=1, stride=1, bias=False)

        self.pooling = pooling
        if self.pooling:
            self.pooling_function = nn.MaxPool2d(kernel_size=(2, 2))

        self.to_out = nn.Conv1d(in_channels=self.dim_head * self.heads, out_channels=dim, kernel_size=1, stride=1, bias=False)

    def forward(self, vision_feature, language_feature, mask):

        language_feature = rearrange(language_feature, "b tokens c -> b c tokens")

        q = self.to_q(language_feature) # [b c tokens]
        k = self.to_k(vision_feature) # [b c h w]
        v = self.to_v(vision_feature) # [b c h w]

        if self.pooling:
            k = self.pooling_function(k)
            v = self.pooling_function(v)

        q = rearrange(q, "b (n c) tokens -> (n b) tokens c", n=self.heads)
        k = rearrange(k, "b (n c) h w -> (n b) c (h w)", n=self.heads)
        v = rearrange(v, "b (n c) h w -> (n b) (h w) c", n=self.heads)



        att = F.softmax(torch.matmul(q, k) * self.scale, dim=-1)
        out = torch.matmul(att, v) #[N tokens c]
        out = rearrange(out, '(n b) tokens c -> b (n c) tokens', n=self.heads)
        out = self.to_out(out)

        out = rearrange(out, 'b c tokens -> b tokens c')

        return out


class LanguageAttention(nn.Module):

    def __init__(self, dim, heads=8, dim_head=64, dropout=0., pooling=False):
        super(LanguageAttention, self).__init__()

        self.heads = heads
        self.dim_head = dim_head
        self.scale = self.dim_head ** -0.5

        self.to_q = nn.Conv1d(in_channels=dim, out_channels=self.dim_head * self.heads, kernel_size=1, stride=1, bias=False)
        self.to_k = nn.Conv1d(in_channels=dim, out_channels=self.dim_head * self.heads, kernel_size=1, stride=1, bias=False)
        self.to_v = nn.Conv1d(in_channels=dim, out_channels=self.dim_head * self.heads, kernel_size=1, stride=1, bias=False)

        self.to_out = nn.Conv1d(in_channels=self.dim_head * self.heads, out_channels=dim, kernel_size=1, stride=1, bias=False)

    def forward(self, language_feature, mask):

        language_feature = rearrange(language_feature, "b tokens c -> b c tokens")

        q = self.to_q(language_feature) # [b c tokens]
        k = self.to_k(language_feature) # [b c tokens]
        v = self.to_v(language_feature) # [b c tokens]

        q = rearrange(q, "b (n c) tokens -> (n b) tokens c", n=self.heads)
        k = rearrange(k, "b (n c) tokens -> (n b) c tokens", n=self.heads)
        v = rearrange(v, "b (n c) tokens -> (n b) tokens c", n=self.heads)

        scores = torch.matmul(q, k) * self.scale
        mask = mask.repeat((self.heads, 1)).unsqueeze(1)
        scores = scores.masked_fill(mask == 0, 1e-9)

        att = F.softmax(scores, dim=-1)
        out = torch.matmul(att, v) #[N tokens c]
        out = rearrange(out, '(n b) tokens c -> b (n c) tokens', n=self.heads)
        out = self.to_out(out)

        out = rearrange(out, 'b c tokens -> b tokens c')

        return out


class SpatialTemporalAttention(nn.Module):

    def __init__(self, dim, heads=8, dim_head=64, dropout = 0., pooling=False):
        super(SpatialTemporalAttention, self).__init__()

        self.heads = heads
        self.dim_head = dim_head
        self.scale = self.dim_head ** -0.5

        self.to_q = nn.Conv3d(in_channels=dim, out_channels=self.dim_head * self.heads, kernel_size=1, stride=1, bias=False)
        self.to_k = nn.Conv3d(in_channels=dim, out_channels=self.dim_head * self.heads, kernel_size=1, stride=1, bias=False)
        self.to_v = nn.Conv3d(in_channels=dim, out_channels=self.dim_head * self.heads, kernel_size=1, stride=1, bias=False)

        self.pooling = pooling
        if self.pooling:
            self.pooling_function = nn.MaxPool3d(kernel_size=(1, 2, 2))

        self.to_out = nn.Conv3d(in_channels=self.dim_head * self.heads, out_channels=dim, kernel_size=1, stride=1, bias=False)

    def forward(self, vision_feature):
        t, h, w = vision_feature.shape[1:4]
        vision_feature = rearrange(vision_feature, "b t h w c -> b c t h w")

        q = self.to_q(vision_feature) # [b t c h w]
        k = self.to_k(vision_feature) # [b t c h w]
        v = self.to_v(vision_feature) # [b t c h w]

        if self.pooling:
            k = self.pooling_function(k)
            v = self.pooling_function(v)

        q = rearrange(q, "b (n c) t h w -> (b n) (t h w) c", n=self.heads)
        k = rearrange(k, "b (n c) t h w -> (b n) c (t h w)", n=self.heads)
        v = rearrange(v, "b (n c) t h w -> (b n) (t h w) c", n=self.heads)



        att = F.softmax(torch.matmul(q, k) * self.scale, dim=-1)
        out = torch.matmul(att, v)
        out = rearrange(out, '(b n) (t h w) c -> b (n c) t h w', n=self.heads, t=t, h=h, w=w)
        out = self.to_out(out)

        out = rearrange(out, 'b c t h w -> b t h w c')

        return out



class CrossModalAttention(nn.Module):

    def __init__(self, dim, heads=8, dim_head=64, dropout=0., pooling=False):
        super(CrossModalAttention, self).__init__()

        self.heads = heads
        self.dim_head = dim_head
        self.scale = self.dim_head ** -0.5

        self.to_q = nn.Conv1d(in_channels=dim, out_channels=self.dim_head * self.heads, kernel_size=1, stride=1, bias=False)
        self.to_k = nn.Conv1d(in_channels=dim, out_channels=self.dim_head * self.heads, kernel_size=1, stride=1, bias=False)
        self.to_v = nn.Conv1d(in_channels=dim, out_channels=self.dim_head * self.heads, kernel_size=1, stride=1, bias=False)

        self.to_out = nn.Conv1d(in_channels=self.dim_head * self.heads, out_channels=dim, kernel_size=1, stride=1, bias=False)

    def forward(self, input_feature, mask):

        feature = rearrange(input_feature, "b tokens c -> b c tokens")

        q = self.to_q(feature)  # [b c tokens]
        k = self.to_k(feature)  # [b c tokens]
        v = self.to_v(feature)  # [b c tokens]

        q = rearrange(q, "b (n c) tokens -> (n b) tokens c", n=self.heads)
        k = rearrange(k, "b (n c) tokens -> (n b) c tokens", n=self.heads)
        v = rearrange(v, "b (n c) tokens -> (n b) tokens c", n=self.heads)

        scores = torch.matmul(q, k) * self.scale
        mask = mask.repeat((self.heads, 1)).unsqueeze(1)
        scores = scores.masked_fill(mask == 0, 1e-9)

        att = F.softmax(scores, dim=-1)
        out = torch.matmul(att, v) #[N tokens c]
        out = rearrange(out, '(n b) tokens c -> b (n c) tokens', n=self.heads)
        out = self.to_out(out)

        out = rearrange(out, 'b c tokens -> b tokens c')

        return out


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)



class TemporalAttention(nn.Module):

    def __init__(self, dim, heads=8, dim_head=64, dropout=0., pooling=False):
        super(TemporalAttention, self).__init__()

        self.heads = heads
        self.dim_head = dim_head
        self.scale = self.dim_head ** -0.5

        self.to_q = nn.Conv1d(in_channels=dim, out_channels=self.dim_head * self.heads, kernel_size=1, stride=1, bias=False)
        self.to_k = nn.Conv1d(in_channels=dim, out_channels=self.dim_head * self.heads, kernel_size=1, stride=1, bias=False)
        self.to_v = nn.Conv1d(in_channels=dim, out_channels=self.dim_head * self.heads, kernel_size=1, stride=1, bias=False)

        self.to_out = nn.Conv1d(in_channels=self.dim_head * self.heads, out_channels=dim, kernel_size=1, stride=1, bias=False)

    def forward(self, vision_feature, sequence_length):
        feature = rearrange(vision_feature, "(b t) tokens c -> b c (t tokens)", t=sequence_length)


        q = self.to_q(feature)  # [b c tokens]
        k = self.to_k(feature)  # [b c tokens]
        v = self.to_v(feature)  # [b c tokens]

        q = rearrange(q, "b (n c) tokens -> (n b) tokens c", n=self.heads)
        k = rearrange(k, "b (n c) tokens -> (n b) c tokens", n=self.heads)
        v = rearrange(v, "b (n c) tokens -> (n b) tokens c", n=self.heads)

        scores = torch.matmul(q, k) * self.scale

        att = F.softmax(scores, dim=-1)
        out = torch.matmul(att, v) #[N tokens c]
        out = rearrange(out, '(n b) tokens c -> b (n c) tokens', n=self.heads)
        out = self.to_out(out)

        out = rearrange(out, 'b c (t tokens) -> (b t) tokens c', t=sequence_length)

        return out






class TransformerEncoder(nn.Module):
    def __init__(self, depth, dim, heads, dim_head, mlp_dim, dropout = 0., pooling=False):
        super(TransformerEncoder, self).__init__()

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                CrossModalAttention(dim=dim, heads=heads, dim_head=dim_head, dropout=dropout, pooling=pooling),
                nn.LayerNorm(dim),
                TemporalAttention(dim=dim, heads=heads, dim_head=dim_head, dropout=dropout, pooling=pooling),
                nn.LayerNorm(dim),
                FeedForward(dim=dim, hidden_dim=mlp_dim, dropout=dropout)
            ]))


    def forward(self, input_feature, mask, sequence_length):

        x = input_feature  # n t h w c


        for \
            norm1, visionlanguageattention, \
            norm2, temporalattention, \
            norm3, MLP \
                in self.layers:

            x = visionlanguageattention(norm1(x), mask) + x


            vision_feature = x[:, 20:420, :]
            vision_feature = temporalattention(norm2(vision_feature), sequence_length) + vision_feature
            x = torch.cat((x[:, :20, :], vision_feature, x[:, 420:, :]), dim=1)


            x = MLP(norm3(x)) + x

        return x