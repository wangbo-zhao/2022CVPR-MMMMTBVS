
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from transformers import *
from .vision import VisionModel, ResNet34
from .language import LanguageModel
from .segmentation import LanguageGuidedSegmentation
from einops import rearrange

class ContrastiveModule(nn.Module):
    def __init__(self):
        super(ContrastiveModule, self).__init__()
        self.vision_proj = nn.Conv2d(in_channels=128 * 3, out_channels=128, stride=1, kernel_size=1)

        self.flow_proj = nn.Conv2d(in_channels=128 * 3, out_channels=128, stride=1, kernel_size=1)

        self.language_proj = nn.Conv2d(in_channels=768, out_channels=128, stride=1, kernel_size=1)



    def forward(self, vision, flow, language, cls_gt):
        #cls_gt [n, 1, 320, 320]
        h, w = vision.shape[2:]
        batch_size = vision.shape[0]
        vision_embedding = self.vision_proj(vision)
        flow_embedding = self.flow_proj(flow)
        language_embedding = self.language_proj(language)

        vision_embedding = rearrange(vision_embedding, "n c h w -> n (h w) c")
        flow_embedding = rearrange(flow_embedding, "n c h w -> n (h w) c")
        language_embedding = rearrange(language_embedding, "n c h w -> n c (h w)")

        vision_embedding = F.normalize(vision_embedding, dim=2)
        flow_embedding = F.normalize(flow_embedding, dim=2)
        language_embedding = F.normalize(language_embedding, dim=1)

        vision_map = torch.matmul(vision_embedding, language_embedding)
        flow_map = torch.matmul(flow_embedding, language_embedding)
        vision_flow_map = torch.matmul(vision_embedding, flow_embedding.permute(0, 2, 1))

        vision_map = rearrange(vision_map, "n (h w) c -> n c h w", h=h, w=w)
        flow_map = rearrange(flow_map, "n (h w) c -> n c h w", h=h, w=w)

        vision_map = torch.tan(vision_map * 3.141592 / 2)
        flow_map = torch.tan(flow_map * 3.141592 / 2)
        vision_flow_map = torch.tan(vision_flow_map * 3.141592 / 2)

        return vision_map, flow_map, vision_flow_map



class PropagationNetwork(nn.Module):
    def __init__(self, args):
        super(PropagationNetwork, self).__init__()
        self.model = VisionModel(backbone_name="resnet101", pretrained_dir=args["pretrained_dir"])
        self.Visionmodel = self.model
        self.Flowmodel = ResNet34(nInputChannels=3, os=16, pretrained=True, model_path=args["flow_pretrained_dir"])

        self.Languagemodel = LanguageModel(args)
        self.LanguagemodelGuidedSegmentation = LanguageGuidedSegmentation()

        self.contrastive = ContrastiveModule()


    def forward(self, image, flow, sentences, attentions, cls_gt):

        input_shape = image.shape[-2:]

        length = image.shape[1]
        image = rearrange(image,  "n t c h w -> (n t) c h w")
        flow = rearrange(flow, "n t c h w -> (n t) c h w")
        sentences = rearrange(sentences, "n t c tokens -> (n t) c tokens").squeeze(1)
        attentions = rearrange(attentions, "n t c tokens -> (n t) c tokens").squeeze(1)

        language_feature = self.Languagemodel(sentences=sentences, attentions=attentions)
        vision_feature = self.Visionmodel(image)
        flow_feature = self.Flowmodel(flow)

        fused_vision, fused_flow, output = self.LanguagemodelGuidedSegmentation(vision_feature, flow_feature, language_feature, attentions, input_shape, length)
        language_feature = language_feature[2::3, 0, :].unsqueeze(2).unsqueeze(3)

        vision_map, flow_map, vision_flow_map = self.contrastive(fused_vision, fused_flow, language_feature, cls_gt)

        logits = output
        prob = torch.sigmoid(logits)

        return vision_map, flow_map, vision_flow_map, logits, prob

    def inference(self, image, flow, sentences, attentions):

        input_shape = image.shape[-2:]
        length = image.shape[1]

        image = rearrange(image,  "n t c h w -> (n t) c h w")

        flow = rearrange(flow, "n t c h w -> (n t) c h w")

        sentences = rearrange(sentences, "n t c tokens -> (n t) c tokens").squeeze(1)
        attentions = rearrange(attentions, "n t c tokens -> (n t) c tokens").squeeze(1)

        language_feature = self.Languagemodel(sentences=sentences, attentions=attentions)
        vision_feature = self.Visionmodel(image)
        flow_feature = self.Flowmodel(flow)

        _, _, output = self.LanguagemodelGuidedSegmentation(vision_feature, flow_feature, language_feature, attentions, input_shape, length)

        return output













if __name__ == '__main__':
    from hyper_para import HyperParameters

    args = HyperParameters()
    args.parse()



    image = torch.rand([2, 3, 320, 320])
    sentences = torch.randint(200, [2, 20])
    attentions = torch.randint(2, [2, 20])

    sentences_removed = torch.randint(200, [2, 20])
    attentions_removed = torch.randint(2, [2, 20])


    model = PropagationNetwork(args)