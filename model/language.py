
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import *

from einops import rearrange

class LanguageModel(nn.Module):
    def __init__(self, args):
        super(LanguageModel, self).__init__()

        self.Languagemodel = BertModel.from_pretrained(args["ck_bert"])

    def forward(self, sentences, attentions):

        language_feature = self.Languagemodel(sentences, attention_mask=attentions)[0]

        return language_feature







