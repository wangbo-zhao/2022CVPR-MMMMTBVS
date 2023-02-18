import os
from os import path

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import transformers
from dataset.utils import im_normalization, im_mean
from dataset.utils import reseed
import random
from refer.refer import REFER
from .utils import ResizeAndPad


class RefCOCOTrainDataset(Dataset):
    """
    Generate pseudo VOS data by applying random transforms on static images.
    Single-object only.

    Method 0 - FSS style (class/1.jpg class/1.png)
    Method 1 - Others style (XXX.jpg XXX.png)
    """
    def __init__(self, args, root, split="train"):
        self.root = root

        self.refer = REFER(root, args['refcoco_dataset'], args['splitBy'])
        self.max_tokens = 20
        self.split = split
        ref_ids = self.refer.getRefIds(split=self.split)
        img_ids = self.refer.getImgIds(ref_ids)
        all_imgs = self.refer.Imgs
        self.imgs = list(all_imgs[i] for i in img_ids)
        self.ref_ids = ref_ids
        self.input_ids = []
        self.attention_masks = []
        self.tokenizer = transformers.BertTokenizer.from_pretrained(args['bert_tokenizer'])

        pad_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('[PAD]'))

        for r in ref_ids:
            ref = self.refer.Refs[r]

            sentences_for_ref = []
            attentions_for_ref = []

            for i, (el, sent_id) in enumerate(zip(ref['sentences'], ref['sent_ids'])):
                sentence_raw = el['sent']


                attention_mask = [0] * self.max_tokens
                padded_input_ids = [0] * self.max_tokens

                input_ids = self.tokenizer.encode(text=sentence_raw, add_special_tokens=True)


                input_ids = input_ids[:self.max_tokens]

                padded_input_ids[:len(input_ids)] = input_ids
                attention_mask[:len(input_ids)] = [1] * len(input_ids)

                sentences_for_ref.append(torch.tensor(padded_input_ids).unsqueeze(0))
                attentions_for_ref.append(torch.tensor(attention_mask).unsqueeze(0))

            self.input_ids.append(sentences_for_ref)
            self.attention_masks.append(attentions_for_ref)


        self.pair_im_lone_transform = transforms.Compose([
            transforms.ColorJitter(0.1, 0.05, 0.05, 0), # No hue change here as that's not realistic
        ])

        self.pair_im_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=0, scale=(0.9, 1.1), resample=Image.BICUBIC, fillcolor=im_mean),
            ResizeAndPad(320, 320, Image.BICUBIC)
            # transforms.Resize(320, Image.BICUBIC),
            # transforms.RandomCrop((320, 320), pad_if_needed=True, fill=im_mean),
        ])

        self.pair_gt_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=0, scale=(0.9, 1.1), resample=Image.BICUBIC, fillcolor=0),
            ResizeAndPad(320, 320, Image.NEAREST)
            # transforms.Resize(320, Image.NEAREST),
            # transforms.RandomCrop((320, 320), pad_if_needed=True, fill=0),
        ])

        self.final_im_transform = transforms.Compose([
            transforms.ToTensor(),
            im_normalization,
        ])

        self.final_gt_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        this_ref_id = self.ref_ids[index]
        this_img_id = self.refer.getImgIds(this_ref_id)
        this_img = self.refer.Imgs[this_img_id[0]]

        img = Image.open(os.path.join(self.refer.IMAGE_DIR, this_img['file_name'])).convert("RGB")
        ref = self.refer.loadRefs(this_ref_id)
        this_sent_ids = ref[0]['sent_ids']

        ref_mask = np.array(self.refer.getMask(ref[0])['mask'])
        annot = np.zeros(ref_mask.shape)
        annot[ref_mask == 1] = 255

        gt = Image.fromarray(annot.astype(np.uint8), mode="L")

        pairwise_seed = np.random.randint(2147483647)
        reseed(pairwise_seed)
        img = self.pair_im_dual_transform(img)
        img = self.pair_im_lone_transform(img)
        reseed(pairwise_seed)
        gt = self.pair_gt_dual_transform(gt)

        img = self.final_im_transform(img)
        gt = self.final_gt_transform(gt)

        cls_gt = np.zeros((gt.shape[1:]), dtype=np.int)
        cls_gt[gt.squeeze(0) > 0.5] = 1
        cls_gt = torch.tensor(cls_gt)


        choice_sent = np.random.choice(len(self.input_ids[index]))
        tensor_embeddings = self.input_ids[index][choice_sent]
        attention_mask = self.attention_masks[index][choice_sent]

        random_remove = random.randint(1, attention_mask.sum() - 2)


        length_tokens = tensor_embeddings.shape[1]
        tensor_embeddings_remove = torch.zeros(tensor_embeddings.shape, device=tensor_embeddings.device, dtype=tensor_embeddings.dtype)
        attention_mask_remove = torch.zeros(attention_mask.shape, device=attention_mask.device, dtype=attention_mask.dtype)

        tensor_embeddings_remove[:, :length_tokens - 1] = torch.index_select(tensor_embeddings, dim=1, index=torch.tensor([i for i in range(length_tokens) if i!= random_remove],
                                                                                         dtype=torch.long, device=tensor_embeddings.device))

        attention_mask_remove[:, :length_tokens - 1] = torch.index_select(attention_mask, dim=1, index=torch.tensor([i for i in range(length_tokens) if i!= random_remove],
                                                                                         dtype=torch.long, device=attention_mask.device))



        return img, cls_gt, gt, tensor_embeddings, attention_mask, tensor_embeddings_remove, attention_mask_remove



    def __len__(self):
        return len(self.ref_ids)












class RefCOCOTestDataset(Dataset):
    """
    Generate pseudo VOS data by applying random transforms on static images.
    Single-object only.

    Method 0 - FSS style (class/1.jpg class/1.png)
    Method 1 - Others style (XXX.jpg XXX.png)
    """
    def __init__(self, args, root, split="train"):
        self.root = root

        self.refer = REFER(root, args['refcoco_dataset'], args['splitBy'])
        self.max_tokens = 20
        self.split = split
        ref_ids = self.refer.getRefIds(split=self.split)
        img_ids = self.refer.getImgIds(ref_ids)
        all_imgs = self.refer.Imgs
        self.imgs = list(all_imgs[i] for i in img_ids)
        self.ref_ids = ref_ids
        self.input_ids = []
        self.attention_masks = []
        self.tokenizer = transformers.BertTokenizer.from_pretrained(args['bert_tokenizer'])

        pad_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('[PAD]'))

        for r in ref_ids:
            ref = self.refer.Refs[r]

            sentences_for_ref = []
            attentions_for_ref = []

            for i, (el, sent_id) in enumerate(zip(ref['sentences'], ref['sent_ids'])):
                sentence_raw = el['sent']
                attention_mask = [0] * self.max_tokens
                padded_input_ids = [0] * self.max_tokens

                input_ids = self.tokenizer.encode(text=sentence_raw, add_special_tokens=True)

                # truncation of tokens
                input_ids = input_ids[:self.max_tokens]

                padded_input_ids[:len(input_ids)] = input_ids
                attention_mask[:len(input_ids)] = [1] * len(input_ids)

                sentences_for_ref.append(torch.tensor(padded_input_ids).unsqueeze(0))
                attentions_for_ref.append(torch.tensor(attention_mask).unsqueeze(0))

            self.input_ids.append(sentences_for_ref)
            self.attention_masks.append(attentions_for_ref)

            self.im_transform = transforms.Compose([
                # transforms.Resize(320, Image.BICUBIC),
                ResizeAndPad(320, 320, Image.BICUBIC),
                transforms.ToTensor(),
                im_normalization,
            ])

            self.gt_transform = transforms.Compose([
                transforms.ToTensor(),
            ])



    def __getitem__(self, index):
        this_ref_id = self.ref_ids[index]
        this_img_id = self.refer.getImgIds(this_ref_id)
        this_img = self.refer.Imgs[this_img_id[0]]

        img = Image.open(os.path.join(self.refer.IMAGE_DIR, this_img['file_name'])).convert("RGB")

        ref = self.refer.loadRefs(this_ref_id)
        this_sent_ids = ref[0]['sent_ids']

        ref_mask = np.array(self.refer.getMask(ref[0])['mask'])
        annot = np.zeros(ref_mask.shape)
        annot[ref_mask == 1] = 255

        gt = Image.fromarray(annot.astype(np.uint8), mode="L")


        img = self.im_transform(img)
        gt = self.gt_transform(gt)


        embedding = []
        att = []
        for s in range(len(self.input_ids[index])):
            e = self.input_ids[index][s]
            a = self.attention_masks[index][s]
            embedding.append(e.unsqueeze(-1))
            att.append(a.unsqueeze(-1))

        tensor_embeddings = torch.cat(embedding, dim=-1)
        attention_mask = torch.cat(att, dim=-1)

        return img, gt, tensor_embeddings, attention_mask



    def __len__(self):
        return len(self.ref_ids)


