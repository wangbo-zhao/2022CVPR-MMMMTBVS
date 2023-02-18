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
import json
from torch.utils.data import DataLoader
from dataset.utils import ResizeAndPad

class YoutubevosTrainDataset(Dataset):

    def __init__(self, args, root, max_jump=1):
        self.db_root_dir = root
        self.max_jump = max_jump
        data_root = os.path.join(self.db_root_dir, 'meta_expressions/train/meta_expressions.json')
        self.tokenizer = transformers.BertTokenizer.from_pretrained(args["bert_tokenizer"])
        max_tokens = 20

        with open(data_root, "r") as f:
            ref_data = json.load(f)


        self.objects_list = []


        for sequence_name, _ in ref_data["videos"].items():

            frames_list = sorted(ref_data["videos"][sequence_name]["frames"])
            expression = ref_data["videos"][sequence_name]["expressions"]
            num_objs = int(expression[str(len(expression) - 1)]["obj_id"])

            for i in range(num_objs): #the number of objects in this video sequence

                # sentences_for_ref = []
                # attentions_for_ref = []
                expression_nums = 0

                for k, v in expression.items(): #add all expressions of the current object
                    if (v["obj_id"]) == str(i + 1):
                        expression_nums = expression_nums + 1

                        sentence_raw = v["exp"]
                        sentence_raw_split = sentence_raw.split(" ")
                        sentence_raw_split = sentence_raw_split[:max_tokens - 2]
                        input_ids = self.tokenizer.encode(text=sentence_raw_split, add_special_tokens=True)

                        input_ids = input_ids[:max_tokens]
                        attention_mask = [0] * max_tokens
                        padded_input_ids = [0] * max_tokens

                        padded_input_ids[:len(input_ids)] = input_ids
                        attention_mask[:len(input_ids)] = [1] * len(input_ids)

                        self.objects_list.append(
                            {"name": sequence_name + "_" + str(i + 1) + "_" + str(expression_nums),
                             "sentences_list": torch.tensor(padded_input_ids).unsqueeze(0),
                             "attentions_list": torch.tensor(attention_mask).unsqueeze(0),
                             "frames_list": frames_list})


        self.pair_im_lone_transform = transforms.Compose([
            transforms.ColorJitter(0.1, 0.05, 0.05, 0), # No hue change here as that's not realistic
        ])

        self.pair_im_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=10, scale=(0.8, 1.2), resample=Image.BICUBIC, fillcolor=im_mean),
            ResizeAndPad(320, 320, Image.BICUBIC),
        ])

        self.pair_gt_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=10, scale=(0.8, 1.2), resample=Image.BICUBIC, fillcolor=0),
            ResizeAndPad(320, 320, Image.NEAREST),
        ])

        self.final_im_transform = transforms.Compose([
            transforms.ToTensor(),
            im_normalization,
        ])

        self.final_gt_transform = transforms.Compose([
            transforms.ToTensor(),
        ])


    def __getitem__(self, index):
        #object_info["name"].split("_")  [video name, object id, expression id]
        object_info = self.objects_list[index]
        sequence_name = object_info["name"].split("_")[0]
        obj_id = int(object_info["name"].split("_")[1])

        frames = object_info["frames_list"]

        while (len(frames) % 3 != 0):
            frames = frames + [frames[-1]]

        segment_nums = len(frames) // 3
        start_idx = np.random.randint(segment_nums)
        f1_idx = np.random.randint(segment_nums, 2 * segment_nums)
        f2_idx = np.random.randint(2 * segment_nums, 3 * segment_nums)

        frames_idx = [start_idx, f1_idx, f2_idx]
        # print(frames_idx)
        sequence_seed = np.random.randint(2147483647)

        images = []
        flows = []
        gts = []
        cls_gts = []
        tensor_embeddings = []
        attention_masks = []

        for f_idx in frames_idx:
            image_dir = os.path.join(self.db_root_dir, "train/JPEGImages", sequence_name, frames[f_idx] + ".jpg")
            flow_dir = os.path.join(self.db_root_dir, "train/FlowMaps", sequence_name, frames[f_idx] + ".jpg")
            label_dir = os.path.join(self.db_root_dir, "train/Annotations", sequence_name, frames[f_idx] + ".png")
            img = Image.open(image_dir).convert('RGB')
            flow = Image.open(flow_dir).convert("RGB")
            label = np.array(Image.open(label_dir))

            mask = (label == obj_id).astype(np.float32)
            mask[mask == 1] = 255
            gt = Image.fromarray(mask.astype(np.uint8), mode="L")

            reseed(sequence_seed)
            img = self.pair_im_dual_transform(img)
            img = self.pair_im_lone_transform(img)
            # img.save("img.png")

            reseed(sequence_seed)
            flow = self.pair_im_dual_transform(flow)
            flow = self.pair_im_lone_transform(flow)
            # flow.save("flow.png")

            reseed(sequence_seed)
            gt = self.pair_gt_dual_transform(gt)
            # gt.save("gt.png")

            img = self.final_im_transform(img)
            flow = self.final_im_transform(flow)
            gt = self.final_gt_transform(gt)

            cls_gt = np.zeros((gt.shape[1:]), dtype=np.int)
            cls_gt[gt.squeeze(0) > 0.5] = 1
            cls_gt = torch.tensor(cls_gt)

            tensor_embedding = object_info["sentences_list"]
            attention_mask = object_info["attentions_list"]

            images.append(img)
            flows.append(flow)
            gts.append(gt)
            cls_gts.append(cls_gt)
            tensor_embeddings.append(tensor_embedding)
            attention_masks.append(attention_mask)

        images = torch.stack(images, dim=0)
        flows = torch.stack(flows, dim=0)
        gts = torch.stack(gts, dim=0)
        cls_gts = torch.stack(cls_gts, dim=0)
        tensor_embeddings = torch.stack(tensor_embeddings, dim=0)
        attention_masks = torch.stack(attention_masks, dim=0)

        return images, flows, cls_gts, gts, tensor_embeddings, attention_masks

    def __len__(self):
        return len(self.objects_list)









class YoutubevosTestDataset(Dataset):

    def __init__(self, args, root):
        self.db_root_dir = root

        data_root = os.path.join(self.db_root_dir, 'meta_expressions/valid/meta_expressions.json')
        self.tokenizer = transformers.BertTokenizer.from_pretrained(args["bert_tokenizer"])
        max_tokens = 20

        with open(data_root, "r") as f:
            ref_data = json.load(f)


        val_list = os.listdir(os.path.join(self.db_root_dir, "valid", "JPEGImages")) # only use val set

        self.video_list = []
        for sequence_name, _ in ref_data["videos"].items():
            if sequence_name not in val_list:
                continue

            frames_list = ref_data["videos"][sequence_name]["frames"]
            expression = ref_data["videos"][sequence_name]["expressions"]

            sentences_list = []
            attentions_list = []
            sentences_raw = []

            for i in range(len(expression)):
                sentence_raw = expression[str(i)]["exp"]

                sentence_raw_split = sentence_raw.split(" ")
                sentence_raw_split = sentence_raw_split[:max_tokens - 2]

                input_ids = self.tokenizer.encode(text=sentence_raw_split, add_special_tokens=True)

                input_ids = input_ids[:max_tokens]
                attention_mask = [0] * max_tokens
                padded_input_ids = [0] * max_tokens

                padded_input_ids[:len(input_ids)] = input_ids
                attention_mask[:len(input_ids)] = [1] * len(input_ids)

                sentences_list.append(torch.tensor(padded_input_ids).unsqueeze(0))
                attentions_list.append(torch.tensor(attention_mask).unsqueeze(0))
                sentences_raw.append(sentence_raw)

            self.video_list.append({"name": sequence_name, "sentences_list": sentences_list,
                                      "attentions_list": attentions_list, "frames_list": frames_list, "sentences_raw": sentences_raw})

        self.im_transform = transforms.Compose([
            ResizeAndPad(320, 320, Image.BICUBIC),
            transforms.ToTensor(),
            im_normalization,
        ])



    def __getitem__(self, index):
        video = self.video_list[index]


        frames_list = video["frames_list"]

        images = []
        flows = []
        for frame_name in frames_list:
            image_dir = os.path.join(self.db_root_dir, "valid", "JPEGImages", video["name"], frame_name + ".jpg")
            images.append(self.im_transform(Image.open(image_dir).convert('RGB')))

            flow_dir = os.path.join(self.db_root_dir, "valid", "FlowMaps", video["name"], frame_name + ".jpg")
            flows.append(self.im_transform(Image.open(flow_dir).convert('RGB')))

        original_size = Image.open(image_dir).convert('RGB').size
        images = torch.stack(images, dim=0)
        flows = torch.stack(flows, dim=0)

        data = {
            "images": images,
            "flows": flows,
            "name": video["name"],
            "frames_name": frames_list,
            "sentences_list": video["sentences_list"],
            "attentions_list": video["attentions_list"],
            "sentences_raw": video["sentences_raw"],
            "original_size": original_size #(w, h)
        }

        return data






    def __len__(self):
        return len(self.video_list)





if __name__ == '__main__':
    from hyper_para import HyperParameters
    p = HyperParameters()
    p.parse()


    # train_dataset = YoutubevosTrainDataset(p, p["yv_root"])
    test_dataset = YoutubevosTestDataset(p, p["yv_root"])

    for i in test_dataset:
        print("A")