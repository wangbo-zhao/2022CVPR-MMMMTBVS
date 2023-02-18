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
import h5py
import numpy as np




palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128, 64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128, 128, 64, 128, 0, 192, 128, 128, 192, 128, 64, 64, 0, 192, 64, 0, 64, 192, 0, 192, 192, 0, 64, 64, 128, 192, 64, 128, 64, 192, 128, 192, 192, 128, 0, 0, 64, 128, 0, 64, 0, 128, 64, 128, 128, 64, 0, 0, 192, 128, 0, 192, 0, 128, 192, 128, 128, 192, 64, 0, 64, 192, 0, 64, 64, 128, 64, 192, 128, 64, 64, 0, 192, 192, 0, 192, 64, 128, 192, 192, 128, 192, 0, 64, 64, 128, 64, 64, 0, 192, 64, 128, 192, 64, 0, 64, 192, 128, 64, 192, 0, 192, 192, 128, 192, 192, 64, 64, 64, 192, 64, 64, 64, 192, 64, 192, 192, 64, 64, 64, 192, 192, 64, 192, 64, 192, 192, 192, 192, 192, 32, 0, 0, 160, 0, 0, 32, 128, 0, 160, 128, 0, 32, 0, 128, 160, 0, 128, 32, 128, 128, 160, 128, 128, 96, 0, 0, 224, 0, 0, 96, 128, 0, 224, 128, 0, 96, 0, 128, 224, 0, 128, 96, 128, 128, 224, 128, 128, 32, 64, 0, 160, 64, 0, 32, 192, 0, 160, 192, 0, 32, 64, 128, 160, 64, 128, 32, 192, 128, 160, 192, 128, 96, 64, 0, 224, 64, 0, 96, 192, 0, 224, 192, 0, 96, 64, 128, 224, 64, 128, 96, 192, 128, 224, 192, 128, 32, 0, 64, 160, 0, 64, 32, 128, 64, 160, 128, 64, 32, 0, 192, 160, 0, 192, 32, 128, 192, 160, 128, 192, 96, 0, 64, 224, 0, 64, 96, 128, 64, 224, 128, 64, 96, 0, 192, 224, 0, 192, 96, 128, 192, 224, 128, 192, 32, 64, 64, 160, 64, 64, 32, 192, 64, 160, 192, 64, 32, 64, 192, 160, 64, 192, 32, 192, 192, 160, 192, 192, 96, 64, 64, 224, 64, 64, 96, 192, 64, 224, 192, 64, 96, 64, 192, 224, 64, 192, 96, 192, 192, 224, 192, 192, 0, 32, 0, 128, 32, 0, 0, 160, 0, 128, 160, 0, 0, 32, 128, 128, 32, 128, 0, 160, 128, 128, 160, 128, 64, 32, 0, 192, 32, 0, 64, 160, 0, 192, 160, 0, 64, 32, 128, 192, 32, 128, 64, 160, 128, 192, 160, 128, 0, 96, 0, 128, 96, 0, 0, 224, 0, 128, 224, 0, 0, 96, 128, 128, 96, 128, 0, 224, 128, 128, 224, 128, 64, 96, 0, 192, 96, 0, 64, 224, 0, 192, 224, 0, 64, 96, 128, 192, 96, 128, 64, 224, 128, 192, 224, 128, 0, 32, 64, 128, 32, 64, 0, 160, 64, 128, 160, 64, 0, 32, 192, 128, 32, 192, 0, 160, 192, 128, 160, 192, 64, 32, 64, 192, 32, 64, 64, 160, 64, 192, 160, 64, 64, 32, 192, 192, 32, 192, 64, 160, 192, 192, 160, 192, 0, 96, 64, 128, 96, 64, 0, 224, 64, 128, 224, 64, 0, 96, 192, 128, 96, 192, 0, 224, 192, 128, 224, 192, 64, 96, 64, 192, 96, 64, 64, 224, 64, 192, 224, 64, 64, 96, 192, 192, 96, 192, 64, 224, 192, 192, 224, 192, 32, 32, 0, 160, 32, 0, 32, 160, 0, 160, 160, 0, 32, 32, 128, 160, 32, 128, 32, 160, 128, 160, 160, 128, 96, 32, 0, 224, 32, 0, 96, 160, 0, 224, 160, 0, 96, 32, 128, 224, 32, 128, 96, 160, 128, 224, 160, 128, 32, 96, 0, 160, 96, 0, 32, 224, 0, 160, 224, 0, 32, 96, 128, 160, 96, 128, 32, 224, 128, 160, 224, 128, 96, 96, 0, 224, 96, 0, 96, 224, 0, 224, 224, 0, 96, 96, 128, 224, 96, 128, 96, 224, 128, 224, 224, 128, 32, 32, 64, 160, 32, 64, 32, 160, 64, 160, 160, 64, 32, 32, 192, 160, 32, 192, 32, 160, 192, 160, 160, 192, 96, 32, 64, 224, 32, 64, 96, 160, 64, 224, 160, 64, 96, 32, 192, 224, 32, 192, 96, 160, 192, 224, 160, 192, 32, 96, 64, 160, 96, 64, 32, 224, 64, 160, 224, 64, 32, 96, 192, 160, 96, 192, 32, 224, 192, 160, 224, 192, 96, 96, 64, 224, 96, 64, 96, 224, 64, 224, 224, 64, 96, 96, 192, 224, 96, 192, 96, 224, 192, 224, 224, 192]

def get_list(text):
    output_list = []

    for i in text:
        sequence_name = i.split(",")[0].split("/")[0]
        frame_name = i.split(",")[0].split("/")[1]
        object_id = i.split(",")[3]
        sentence = i.split(",")[4]

        output_list.append({"sequence name": sequence_name, "frame": frame_name, "expression": sentence, "object_id": object_id})

    return output_list


class A2DTrainDataset(Dataset):
    def __init__(self, args, root):
        self.db_root_dir = root
        self.tokenizer = transformers.BertTokenizer.from_pretrained(args["bert_tokenizer"])
        max_tokens = 20

        with open(os.path.join(root, "a2d_txt/train.txt")) as f:
            train_text = f.readlines()

        train_list = get_list(train_text)


        self.objects_list = []
        for train_sample in train_list:
            frames_name = train_sample["frame"]
            expression = train_sample["expression"]
            sequence_name = train_sample["sequence name"]
            k = train_sample["object_id"]

            sequence_frames_list = sorted(os.listdir(os.path.join(self.db_root_dir, "allframes", sequence_name + ".mp4")))


            sentence_raw = expression
            sentence_raw_split = sentence_raw.split(" ")
            sentence_raw_split = sentence_raw_split[:max_tokens - 2]
            input_ids = self.tokenizer.encode(text=sentence_raw_split, add_special_tokens=True)

            input_ids = input_ids[:max_tokens]
            attention_mask = [0] * max_tokens
            padded_input_ids = [0] * max_tokens

            padded_input_ids[:len(input_ids)] = input_ids
            attention_mask[:len(input_ids)] = [1] * len(input_ids)

            self.objects_list.append(
                {"name": sequence_name + "___" + k,
                 "sentences_list": torch.tensor(padded_input_ids).unsqueeze(0),
                 "attentions_list": torch.tensor(attention_mask).unsqueeze(0),
                 "frame_name": frames_name,
                 "sentence_raw": sentence_raw,
                 "sequence_frames_list": sequence_frames_list,
                 "frame_index": sequence_frames_list.index(frames_name + ".png")
                 })


        self.pair_im_lone_transform = transforms.Compose([
            transforms.ColorJitter(0.1, 0.05, 0.05, 0), # No hue change here as that's not realistic
        ])

        self.pair_im_dual_transform = transforms.Compose([
            ResizeAndPad(320, 320, Image.BICUBIC),
        ])

        self.pair_gt_dual_transform = transforms.Compose([
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
        object_info = self.objects_list[index]
        sequence_name = object_info["name"].split("___")[0]
        # print(sequence_name)
        obj_id = int(object_info["name"].split("___")[1]) + 1 # real obj_id in mask = id + 1
        sentence_raw = object_info["sentence_raw"]
        frame_name = object_info["frame_name"]
        sequence_frames_list = object_info["sequence_frames_list"]
        target_frame_idx = object_info["frame_index"]# random sample 3 frames

        back_frame_idx = target_frame_idx - random.randint(1, 5)
        forward_frame_idx = target_frame_idx + random.randint(1, 5)


        reference_frames_idx = [back_frame_idx, forward_frame_idx]


        images = []
        flows = []
        gts = []
        cls_gts = []
        tensor_embeddings = []
        attention_masks = []

        sequence_seed = np.random.randint(2147483647)

        for f_idx in reference_frames_idx:

            image_dir = os.path.join(self.db_root_dir, "allframes", sequence_name + ".mp4", sequence_frames_list[f_idx])
            flow_dir = os.path.join(self.db_root_dir, "allframes_flow", sequence_name + ".mp4", sequence_frames_list[f_idx])

            img = Image.open(image_dir).convert('RGB')
            flow = Image.open(flow_dir).convert("RGB")

            reseed(sequence_seed)
            img = self.pair_im_dual_transform(img)
            img = self.pair_im_lone_transform(img)

            reseed(sequence_seed)
            flow = self.pair_im_dual_transform(flow)
            flow = self.pair_im_lone_transform(flow)

            img = self.final_im_transform(img)
            flow = self.final_im_transform(flow)

            tensor_embedding = object_info["sentences_list"]
            attention_mask = object_info["attentions_list"]

            images.append(img)
            flows.append(flow)
            tensor_embeddings.append(tensor_embedding)
            attention_masks.append(attention_mask)


        image_dir = os.path.join(self.db_root_dir, "allframes", sequence_name + ".mp4", sequence_frames_list[target_frame_idx])
        flow_dir = os.path.join(self.db_root_dir, "allframes_flow", sequence_name + ".mp4", sequence_frames_list[target_frame_idx])
        label_dir = os.path.join(self.db_root_dir, "Annotations_visualize", sequence_name, sequence_frames_list[target_frame_idx])

        img = Image.open(image_dir).convert('RGB')
        flow = Image.open(flow_dir).convert("RGB")
        label = np.array(Image.open(label_dir))

        mask = (label == obj_id).astype(np.float32)
        mask[mask == 1] = 255
        gt = Image.fromarray(mask.astype(np.uint8), mode="L")


        reseed(sequence_seed)
        img = self.pair_im_dual_transform(img)
        img = self.pair_im_lone_transform(img)

        reseed(sequence_seed)
        flow = self.pair_im_dual_transform(flow)
        flow = self.pair_im_lone_transform(flow)

        reseed(sequence_seed)
        gt = self.pair_gt_dual_transform(gt)

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
        tensor_embeddings.append(tensor_embedding)
        attention_masks.append(attention_mask)

        images = torch.stack(images, dim=0)
        flows = torch.stack(flows, dim=0)
        gt = gt.unsqueeze(0)
        cls_gt = cls_gt.unsqueeze(0)
        tensor_embeddings = torch.stack(tensor_embeddings, dim=0)
        attention_masks = torch.stack(attention_masks, dim=0)

        return images, flows, cls_gt, gt, tensor_embeddings, attention_masks

    def __len__(self):
        return len(self.objects_list)





















class A2DTestDataset(Dataset):
    def __init__(self, args, root):
        self.db_root_dir = root
        self.tokenizer = transformers.BertTokenizer.from_pretrained(args["bert_tokenizer"])
        max_tokens = 20

        with open(os.path.join(root, "a2d_txt/test.txt")) as f:
            test_text = f.readlines()

        test_list = get_list(test_text)

        self.objects_list = []
        for test_sample in test_list:
            frames_name = test_sample["frame"]
            expression = test_sample["expression"]
            sequence_name = test_sample["sequence name"]
            k = test_sample["object_id"]

            sequence_frames_list = sorted(
                os.listdir(os.path.join(self.db_root_dir, "allframes", sequence_name + ".mp4")))

            sentence_raw = expression
            sentence_raw_split = sentence_raw.split(" ")
            sentence_raw_split = sentence_raw_split[:max_tokens - 2]
            input_ids = self.tokenizer.encode(text=sentence_raw_split, add_special_tokens=True)

            input_ids = input_ids[:max_tokens]
            attention_mask = [0] * max_tokens
            padded_input_ids = [0] * max_tokens

            padded_input_ids[:len(input_ids)] = input_ids
            attention_mask[:len(input_ids)] = [1] * len(input_ids)

            self.objects_list.append(
                {"name": sequence_name + "___" + k,
                 "sentences_list": torch.tensor(padded_input_ids).unsqueeze(0),
                 "attentions_list": torch.tensor(attention_mask).unsqueeze(0),
                 "frame_name": frames_name,
                 "sentence_raw": sentence_raw,
                 "sequence_frames_list": sequence_frames_list,
                 "frame_index": sequence_frames_list.index(frames_name + ".png")
                 })

        self.im_transform = transforms.Compose([
            ResizeAndPad(320, 320, Image.BICUBIC),
            transforms.ToTensor(),
            im_normalization,
        ])


    def __getitem__(self, index):
        object_info = self.objects_list[index]
        sequence_name = object_info["name"].split("___")[0]
        obj_id = int(object_info["name"].split("___")[1]) + 1 # real obj_id in mask = id + 1
        sentence_raw = object_info["sentence_raw"]
        frame_name = object_info["frame_name"]
        sequence_frames_list = object_info["sequence_frames_list"]
        target_frame_idx = object_info["frame_index"]# random sample 3 frames


        back_frame_idx = target_frame_idx - 1
        forward_frame_idx = target_frame_idx + 1


        reference_frames_idx = [back_frame_idx, forward_frame_idx]

        images = []
        flows = []
        gts = []
        cls_gts = []
        tensor_embeddings = []
        attention_masks = []



        for f_idx in reference_frames_idx:

            image_dir = os.path.join(self.db_root_dir, "allframes", sequence_name + ".mp4", sequence_frames_list[f_idx])
            flow_dir = os.path.join(self.db_root_dir, "allframes_flow", sequence_name + ".mp4", sequence_frames_list[f_idx])

            img = Image.open(image_dir).convert('RGB')
            flow = Image.open(flow_dir).convert("RGB")

            img = self.im_transform(img)
            flow = self.im_transform(flow)

            tensor_embedding = object_info["sentences_list"]
            attention_mask = object_info["attentions_list"]

            images.append(img)
            flows.append(flow)
            tensor_embeddings.append(tensor_embedding)
            attention_masks.append(attention_mask)

        image_dir = os.path.join(self.db_root_dir, "allframes", sequence_name + ".mp4", sequence_frames_list[target_frame_idx])
        flow_dir = os.path.join(self.db_root_dir, "allframes_flow", sequence_name + ".mp4", sequence_frames_list[target_frame_idx])
        label_dir = os.path.join(self.db_root_dir, "Annotations_visualize", sequence_name, sequence_frames_list[target_frame_idx])

        label = np.array(Image.open(label_dir))
        mask = (label == obj_id).astype(np.float32)
        mask = torch.tensor(mask.astype(np.uint8))

        tensor_embedding = object_info["sentences_list"]
        attention_mask = object_info["attentions_list"]


        images.append(self.im_transform(Image.open(image_dir).convert('RGB')))
        flows.append(self.im_transform(Image.open(flow_dir).convert('RGB')))
        tensor_embeddings.append(tensor_embedding)
        attention_masks.append(attention_mask)

        images = torch.stack(images, dim=0)
        flows = torch.stack(flows, dim=0)
        tensor_embeddings = torch.stack(tensor_embeddings, dim=0)
        attention_masks = torch.stack(attention_masks, dim=0)

        data = {
            "images": images,
            "flows": flows,
            "masks": mask,
            "sequence_name": sequence_name,
            "frame_name": frame_name,
            "tensor_embeddings": tensor_embeddings,
            "attention_masks": attention_masks,
            "sentence_raw": sentence_raw
        }

        return data

    def __len__(self):
        return len(self.objects_list)


























class A2DTestDatasetMoreFrames(Dataset):
    def __init__(self, args, root):
        self.db_root_dir = root
        self.tokenizer = transformers.BertTokenizer.from_pretrained(args["bert_tokenizer"])
        max_tokens = 20

        with open(os.path.join(root, "a2d_txt/test.txt")) as f:
            test_text = f.readlines()

        test_list = get_list(test_text)

        self.objects_list = []
        for test_sample in test_list:
            frames_name = test_sample["frame"]
            expression = test_sample["expression"]
            sequence_name = test_sample["sequence name"]
            k = test_sample["object_id"]

            sequence_frames_list = sorted(
                os.listdir(os.path.join(self.db_root_dir, "allframes", sequence_name + ".mp4")))

            sentence_raw = expression
            sentence_raw_split = sentence_raw.split(" ")
            sentence_raw_split = sentence_raw_split[:max_tokens - 2]
            input_ids = self.tokenizer.encode(text=sentence_raw_split, add_special_tokens=True)

            input_ids = input_ids[:max_tokens]
            attention_mask = [0] * max_tokens
            padded_input_ids = [0] * max_tokens

            padded_input_ids[:len(input_ids)] = input_ids
            attention_mask[:len(input_ids)] = [1] * len(input_ids)

            self.objects_list.append(
                {"name": sequence_name + "___" + k,
                 "sentences_list": torch.tensor(padded_input_ids).unsqueeze(0),
                 "attentions_list": torch.tensor(attention_mask).unsqueeze(0),
                 "frame_name": frames_name,
                 "sentence_raw": sentence_raw,
                 "sequence_frames_list": sequence_frames_list,
                 "frame_index": sequence_frames_list.index(frames_name + ".png")
                 })

        self.im_transform = transforms.Compose([
            ResizeAndPad(320, 320, Image.BICUBIC),
            transforms.ToTensor(),
            im_normalization,
        ])

    def __getitem__(self, index):
        object_info = self.objects_list[index]
        sequence_name = object_info["name"].split("___")[0]
        obj_id = int(object_info["name"].split("___")[1]) + 1  # real obj_id in mask = id + 1
        sentence_raw = object_info["sentence_raw"]
        frame_name = object_info["frame_name"]
        sequence_frames_list = object_info["sequence_frames_list"]
        target_frame_idx = object_info["frame_index"]  # random sample 3 frames

        back_frame_idx = target_frame_idx - 1
        forward_frame_idx = target_frame_idx + 1

        reference_frames_idx = [target_frame_idx - 3, target_frame_idx - 2, target_frame_idx - 1, target_frame_idx + 1, target_frame_idx + 2, target_frame_idx + 3]

        images = []
        flows = []
        gts = []
        cls_gts = []
        tensor_embeddings = []
        attention_masks = []

        for f_idx in reference_frames_idx:
            image_dir = os.path.join(self.db_root_dir, "allframes", sequence_name + ".mp4",
                                     sequence_frames_list[f_idx])
            flow_dir = os.path.join(self.db_root_dir, "allframes_flow", sequence_name + ".mp4",
                                    sequence_frames_list[f_idx])

            img = Image.open(image_dir).convert('RGB')
            flow = Image.open(flow_dir).convert("RGB")

            img = self.im_transform(img)
            flow = self.im_transform(flow)

            tensor_embedding = object_info["sentences_list"]
            attention_mask = object_info["attentions_list"]

            images.append(img)
            flows.append(flow)
            tensor_embeddings.append(tensor_embedding)
            attention_masks.append(attention_mask)

        image_dir = os.path.join(self.db_root_dir, "allframes", sequence_name + ".mp4",
                                 sequence_frames_list[target_frame_idx])
        flow_dir = os.path.join(self.db_root_dir, "allframes_flow", sequence_name + ".mp4",
                                sequence_frames_list[target_frame_idx])
        label_dir = os.path.join(self.db_root_dir, "Annotations_visualize", sequence_name,
                                 sequence_frames_list[target_frame_idx])

        label = np.array(Image.open(label_dir))
        mask = (label == obj_id).astype(np.float32)
        mask = torch.tensor(mask.astype(np.uint8))

        tensor_embedding = object_info["sentences_list"]
        attention_mask = object_info["attentions_list"]

        images.append(self.im_transform(Image.open(image_dir).convert('RGB')))
        flows.append(self.im_transform(Image.open(flow_dir).convert('RGB')))
        tensor_embeddings.append(tensor_embedding)
        attention_masks.append(attention_mask)

        images = torch.stack(images, dim=0)
        flows = torch.stack(flows, dim=0)
        tensor_embeddings = torch.stack(tensor_embeddings, dim=0)
        attention_masks = torch.stack(attention_masks, dim=0)

        data = {
            "images": images,
            "flows": flows,
            "masks": mask,
            "sequence_name": sequence_name,
            "frame_name": frame_name,
            "tensor_embeddings": tensor_embeddings,
            "attention_masks": attention_masks,
            "sentence_raw": sentence_raw
        }

        return data

    def __len__(self):
        return len(self.objects_list)






















        # warning = 0
        # for sequence_name, _ in test_dict.items():
        #     annotation_path = os.path.join("/home/hadoop-automl/cephfs/data/zhaowangbo/dataset/RVOS/A2D/a2d_annotation_with_instances/", sequence_name)
        #     annotation_list = os.listdir(annotation_path)
        #     annotation_list = [_ for _ in annotation_list if not _.startswith(".")]
        #     annotation_list = sorted(annotation_list)
        #
        #     # print(annotation_list)
        #
        #     for anntation_name in annotation_list:
        #         with h5py.File(os.path.join(annotation_path, anntation_name), "r") as f:
        #             # for key in f.keys():
        #             #     print(f[key], key, f[key].name)
        #                 # print(f[key].value)
        #             # for key in f.keys():
        #             #     print(f[key].name)
        #             #     print(f[key].value)
        #
        #             id_list = list(f["instance"])
        #             mask = f["reMask"]
        #             # print(id_list)
        #             # print(mask.shape)
        #
        #             if len(mask.shape) == 3:
        #                 print(len(id_list))
        #                 print(mask.shape)
        #                 if len(id_list) != mask.shape[0]:
        #                     warning +=1
        #                     print("warning nums={}".format(warning))
        #                 # assert len(id_list) == mask.shape[0]
        #                 w, h = mask.shape[1:]
        #                 out_mask = np.zeros((h, w), dtype=np.uint8)
        #                 for i in range(len(id_list)):
        #                     id = id_list[i] + 1
        #                     current_mask = mask[i]
        #                     current_mask = current_mask.transpose(1, 0)
        #                     out_mask[current_mask != 0] = id
        #
        #             elif len(mask.shape) == 2:
        #                 assert len(id_list) == 1
        #                 w, h = mask.shape
        #                 out_mask = np.zeros((h, w), dtype=np.uint8)
        #                 id = id_list[0] + 1
        #                 current_mask = mask[:, :]
        #                 current_mask = current_mask.transpose(1, 0)
        #                 out_mask[current_mask != 0] = id
        #
        #         out_mask = Image.fromarray(out_mask.astype(np.uint8))
        #         out_mask.putpalette(palette)
        #         save_path = os.path.join("/home/hadoop-automl/cephfs/data/zhaowangbo/dataset/RVOS/A2D/Annotations_visualize", sequence_name)
        #         os.makedirs(save_path, exist_ok=True)
        #         out_mask.save(os.path.join(save_path,  anntation_name[:-2] + "png"))
        # print("warning nums={}".format(warning))