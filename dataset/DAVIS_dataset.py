import os
from os import path

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

from dataset.utils import im_normalization, im_mean
from dataset.utils import reseed
import transformers
import random
from dataset.utils import ResizeAndPad

class DAVISTrainDataset(Dataset):

    def __init__(self, args, root, max_skip, split="train"):

        self.db_root_dir = root
        self.max_jump = max_skip

        max_tokens = 20

        self.input_ids = []
        self.attention_masks = []

        self.attention_masks = []
        self.tokenizer = transformers.BertTokenizer.from_pretrained(args["bert_tokenizer"])
        data_root = os.path.join(self.db_root_dir, 'meta_expression/')
        self.objects_list = []

        files = ['Davis17_annot1_full_video.txt', 'Davis17_annot2_full_video.txt', 'Davis17_annot1.txt', 'Davis17_annot2.txt']
        lines = []
        for i in range(len(files)):
            with open(os.path.join(data_root, files[i]), mode='r', encoding="latin-1") as a:
                lines.append(a.readlines())

        annotations = {}
        for l1, l2, l3, l4 in zip(*lines):
            words1 = l1.split()
            words2 = l2.split()
            words3 = l3.split()
            words4 = l4.split()

            sentences = [words1, words2, words3, words4]
            for i, s in enumerate(sentences):
                raw_s = ' '.join(s[2:])[1:-1]
                annotations[s[0] + '_' + str(int(s[1])-1) + '_' + str(i)] = raw_s

        with open(os.path.join(self.db_root_dir, 'ImageSets/2017/' + split + '.txt')) as f:
            seqs = f.readlines()

        for i in range(len(seqs)):
            seqs[i] = seqs[i].split("\n")[0]


        for seq in seqs:
            images = sorted(os.listdir(os.path.join(self.db_root_dir, 'JPEGImages/480p/', seq.strip())))
            image_id_first_frame = images[0].split('.')[0]

            annot_path = os.path.join('Annotations/480p', seq.strip(), image_id_first_frame + '.png')
            annot = np.asarray(Image.open(os.path.join(self.db_root_dir, annot_path)))
            num_objs = len(np.unique(annot)) - 1


            for i in range(num_objs):

                # consider which annotator
                for l in range(4):

                    if seq.strip() + '_' + str(i) + '_' + str(l) in annotations:
                        # print(seq.strip() + '_' + str(i) + '_' + str(l))
                        sentence_raw = annotations[seq.strip() + '_' + str(i) + '_' + str(l)]

                        input_ids = self.tokenizer.encode(text=sentence_raw, add_special_tokens=True)

                        input_ids = input_ids[:max_tokens]
                        attention_mask = [0] * max_tokens
                        padded_input_ids = [0] * max_tokens

                        padded_input_ids[:len(input_ids)] = input_ids
                        attention_mask[:len(input_ids)] = [1] * len(input_ids)


                        self.objects_list.append(
                            {"name": seq + "_" + str(i + 1) + "_" + str(l),
                             "sentences_list": torch.tensor(padded_input_ids).unsqueeze(0),
                             "attentions_list": torch.tensor(attention_mask).unsqueeze(0),
                             "frames_list": images})

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
        object_info = self.objects_list[index]
        sequence_name = object_info["name"].split("_")[0]
        obj_id = int(object_info["name"].split("_")[1])
        frames = object_info["frames_list"]

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
            image_dir = os.path.join(self.db_root_dir, "JPEGImages/480p", sequence_name, frames[f_idx])
            flow_dir = os.path.join(self.db_root_dir, "FlowMaps/480p", sequence_name, frames[f_idx])
            label_dir = os.path.join(self.db_root_dir, "Annotations/480p", sequence_name, frames[f_idx][:-3] + "png")
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








class DAVISTestDataset(Dataset):

    def __init__(self, args, root, split="train", emb_type="first_mask", annotator=1):

        self.db_root_dir = root
        self.emb_type = emb_type  # first mask or full video
        self.annotator = annotator  # 0 ior 1

        max_tokens = 20

        self.input_ids = []
        self.attention_masks = []

        self.attention_masks = []
        self.tokenizer = transformers.BertTokenizer.from_pretrained(args["bert_tokenizer"])
        data_root = os.path.join(self.db_root_dir, 'meta_expression/')

        if self.emb_type == 'first_mask':
            files = ['Davis17_annot1.txt', 'Davis17_annot2.txt']
        else:
            files = ['Davis17_annot1_full_video.txt', 'Davis17_annot2_full_video.txt']

        # read annotations from annotator
        with open(os.path.join(data_root, files[0]), mode='r') as a:
            lines = a.readlines()

        annotations = {}
        for l in lines:
            words = l.split()
            raw_s = ' '.join(words[2:])[1:-1]

            if words[0] not in annotations:
                annotations[words[0]] = {str(int(words[1]) - 1): raw_s}
            else:
                annotations[words[0]].update({str(int(words[1]) - 1): raw_s})


        with open(os.path.join(self.db_root_dir, 'ImageSets/2017/' + split+ '.txt')) as f:
            seqs = f.readlines()

        for i in range(len(seqs)):
            seqs[i] = seqs[i].split("\n")[0]

        self.video_list = []

        for seq in seqs:
            images = sorted(os.listdir(os.path.join(self.db_root_dir, 'JPEGImages/480p/', seq.strip())))
            image_id_first_frame = images[0].split('.')[0]

            annot_path = os.path.join('Annotations/480p', seq.strip(), image_id_first_frame + '.png')
            annot = np.asarray(Image.open(os.path.join(self.db_root_dir, annot_path)))
            num_objs = len(np.unique(annot)) - 1

            sentences_list = []
            attentions_list = []
            sentences_raw = []
            sequence_annotations = annotations[seq]

            for i in range(num_objs):
                sentence_raw = sequence_annotations[str(i)]
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


            self.video_list.append({"name": seq,
                                    "sentences_list": sentences_list,
                                    "attentions_list": attentions_list,
                                    "frames_list": images,
                                    "sentences_raw": sentences_raw})

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
            image_dir = os.path.join(self.db_root_dir,  "JPEGImages/480p", video["name"], frame_name)
            images.append(self.im_transform(Image.open(image_dir).convert('RGB')))

            flow_dir = os.path.join(self.db_root_dir, "FlowMaps/480p", video["name"], frame_name)
            flows.append(self.im_transform(Image.open(flow_dir).convert("RGB")))

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