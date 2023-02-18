import datetime
import os
import math
import sys
import random
import numpy as np
import torch
from hyper_para import HyperParameters
from torch import nn
from model.model import PropagationModel
from dataset.A2D_dataset import A2DTestDataset
from torch.nn import functional as F
from model.network import PropagationNetwork
import matplotlib.pyplot as plt
import pandas as pd
import imageio
from PIL import Image
import cv2
import zipfile
from einops import rearrange
import time
from dataset.JHMDB_dataset import JHMDBTestDataset

def aggregate_wbg(prob, keep_bg=False):
    new_prob = torch.cat([
        torch.prod(1-prob, dim=1, keepdim=True),
        prob
    ], 1).clamp(1e-7, 1-1e-7)
    logits = torch.log((new_prob /(1-new_prob)))

    if keep_bg:
        return F.softmax(logits, dim=1)
    else:
        return F.softmax(logits, dim=1)[1:]

def resize_and_crop(im, input_h, input_w):
    # Resize and crop im to input_h x input_w size
    im_h, im_w = im.shape[-2:]
    scale = max(input_h / im_h, input_w / im_w)
    resized_h = int(np.round(im_h * scale))
    resized_w = int(np.round(im_w * scale))
    crop_h = int(np.floor(resized_h - input_h) / 2)
    crop_w = int(np.floor(resized_w - input_w) / 2)

    resized_im = F.interpolate(im, [resized_h, resized_w], mode="bilinear")

    new_im = torch.zeros((im.shape[0], 1, input_h, input_w))

    new_im[...] = resized_im[:, :, crop_h:crop_h+input_h, crop_w:crop_w+input_w]

    return new_im


SMOOTH = 1e-6
def calculate_IoU(pred, gt):
    IArea = (pred & (gt == 1.0)).astype(float).sum()
    OArea = (pred | (gt == 1.0)).astype(float).sum()
    IoU = (IArea + SMOOTH) / (OArea + SMOOTH)
    return IoU, IArea, OArea


def jhmdb_evaluate(args, model, data_loader, device, iterations, output_dir):
    model.eval()
    save_root = os.path.join(output_dir, "{}/".format(iterations))
    MeanIoU, IArea, OArea, Overlap = [], [], [], []

    with torch.no_grad():
        start_time = time.time()

        for data in data_loader:
            image = data["images"].to(device)
            flow = data["flows"].to(device)
            gt = data["masks"]
            # target_index = data["target_index"]

            original_w, original_h = gt.shape[2],  gt.shape[1] #(w, h)

            sequence_name = data["sequence_name"][0]
            sentences_raw = data["sentence_raw"][0]

            sentences = data["tensor_embeddings"].to(device)
            attentions = data["attention_masks"].to(device)

            logits = model.inference(image, flow, sentences, attentions)
            pred = torch.sigmoid(logits)
            target_pred = pred
            target_pred = resize_and_crop(target_pred, original_h, original_w)
            target_gt = gt.unsqueeze(0)


            target_pred = target_pred.cpu().data.numpy()
            target_pred = (target_pred > np.max(target_pred) * 0.5).astype(np.uint8)
            target_gt = target_gt.data.numpy()

            iou, iarea, oarea = calculate_IoU(target_pred, target_gt)
            MeanIoU.append(iou)
            IArea.append(iarea)
            OArea.append(oarea)
            Overlap.append(iou)
            # print("sequence: ", sequence_name, "frame:", data["frame_name"][0],  "sentences_raw: ", sentences_raw)

        prec5, prec6, prec7, prec8, prec9 = np.zeros((len(Overlap), 1)), np.zeros((len(Overlap), 1)), np.zeros((len(Overlap), 1)), np.zeros((len(Overlap), 1)), np.zeros((len(Overlap), 1))
        for i in range(len(Overlap)):
            if Overlap[i] >= 0.5:
                prec5[i] = 1
            if Overlap[i] >= 0.6:
                prec6[i] = 1
            if Overlap[i] >= 0.7:
                prec7[i] = 1
            if Overlap[i] >= 0.8:
                prec8[i] = 1
            if Overlap[i] >= 0.9:
                prec9[i] = 1

        mAP_thres_list = list(range(50, 95 + 1, 5))
        mAP = []
        for i in range(len(mAP_thres_list)):
            tmp = np.zeros((len(Overlap), 1))
            for j in range(len(Overlap)):
                if Overlap[j] >= mAP_thres_list[i] / 100.0:
                    tmp[j] = 1
            mAP.append(tmp.sum() / tmp.shape[0])

        end_time = time.time()
        total_time = end_time - start_time

        return np.mean(np.array(MeanIoU)), np.array(IArea).sum() / np.array(OArea).sum(), prec5.sum() / prec5.shape[0], prec6.sum() / prec6.shape[0], prec7.sum() / prec7.shape[0], prec8.sum() / prec8.shape[0], prec9.sum() / prec9.shape[0], np.mean(np.array(mAP)), total_time



if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    para = HyperParameters()
    para.parse()
    num_gpus = torch.cuda.device_count()

    jhmdb_dataset_root = "dataset_path"
    dataset_test = JHMDBTestDataset(para, jhmdb_dataset_root)
    # for i in dataset_test:
    #     pass
    model_path = "modem_path"
    print(model_path)
    model_list = sorted(os.listdir(model_path))
    for model_name in model_list:
        model_path = os.path.join("model_path", model_name)

        output_dir = "output_path"

        test_sampler = torch.utils.data.SequentialSampler(dataset_test)
        data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, sampler=test_sampler, num_workers=8)

        model = PropagationNetwork(para).cuda()
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint)

        device = torch.device("cuda")
        iterations = int(model_path.split("/")[-1].split("_")[-1][:-4])

        mean_iou, overall_iou, precision5, precision6, precision7, precision8, precision9, precision_mAP, total_time = jhmdb_evaluate(args=para, model=model, data_loader=data_loader_test, device=device, iterations=iterations,
                           output_dir=output_dir)
        print(iterations)
        print(f'Test split results:\n'
              f'Precision@0.5 {precision5:.3f}, Precision@0.6 {precision6:.3f}, '
              f'Precision@0.7 {precision7:.3f}, Precision@0.8 {precision8:.3f}, Precision@0.9 {precision9:.3f},\n'
              f'mAP Precision @0.5:0.05:0.95 {precision_mAP:.3f},\n'
              f'Overall IoU {overall_iou:.3f}, Mean IoU {mean_iou:.3f}， Total Time {total_time:.3f}')