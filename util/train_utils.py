import datetime
import os
import math
import sys
import random
import numpy as np
import torch

from dataset.RefCOCO_dataset import RefCOCOTrainDataset #
from dataset.RefCOCO_dataset import RefCOCOTestDataset

from dataset.YoutubeVOS_dataset import YoutubevosTrainDataset
from dataset.YoutubeVOS_dataset import YoutubevosTestDataset


from dataset.DAVIS_dataset import DAVISTrainDataset
from dataset.DAVIS_dataset import DAVISTestDataset


from dataset.A2D_dataset import A2DTrainDataset

from refcoco_test import refcoco_evaluate
from davis_test import davis_evaluate
from youtubevos_test import youtubevos_evaluate
from a2d_test import a2d_evaluate


from torch.utils.data import ConcatDataset

def construct_loader(args, dataset):
    train_sampler = torch.utils.data.RandomSampler(dataset)
    train_loader = torch.utils.data.DataLoader(dataset, args['batch_size'], sampler=train_sampler, num_workers=8, drop_last=True, pin_memory=True)
    return train_sampler, train_loader


def construct_distributed_loader(args, dataset, local_rank):
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, rank=local_rank, shuffle=True)
    train_loader = torch.utils.data.DataLoader(dataset, args['batch_size'], sampler=train_sampler, num_workers=8, drop_last=True, pin_memory=True)
    return train_sampler, train_loader


def renew_loader(para, max_skip=None, local_rank=None):

    if para['stage'] == 0:  # refcoco
        dataset = RefCOCOTrainDataset(para, root)

    elif para['stage'] == 1:  # youtubevos + davis

        a2d_dataset_root = para["a2d_root"]

        traindataset = A2DTrainDataset(para, a2d_dataset_root)
        print("A2D dataset size: ", len(traindataset))


    if local_rank == None:
        train_sampler, train_loader = construct_loader(para, traindataset)
    else:
        train_sampler, train_loader = construct_distributed_loader(para, traindataset, local_rank)


    return traindataset, train_sampler, train_loader



def evaluate_during_training(args, test_datasets, model_test, data_loader_tests, epoch, long_id, epoch_start_eval, iterations=None, output_dir=None):

        if args['stage'] == 0:

            if epoch >= epoch_start_eval:

                ref_ids = test_dataset.ref_ids
                refer = test_dataset.refer
                ids = ref_ids
                device = torch.device("cuda")

                refs_ids_list, outputs, overallIoU = refcoco_evaluate(args=args, model=model_test, data_loader=data_loader_test,
                                                        ref_ids=ref_ids, device=device, display=False, epoch=epoch,
                                                        log_dir="log/" + long_id + "/eval.txt")
            else:
                overallIoU = 0

            return overallIoU


        elif args['stage'] == 1:

            device = torch.device("cuda")
            a2d_data_loader_test = data_loader_tests[0]

            mean_iou, overall_iou, precision5, precision6, precision7, precision8, precision9, precision_mAP, total_time = \
                a2d_evaluate(args=args, model=model_test, data_loader=a2d_data_loader_test, device=device, iterations=iterations, output_dir=output_dir)



            return mean_iou, overall_iou, precision5, precision6, precision7, precision8, precision9, precision_mAP, total_time


