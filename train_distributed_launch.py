import datetime
import os
import math
import sys
import csv
import random
import numpy as np
import torch
import torch.multiprocessing as mp
from hyper_para import HyperParameters
import torch.distributed as distributed
from torch import nn
from util.logger import TensorboardLogger
from model.model import PropagationModel
from dataset.RefCOCO_dataset import RefCOCOTrainDataset #
from dataset.RefCOCO_dataset import RefCOCOTestDataset

from dataset.YoutubeVOS_dataset import YoutubevosTrainDataset
from dataset.YoutubeVOS_dataset import YoutubevosTestDataset
from dataset.A2D_dataset import A2DTrainDataset, A2DTestDataset
from a2d_test import a2d_evaluate
from dataset.DAVIS_dataset import DAVISTrainDataset
from dataset.DAVIS_dataset import DAVISTestDataset


from refcoco_test import refcoco_evaluate
from davis_test import davis_evaluate
from model.network import PropagationNetwork

from util.train_utils import renew_loader, evaluate_during_training

para = HyperParameters()
para.parse()
num_gpus = torch.cuda.device_count()

distributed.init_process_group(backend="nccl")



print('CUDA Device count: ', torch.cuda.device_count())
if para['benchmark']:
    torch.backends.cudnn.benchmark = True


local_rank = torch.distributed.get_rank()
world_size = torch.distributed.get_world_size()
torch.cuda.set_device(local_rank)

print('I am rank %d in this world of size %d!' % (local_rank, world_size))

if para['id'].lower() != 'null':
    print('I will take the role of logging!')
    long_id = para['id']
else:
    long_id = None


logger = TensorboardLogger(para['id'], long_id)
logger.log_string('hyperpara', str(para))

save_path = os.path.join(para["save_mdoel_dir"], para["folder_name"], long_id, long_id) if long_id is not None else None
output_dir = os.path.join(para["save_mdoel_dir"], para["folder_name"], long_id) if long_id is not None else None

if local_rank == 0:
    model = PropagationModel(para, logger=logger, save_path=save_path, local_rank=local_rank, world_size=world_size, distributed=True)
else:
    model = PropagationModel(para, local_rank=local_rank, world_size=world_size, distributed=True).train()


model_test = PropagationNetwork(para).cuda()


total_iter = 0


##########################data load########################

skip_values = [10, 15, 20, 25, 5]

if para['stage'] == 0: # RefCOCO
    dataset_root = os.path.expanduser(para['refcoco_root'])
    train_dataset, train_sampler, train_loader = renew_loader(para, dataset_root, local_rank=local_rank)
    test_dataset = RefCOCOTestDataset(para, dataset_root, split="val")
    print('RefCOCO dataset size: ', len(train_dataset))


elif para['stage'] == 1: # YoutubeVOS

    increase_skip_fraction = [0.1, 0.2, 0.3, 0.4, 0.9, 1.0]


    max_skip = 5
    a2d_dataset_root = para["a2d_root"]
    a2d_test_dataset = A2DTestDataset(para, a2d_dataset_root)

    train_dataset, train_sampler, train_loader = renew_loader(para, max_skip=5, local_rank=local_rank)
    test_datasets = [a2d_test_dataset]

else:
    RuntimeError("Wrong stage")


total_epoch = math.ceil(para['iterations']/len(train_loader))
current_epoch = total_iter // len(train_loader)
print('Number of training epochs (the last epoch might not complete): ', total_epoch)

if para['stage'] == 0: 
    epoch_start_eval = total_epoch - 10
elif para['stage'] == 1:
    epoch_start_eval = total_epoch - 1
elif para['stage'] == 2:  
    epoch_start_eval = 0
print('The epoch that we start to eval: ', epoch_start_eval)


if para['stage'] != 0:
    increase_skip_epoch = [round(total_epoch * f) for f in increase_skip_fraction]
    print('The skip value will increase approximately at the following epochs: ', increase_skip_epoch[:-1])


try:
    IoU = 0
    for e in range(current_epoch, total_epoch):
        print('Epoch %d/%d' % (e, total_epoch))

        # reset the skip value
        if para['stage'] != 0 and e != total_epoch and e >= increase_skip_epoch[0]:

            while e >= increase_skip_epoch[0]:
                cur_skip = skip_values[0]
                skip_values = skip_values[1:]
                increase_skip_epoch = increase_skip_epoch[1:]
            print('Increasing skip to: ', cur_skip)

            if para['stage'] != 0:
                train_dataset, train_sampler, train_loader = renew_loader(para=para, max_skip=cur_skip, local_rank=local_rank)

        train_sampler.set_epoch(e)
        model.train()
        for data in train_loader:
            model.do_pass(data, total_iter)
            total_iter += 1

            if total_iter >= para['iterations']:
                break


        if local_rank == 0:

            if total_iter >= 16000:
                model.save(total_iter)
                model_test.load_state_dict(model.model.module.state_dict())

                test_samplers = [torch.utils.data.SequentialSampler(test_dataset) for test_dataset in test_datasets]
                data_loader_tests = [
                    torch.utils.data.DataLoader(test_datasets[k], batch_size=1, sampler=test_samplers[k], num_workers=8) for
                    k in range(len(test_datasets))]

                mean_iou, overall_iou, precision5, precision6, precision7, precision8, precision9, precision_mAP, total_time = evaluate_during_training(
                    args=para, test_datasets=test_datasets, model_test=model_test, data_loader_tests=data_loader_tests,
                    epoch=e,
                    long_id=long_id, epoch_start_eval=epoch_start_eval, iterations=total_iter, output_dir=output_dir)

                metrics = [total_iter, precision5, precision6, precision7, precision8, precision9, precision_mAP, overall_iou, mean_iou]
                columns_name = ["iteration", "precision5", "precision6", "precision7", "precision8", "precision9",
                                "precision_mAP", "overall_iou", "mean_iou"]
                if not os.path.exists("log/" + long_id + "/eval_{}.csv".format(long_id)):
                    with open("log/" + long_id + "/eval_{}.csv".format(long_id), mode='w', newline='', encoding='utf8') as cfa:
                        pf = csv.writer(cfa)
                        pf.writerow(columns_name)
                        pf.writerow(metrics)
                else:
                    with open("log/" + long_id + "/eval_{}.csv".format(long_id), mode='a', newline='', encoding='utf8') as cfa:
                        pf = csv.writer(cfa)
                        pf.writerow(metrics)




finally:
    if not para['debug'] and model.logger is not None and total_iter>5000:
        model.save(total_iter)
    # Clean up
    distributed.destroy_process_group()












