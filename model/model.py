"""
model.py - warpper and utility functions for network training
Compute loss, back-prop, update parameters, logging, etc.
"""
import operator
from functools import reduce #python 3
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
from model.losses import LossComputer, iou_hooks_mo, iou_hooks_so
from util.log_integrator import Integrator
from model.network import PropagationNetwork
from util.schedule import WarmupMultiStepLR
from einops import rearrange
from torch.nn import functional as F

class PropagationModel:
    def __init__(self, para, logger=None, save_path=None, distributed=False, local_rank=0, world_size=1):
        self.para = para
        self.local_rank = local_rank

        if not distributed:
            self.model = nn.parallel.DataParallel(PropagationNetwork(para)).cuda()
        else:
            self.model = nn.parallel.DistributedDataParallel(nn.SyncBatchNorm.convert_sync_batchnorm(PropagationNetwork(para).cuda()), device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False, find_unused_parameters=True)
            print("distributed!!!!!!")



        self.logger = logger
        self.save_path = save_path

        if logger is not None:
            self.last_time = time.time()


        self.train_integrator = Integrator(self.logger, distributed=distributed, local_rank=local_rank, world_size=world_size)

        self.train_integrator.add_hook(iou_hooks_so)

        self.loss_computer = LossComputer(para)

        params = [
            {"params": [p for p in self.model.module.Visionmodel.parameters() if p.requires_grad], "lr":para["lr"]},
            {"params": [p for p in self.model.module.Flowmodel.parameters() if p.requires_grad], "lr": para["lr"]},

            {"params": reduce(operator.concat, [[p for p in self.model.module.Languagemodel.Languagemodel.encoder.layer[i].parameters() if p.requires_grad] for i in range(12)]), "lr":para["lr"]},
            {"params": [p for p in self.model.module.Languagemodel.Languagemodel.pooler.parameters() if p.requires_grad], "lr":para["lr"]},

            {"params": [p for p in self.model.module.LanguagemodelGuidedSegmentation.parameters() if p.requires_grad], "lr":para["lr"]},

            {"params": [p for p in self.model.module.contrastive.parameters() if p.requires_grad], "lr":para["lr"]}
        ]



        self.optimizer = torch.optim.Adam(params)


        if para["stage"] == 0:
            self.scheduler = WarmupMultiStepLR(optimizer=self.optimizer, milestones=para["steps"], gamma=para["gamma"],
                                               warmup_factor=0.001, warmup_iters=1000, warmup_method="linear")
        else:
            self.scheduler = WarmupMultiStepLR(optimizer=self.optimizer, milestones=para["steps"], gamma=para["gamma"],
                                               warmup_factor=0.001, warmup_iters=0, warmup_method="linear") #do not need warmup for finetune stage



        self.report_interval = 100
        self.save_im_interval = 800

        if para['debug']:
            self.report_interval = self.save_im_interval = 1


    def save(self, it):
        if self.local_rank == 0:
            if self.save_path is None:
                print('Saving has been disabled.')
                return

            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            model_path = self.save_path + ('_%s.pth' % it)
            torch.save(self.model.module.state_dict(), model_path)

            print('Model saved to %s.' % model_path)

        # self.save_checkpoint(it)

    def save_checkpoint(self, it):
        if self.save_path is None:
            print('Saving has been disabled.')
            return

        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        checkpoint_path = self.save_path + '_checkpoint.pth'
        checkpoint = {
            'it': it,
            'network': self.model.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()}
        torch.save(checkpoint, checkpoint_path)

        print('Checkpoint saved to %s.' % checkpoint_path)

    def load_model(self, path):

        checkpoint = torch.load(path)

        it = checkpoint['it']
        network = checkpoint['network']
        optimizer = checkpoint['optimizer']
        scheduler = checkpoint['scheduler']

        self.model.module.load_state_dict(network)
        self.optimizer.load_state_dict(optimizer)
        self.scheduler.load_state_dict(scheduler)

        print('Model loaded.')

        return it

    def load_network(self, path):
        src_dict = torch.load(path)
        self.model.module.load_state_dict(src_dict)
        print('Network weight loaded:', path)


    def train(self):
        self._is_train = True
        self._do_log = True
        self.integrator = self.train_integrator
        return self

    def do_pass(self, data, it=0):

        torch.set_grad_enabled(self._is_train)
        image, flow, cls_gt, gt, sentences, attentions = data
        image, flow, cls_gt, gt, sentences, attentions = image.cuda(non_blocking=True), flow.cuda(non_blocking=True), cls_gt.cuda(non_blocking=True), \
                                                   gt.cuda(non_blocking=True), sentences.cuda(non_blocking=True), attentions.cuda(non_blocking=True)

        gt = rearrange(gt,  "n t c h w -> (n t) c h w")

        vision_map, flow_map, vision_flow_map, logits, mask = self.model(image, flow, sentences, attentions, cls_gt)

        out = {"logits": logits, "mask": mask,
               "vision_map": vision_map, "flow_map": flow_map, "vision_flow_map": vision_flow_map}

        data = {"cls_gt": cls_gt, "gt": gt}


        losses = self.loss_computer.compute({**data, **out}, it)

        self.integrator.add_dict(losses)

        if (it) % self.report_interval == 0 and it != 0:

            if self.logger is not None:
                self.logger.log_scalar('train/lr', self.scheduler.get_last_lr()[0], it)
                self.logger.log_metrics('train', 'time', (time.time()-self.last_time)/self.report_interval, it)
                self.logger.log_metrics('train', 'lr', self.optimizer.param_groups[0]["lr"], it)

            self.last_time = time.time()
            self.train_integrator.finalize('train', it)
            self.train_integrator.reset_except_hooks()


        self.optimizer.zero_grad()
        losses['total_loss'].backward()
        self.optimizer.step()
        self.scheduler.step()



