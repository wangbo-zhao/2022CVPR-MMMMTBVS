from argparse import ArgumentParser




def none_or_default(x, default):
    return x if x is not None else default


class HyperParameters():
    def parse(self, unknown_arg_ok=False):
        parser = ArgumentParser()

        # Enable torch.backends.cudnn.benchmark -- Faster in some cases, test in your own environment
        parser.add_argument('--benchmark', default=True, action='store_true')
        parser.add_argument('--folder_name', default='a', help='BERT tokenizer')

        parser.add_argument('--bert_tokenizer', default='/home/hadoop-automl/cephfs/data/zhaowangbo/tools/models_to_load/bert-base-uncased', help='BERT tokenizer')
        parser.add_argument('--ck_bert',
                            default='/home/hadoop-automl/cephfs/data/zhaowangbo/tools/models_to_load/bert-base-uncased',
                            help='BERT pre-trained weights')

        parser.add_argument('--save_mdoel_dir',
                            default="/home/hadoop-automl/cephfs/data/zhaowangbo/RVOS_experiments/0save_models", type=str,
                            help='path to save models')
        parser.add_argument('--pretrained_dir',
                            default="/home/hadoop-automl/cephfs/data/zhaowangbo/tools/torch/hub/checkpoints/resnet101-5d3b4d8f.pth", type=str,
                            help='Resnet101 Imagenet pretrained model dir')
        parser.add_argument('--flow_pretrained_dir',
                            default="/home/hadoop-automl/cephfs/data/zhaowangbo/tools/torch/hub/checkpoints/resnet34-333f7ec4.pth", type=str,
                            help='Resnet34 Imagenet pretrained model dir for flow backbone')
        # Data parameters

        parser.add_argument('--refcoco_root', help='RefCOCO training data root',
                            default='/home/hadoop-automl/cephfs/data/zhaowangbo/dataset/RVOS/RefCOCO')
        parser.add_argument('--refcoco_dataset', help='dataset name', default='refcoco')
        parser.add_argument('--splitBy', help='split By', default='unc')

        parser.add_argument('--yv_root', help='YouTubeVOS data root', default='/home/hadoop-automl/cephfs/data/zhaowangbo/dataset/RVOS/Youtube/rvos')
        parser.add_argument('--davis_root', help='DAVIS data root', default='/home/hadoop-automl/cephfs/data/zhaowangbo/dataset/RVOS/DAVIS/DAVIS2017')
        parser.add_argument('--a2d_root', help='A2D data root', default='/home/hadoop-automl/cephfs/data/zhaowangbo/dataset/RVOS/A2D/')

        parser.add_argument('--stage',
                            help='Training stage (0-static images, 1-Blender dataset, 2-DAVIS+YouTubeVOS)',
                            type=int, default=0)


        parser.add_argument('--gpu_nums', help='gpu_nums', default=2, type=int)
        # Generic learning parameters
        parser.add_argument('-b', '--batch_size', help='Default is dependent on the training stage, see below',
                            default=None, type=int)
        parser.add_argument('-i', '--iterations', help='Default is dependent on the training stage, see below',
                            default=None, type=int)
        parser.add_argument('--steps', help='Default is dependent on the training stage, see below', nargs="*",
                            default=None, type=int)

        parser.add_argument('--lr', help='Initial learning rate', default=1e-5, type=float)
        parser.add_argument('--gamma', help='LR := LR*gamma at every decay step', default=0.1, type=float)

        # Loading
        parser.add_argument('--load_network', help='Path to pretrained network weight only')
        parser.add_argument('--load_model', help='Path to the model file, including network, optimizer and such')

        # Logging information
        parser.add_argument('--id', help='Experiment UNIQUE id, use NULL to disable logging to tensorboard',
                            default='NULL')
        parser.add_argument('--debug', help='Debug mode which logs information more often', action='store_true')

        # Multiprocessing parameters, not set by users
        parser.add_argument('--local_rank', default=0, type=int, help='Local rank of this process')

        if unknown_arg_ok:
            args, _ = parser.parse_known_args()
            self.args = vars(args)
        else:
            self.args = vars(parser.parse_args())

        # Stage-dependent hyperparameters
        # Assign default if not given
        if self.args['stage'] == 0: 
            self.args['batch_size'] = none_or_default(self.args['batch_size'], 8)
            self.args['iterations'] = none_or_default(self.args['iterations'], 200000)
            self.args['steps'] = none_or_default(self.args['steps'], [150000, 180000])
            self.args['lr'] = 2e-5

        elif self.args['stage'] == 1: 

            if self.args['gpu_nums'] == 1:
                self.args['batch_size'] = none_or_default(self.args['batch_size'], 8) # 1GPU: batchsize=8, 2GPU: batchsize=4, 4GPU: batchsize=2
            elif self.args['gpu_nums'] == 2:
                self.args['batch_size'] = none_or_default(self.args['batch_size'], 4) # 1GPU: batchsize=8, 2GPU: batchsize=4, 4GPU: batchsize=2
            elif self.args['gpu_nums'] == 4:
                self.args['batch_size'] = none_or_default(self.args['batch_size'], 2) # 1GPU: batchsize=8, 2GPU: batchsize=4, 4GPU: batchsize=2

            self.args['iterations'] = none_or_default(self.args['iterations'], 30000)
            self.args['steps'] = none_or_default(self.args['steps'], [25000, 28000])
            self.args['lr'] = 2e-5

        elif self.args['stage'] == 2:
            self.args['batch_size'] = none_or_default(self.args['batch_size'], 8)
            self.args['iterations'] = none_or_default(self.args['iterations'], 30000)
            self.args['steps'] = none_or_default(self.args['steps'], [30000])
            self.args['lr'] = 1e-6


    def __getitem__(self, key):
        return self.args[key]

    def __setitem__(self, key, value):
        self.args[key] = value

    def __str__(self):
        return str(self.args)










