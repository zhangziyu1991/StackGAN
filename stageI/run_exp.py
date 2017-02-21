from __future__ import division
from __future__ import print_function

import dateutil
import dateutil.tz
import datetime
import argparse
import pprint

import platform
import sys
if platform.system() == 'Darwin':
    sys.path.append('/Users/zzhang/StackGAN')
else:
    sys.path.append('/home/zhang/StackGAN')

from misc.datasets import TextDataset
from stageI.model import CondGAN
from stageI.trainer import CondGANTrainer
from misc.utils import mkdir_p
from misc.config import cfg, cfg_from_file


def parse_args():
    parser = argparse.ArgumentParser(description='Train a GAN network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='stageI/cfg/birds.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=-1, type=int)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.gpu_id != -1:
        cfg.GPU_ID = args.gpu_id
    print('Using config:')
    pprint.pprint(cfg)

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

    if platform.system() == 'Darwin':
        datadir = '/Users/zzhang/StackGAN/Data/%s' % cfg.DATASET_NAME
    else:
        datadir = 'Data/%s' % cfg.DATASET_NAME

    dataset = TextDataset(datadir, cfg.EMBEDDING_TYPE, 1)
    filename_test = '{}/train'.format(datadir)
    dataset.test = dataset.get_data(filename_test, 886, aug_flag=True)
    # dataset.test2 = dataset.get_data(filename_test, 251, aug_flag=True, animated='_animated')
    if cfg.TRAIN.FLAG:
        filename_train = '{}/train'.format(datadir)
        dataset.train = dataset.get_data(filename_train, 886, aug_flag=True)
        # dataset.train2 = dataset.get_data(filename_train, 7969, aug_flag=True)
        ckt_logs_dir = "ckt_logs/{}/{}_{}".format(cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)
        mkdir_p(ckt_logs_dir)
    else:
        # s_tmp = cfg.TRAIN.PRETRAINED_MODEL
        # ckt_logs_dir = s_tmp[:s_tmp.find('.ckpt')]
        ckt_logs_dir = cfg.TRAIN.PRETRAINED_MODEL

    model = CondGAN(image_shape=dataset.image_shape)

    algo = CondGANTrainer(model=model, dataset=dataset, ckt_logs_dir=ckt_logs_dir)
    if cfg.TRAIN.FLAG:
        # quit()
        algo.train()
    else:
        ''' For every input text embedding/sentence in the
        training and test datasets, generate cfg.TRAIN.NUM_COPY
        images with randomness from noise z and conditioning augmentation.'''
        algo.evaluate()
