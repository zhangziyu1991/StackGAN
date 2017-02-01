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
    filename_test = '%s/train' % (datadir)
    dataset.test = dataset.get_data(filename_test, aug_flag=False)
    if cfg.TRAIN.FLAG:
        filename_train = '%s/train' % (datadir)
        dataset.train = dataset.get_data(filename_train, aug_flag=False)
        # ckt_logs_dir = "ckt_logs/%s/%s_%s" % \
        #     (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)
        ckt_logs_dir = "ckt_logs/%s" % \
                       (cfg.DATASET_NAME)
        mkdir_p(ckt_logs_dir)
    else:
        s_tmp = cfg.TRAIN.PRETRAINED_MODEL
        ckt_logs_dir = s_tmp[:s_tmp.find('.ckpt')]

    model = CondGAN(
        image_shape=dataset.image_shape
    )

    algo = CondGANTrainer(
        model=model,
        dataset=dataset,
        ckt_logs_dir=ckt_logs_dir
    )
    if cfg.TRAIN.FLAG:
        algo.train()
    else:
        ''' For every input text embedding/sentence in the
        training and test datasets, generate cfg.TRAIN.NUM_COPY
        images with randomness from noise z and conditioning augmentation.'''
        algo.evaluate()
