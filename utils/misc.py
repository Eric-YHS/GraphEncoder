import logging
import os
import random
import time

import numpy as np
import torch
import yaml
from easydict import EasyDict


class BlackHole(object):
    def __setattr__(self, name, value):
        pass

    def __call__(self, *args, **kwargs):
        return self

    def __getattr__(self, name):
        return self


def load_config(path):
    with open(path, 'r') as f:
        return EasyDict(yaml.safe_load(f))


def get_logger(name, log_dir=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s::%(name)s::%(levelname)s] %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        file_handler = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_new_log_dir(root='./logs', prefix='', tag=''):
    fn = time.strftime('%Y_%m_%d__%H_%M_%S', time.localtime())
    if tag != '':
        fn = fn + '_' + tag
    if prefix != '':
        fn = prefix + '_' + fn
    log_dir = os.path.join(root, fn)
    os.makedirs(log_dir)
    return log_dir


def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def log_hyperparams(writer, args):
    from torch.utils.tensorboard.summary import hparams
    vars_args = {k: v if isinstance(v, str) else repr(v) for k, v in vars(args).items()}
    exp, ssi, sei = hparams(vars_args, {})
    writer.file_writer.add_summary(exp)
    writer.file_writer.add_summary(ssi)
    writer.file_writer.add_summary(sei)


def int_tuple(argstr):
    return tuple(map(int, argstr.split(',')))


def str_tuple(argstr):
    return tuple(argstr.split(','))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def update_config_with_args(config, args):
    if args.encoder_layers is not None:
        config.encoder.num_layers = int(args.encoder_layers)
    if args.model_layers is not None:
        config.model.num_layers = int(args.model_layers)
    if args.encoder_name is not None:
        config.encoder.name = args.encoder_name
    if args.denoiser_name is not None:
        config.model.model_type = args.denoiser_name

    return config

def update_config_with_data(config, batch):
    config.data.node_in_dim = int(batch.x.shape[1])

    if getattr(batch, "edge_attr", None) is None:
        edge_in_dim = 0
    else:
        edge_in_dim = int(batch.edge_attr.shape[1])
    config.data.edge_in_dim = edge_in_dim
    config.model.edge_feat_dim = edge_in_dim + 2

    return config