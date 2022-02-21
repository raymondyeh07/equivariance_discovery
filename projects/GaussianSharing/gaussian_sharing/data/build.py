"""Builds a dataset from config."""
import torch

from struct_discovery.data.build import build_batch_data_loader
from struct_discovery.data.samplers.samplers import TrainingSampler, InferenceSampler

from .gaussian_dataset import GaussianDataset


def build_gaussian_train_loader(cfg, mapper=None, A=None):
    return _build_gaussian_loader(cfg, mapper, split='TRAIN', A=A)


def build_gaussian_val_loader(cfg, mapper=None, A=None):
    return _build_gaussian_loader(cfg, mapper, split='VAL', A=A)


def build_gaussian_test_loader(cfg, mapper=None, A=None):
    return _build_gaussian_loader(cfg, mapper, split='TEST', A=A)


def _build_gaussian_loader(cfg, mapper=None, split='TRAIN', A=None):
    num_samples, batch_size, rand_seed = _get_dataset_info(cfg, split=split)
    dataset = GaussianDataset(cfg.DATASET.NUM_DIMS,
                              [1, cfg.DATASET.MAX_CLUSTER_NUM],
                              num_samples,
                              cfg.DATASET.CLUSTER_MU_DIFF,
                              cfg.DATASET.CLUSTER_STD,
                              rand_seed,
                              A)
    if batch_size < 0:
        batch_size = len(dataset)
    if split == 'TEST':
        return torch.utils.data.DataLoader(
            dataset,
            batch_size,
            num_workers=cfg.DATALOADER.NUM_WORKERS)
    else:
        sampler = TrainingSampler(len(dataset), shuffle=True)
        return build_batch_data_loader(
            dataset,
            sampler,
            batch_size,
            num_workers=cfg.DATALOADER.NUM_WORKERS)


def _get_dataset_info(cfg, split='TRAIN'):
    num_samples = getattr(cfg.DATASET, 'NUM_SAMPLES_%s' % split)
    batch_size = getattr(cfg.SOLVER, 'BATCH_SIZE_%s' % split)
    rand_seed = getattr(cfg.DATASET, 'SEED_%s' % split)
    return num_samples, batch_size, rand_seed
