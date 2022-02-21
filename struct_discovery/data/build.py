"""Tools for building dataset and dataset loader."""
import numpy as np
import torch
from struct_discovery.utils.env import seed_all_rng


def build_batch_data_loader(dataset, sampler, batch_size, num_workers):
    return torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        worker_init_fn=worker_init_reset_seed,
    )


def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch


def worker_init_reset_seed(worker_id):
    seed_all_rng(np.random.randint(2 ** 31) + worker_id)
