from operator import xor

from torch.utils.data import ConcatDataset, DataLoader

import hw_asr.augmentations
import hw_asr.datasets
from hw_asr import batch_sampler as batch_sampler_module
from hw_asr.base.base_text_encoder import BaseTextEncoder
from hw_asr.collate_fn.collate import collate_fn
from hw_asr.utils.parse_config import ConfigParser
import torch


def get_dataloaders(configs: ConfigParser):
    dataloaders = {}
    for split, params in configs["data"].items():
        num_workers = params.get("num_workers", 1)

        # set train augmentations
        if split == 'train':
            wave_augs, spec_augs = None, None
            drop_last = True
        else:
            wave_augs, spec_augs = None, None
            drop_last = False

        # create and join datasets
        datasets = []
        for ds in params["datasets"]:
            datasets.append(configs.init_obj(
                ds, hw_asr.datasets))
        assert len(datasets)
        if len(datasets) > 1:
            dataset = ConcatDataset(datasets)
        else:
            dataset = datasets[0]

        # select batch size or batch sampler
        assert xor("batch_size" in params, "batch_sampler" in params), \
            "You must provide batch_size or batch_sampler for each split"
        if "batch_size" in params:
            bs = params["batch_size"]
            shuffle = True
            batch_sampler = None
        elif "batch_sampler" in params:
            batch_sampler = configs.init_obj(params["batch_sampler"], batch_sampler_module,
                                             data_source=dataset)
            bs, shuffle = 1, False
        else:
            raise Exception()

        # Fun fact. An hour of debugging was wasted to write this line
        assert bs <= len(dataset), \
            f"Batch size ({bs}) shouldn't be larger than dataset length ({len(dataset)})"

        # create dataloader
        
        
        if len(configs['data']) == 1 and list(configs['data'].keys())[0] == 'train':
            generator1 = torch.Generator().manual_seed(42)
            train_size = int(0.9 * len(dataset))
            test_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator1)
            
            dataloaders['train'] = DataLoader(
                train_dataset, batch_size=bs, collate_fn=collate_fn,
                shuffle=shuffle, num_workers=num_workers,
                batch_sampler=batch_sampler, drop_last=drop_last
            )
            dataloaders['val'] = DataLoader(
                val_dataset, batch_size=bs, collate_fn=collate_fn,
                shuffle=False, num_workers=num_workers,
                batch_sampler=batch_sampler, drop_last=drop_last
            )
            return dataloaders
            
        
        dataloader = DataLoader(
            dataset, batch_size=bs, collate_fn=collate_fn,
            shuffle=shuffle, num_workers=num_workers,
            batch_sampler=batch_sampler, drop_last=False
        )
        dataloaders[split] = dataloader
    return dataloaders
