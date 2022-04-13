from __future__ import annotations
from ast import AugAssign
from asyncore import write

from dataclasses import dataclass
from enum import Enum
from itertools import chain
import logging
from pathlib import Path
from typing import Any, List, Union
from torch.optim import Optimizer
from torch.nn.functional import cross_entropy
from torchbooster.dataset import Split
from torch.nn.parallel import DistributedDataParallel
from torchbooster.config import (
    BaseConfig,
    DatasetConfig,
    EnvironementConfig,
    LoaderConfig,
    OptimizerConfig,
    SchedulerConfig,
)
from copy import deepcopy
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from random import random as rand
import nlpaug.augmenter.sentence as nas
import nlpaug.augmenter.word as naw

import bayeformers
import torch
import nltk
import torch.nn as nn
import torchbooster.distributed as dist
import torchbooster.utils as utils
import torch.nn.functional as F
import torchvision.transforms as T
import transformers

ZERO = torch.tensor(0.)

class BayeMethod(str, Enum):
    BAYE_BY_BACKPROP = "baye_by_backprop"
    BAYESIAN_DROPOUT = "dropout"
    FREQUENTIST      = "frequentist"

@dataclass
class MyDataset(BaseConfig):
    train_dataset: DatasetConfig
    ood_detection_dataset: DatasetConfig

@dataclass
class ModelParam(BaseConfig):
    backbone_network: str
    projection_size: int
    projection_hidden: int
    dropout_p: float

@dataclass
class Config(BaseConfig):
    freeze_backbone: bool
    debug: bool
    epochs: int
    samples: int
    noise_samples: int
    clip: float
    task: str

    #NLP task
    tokenizer_max_length: int

    # Baye sampling method
    method: str

    model: ModelParam
    dataset: MyDataset
    env: EnvironementConfig
    loader: LoaderConfig
    optim: OptimizerConfig
    scheduler: SchedulerConfig

class CustomModel(nn.Module):
    def __init__(self, conf: Config, backbone, projector):
        super().__init__()
        self.conf: Config = conf
        self.backbone = backbone
        self.projector = projector
        if conf.freeze_backbone:
            logging.info(f'Backbone model is frozen')
            for param in self.backbone.parameters():
                param.requires_grad = False
        logging.info(f'Flat size = {get_backbone_model_output_features(backbone, conf)}')

    def forward(self, *kargs, **kwargs):
        o = self.backbone(**kwargs)

        if hasattr(o, "last_hidden_state"): # transformers/ bert model
            o : torch.FloatTensor = o.last_hidden_state#.view(conf.loader.batch_size, self.flat_size)
            o = self.projector(o.flatten(1))
        else:
            o = self.projector(o)

        return o

def get_backbone_model_output_features(backbone_model, conf):
    if isinstance(backbone_model, DistributedDataParallel):
        backbone_model = backbone_model.module
    return backbone_model.config.dim * conf.tokenizer_max_length #! make gen 

def get_projector(backbone_network: torch.Model, conf: Config) -> torch.Module:
    """get_projector

    Return untrained projector with the appropriate method

    Parameters
    ----------
    backbone_network : torch.Model
        The base backbone model that will be used, passed to return a matching projector
    conf : Config
        The torchbooster config

    Returns
    -------
    torch.Module
        The Projector model

    Raises
    ------
    ValueError
        Bad Bayemethod passed
    """

    base_projector = nn.Sequential(
        nn.SiLU(),
        nn.Linear(get_backbone_model_output_features(backbone_network, conf), conf.model.projection_size), # TODO switch to baye
        # nn.SiLU(), #TODO param?
        # nn.Dropout(p = conf.model.dropout_p),
        # nn.Linear(conf.model.projection_hidden, conf.model.projection_size), # TODO switch to baye
    )

    # base_projector = nn.Sequential(
    #     nn.SiLU(),
    #     nn.Linear(get_backbone_model_output_features(backbone_network, conf), conf.model.projection_hidden), # TODO switch to baye
    #     nn.SiLU(), #TODO param?
    #     # nn.Dropout(p = conf.model.dropout_p),
    #     nn.Linear(conf.model.projection_hidden, conf.model.projection_size), # TODO switch to baye
    # )
    
    return base_projector
    if conf.method == BayeMethod.FREQUENTIST.value or conf.method == BayeMethod.BAYESIAN_DROPOUT.value: # Train handles baye dropout
        return base_projector
    elif conf.method == BayeMethod.BAYE_BY_BACKPROP.value:
        logging.info("Using Baye by Backprop")
        return bayeformers.to_bayesian(base_projector) # TODO change ð or init better
    else:
        raise ValueError()


# def get_model(backbone_network, conf: Config):
#     #! Add classification head for testing 

#     return torch.nn.Sequential(
#         backbone_network,
#         get_projector(backbone_network, conf)
#     )


class TextDataPreparator():
    def __init__(self, dataset_len, tokenizer, conf, max_classes = -1) -> None:
        self.batch_num = 0
        self.dataset_len: int = dataset_len
        self.tokenizer = tokenizer
        self.conf: Config = conf
        self.max_classes = max_classes

    def tokenize_and_make(self, strs: Union[str, List[str]], **kwargs) -> torch.Tensor:
        if isinstance(strs, str):
            return self.conf.env.make(utils.to_tensor(self.tokenizer(strs, truncation=True, padding='max_length', max_length=self.conf.tokenizer_max_length, **kwargs))) # use return_tensors = pt and unsqueeze
        return self.conf.env.make(utils.stack_dictionaries([conf.env.make(utils.to_tensor(self.tokenizer(s, truncation=True, padding='max_length', max_length=self.conf.tokenizer_max_length, **kwargs))) for s in strs]))

    def get_augmenter(self, samples: int):
        strength = .8 #rand() #! change me    
        # augmenter1 = naw.AntonymAug(aug_p=strength/2)
        augmenter2 = naw.SynonymAug(aug_p=strength)    

        def wrap(inp: str):
            return augmenter2.augment(inp, n = samples) # augmenter1.augment(inp)

        return wrap


    def get_noise_samples(self, batch: List[str], conf: Config) -> List[List[str]]:
        """get_noise_samples

            Returns samples from the augmenter (get_augmenter) fir the current batch of data

        Parameters
        ----------
        batch : List[str]
            The batch provided
        conf : Config
            The config
        
        Returns
        -------
        List[str]
            The augmented data from the batch
        """
        if conf.loader.batch_size < 2:
            raise ValueError("Batch size has to be at least 2")
        aug = self.get_augmenter(conf.noise_samples)
        return [aug(a) for a in batch]

    def augment_and_prepare_batch(self, batch):
        augmenter = self.get_augmenter(1)
        new_batch = []
        for elem in batch:
            new_batch.append([elem[0], augmenter(elem[1])])

        x, y = [], []
        for elem in new_batch:
            x.append(elem[1]) #! specific
            y.append(elem[0])

        return self.tokenize_and_make(x), y

    def collate_fn(self, data):
        # self.batch_num += 1
        # x, y = [], []
        # for e in data:
        #     #! dataset specific
        #     x.append(e[1])
        #     y.append(e[0])

        # clear_x = self.tokenize_and_make(self.conf, self.tokenizer, x, padding='max_length', max_length=self.conf.tokenizer_max_length)
        
        # # strength = 0. * (self.batch_num * .5/(self.dataset_len/self.conf.loader.batch_size)) * .3 #! put in config TODO: increase during train
        # noisy_samples: List[List[str]] = self.get_noise_samples(x, self.conf) # batch x samples
        # # noisy_samples = list(filter(lambda e: type(e) == list, noisy_samples))
        # noisy_samples = [self.tokenize_and_make(self.conf, self.tokenizer, [a[i] for a in noisy_samples], padding='max_length', max_length=self.conf.tokenizer_max_length) for i in range(len(noisy_samples[0]))] # List(l=samples)[Tensor[BSxSEQ]]
        # return clear_x, noisy_samples, y
        return data

def fit(conf: Config, model, text_preparator: TextDataPreparator, tokenizer, dataset, optim, scheduler, prototypes):

    temp = 1.0 #! TODO config

    writer = SummaryWriter() if dist.is_primary() else None
    model = conf.env.make(model).train()

    C = text_preparator.max_classes

    alpha = 0.05 # TODO conf.prototype_shift
   

    for epoch in range(conf.epochs):
        pbar = tqdm(dataset, disable=not dist.is_primary())
        for batch_num, batch in enumerate(pbar):
            x,y = text_preparator.augment_and_prepare_batch(batch)
            out = F.normalize(model(**x)) # bs x ProjectorSize

            for i, label in enumerate(y):
                # Update class prototype
                l =  label - 1
                prototypes[l] = F.normalize(prototypes[l] * alpha + (1 - alpha) * out.select(0, i), dim=0)

            # compute L_compactness
            l_compactness = 0.
            for i in range(len(x)):
                l_compactness += torch.log(torch.exp(torch.dot(out.select(0, i), prototypes[y[i] - 1].detach())/temp) / torch.sum(torch.stack([torch.exp(torch.dot(out.select(0, i), prototypes[j])/temp) for j in range(C)]), dim=0))
            l_compactness *= -1
            
            # compute L_dispersion
            l_dispersion = 1/C * torch.sum(
                torch.stack([ torch.log(1/(C-1) * torch.sum(
                    torch.stack([ torch.exp(torch.dot(prototypes[i], prototypes[j])/temp) for j in range(C) if i != j ])
                , dim=0)
                )  for i in range(C)])
            , dim = 0)

            loss = 10 * l_dispersion + l_compactness

            utils.step(loss, optim, scheduler, retain_graph=True)
            pbar.set_postfix(l_d=l_dispersion.detach().item(), l_c=l_compactness.detach().item())

            if dist.is_primary():
                step = batch_num*(epoch+1)
                writer.add_scalar("loss/L_disper",  l_dispersion.detach().item(), step)
                writer.add_scalar("loss/L_compact", l_compactness.detach().item(), step)
                writer.add_scalar("loss/L_total",   loss.detach().item(), step)
                writer.add_scalar("proto/std", torch.stack(prototypes).std(0).mean().detach().item(), step)
            

def main(conf: Config):
    utils.seed(42)
    utils.boost(not conf.debug)

    logging.info(f'Loading model')
    backbone_network = conf.env.make(torch.hub.load('huggingface/transformers', 'model', conf.model.backbone_network))
    logging.info(f'Loading tokenizer')
    tokenizer = torch.hub.load('huggingface/transformers', 'tokenizer', conf.model.backbone_network) 

    model = conf.env.make(CustomModel(conf, backbone_network, get_projector(backbone_network, conf)))

    logging.info(f'Loading train dataset')
    train_dataset = conf.dataset.train_dataset.make(Split.TRAIN) # acceptance_fn=lambda x: len(x.strip()) > 0
    prep = TextDataPreparator(len(train_dataset), tokenizer, conf, max_classes=4) #! make general with other datasets
    train_dataset = conf.loader.make(train_dataset, shuffle=not conf.env.distributed, distributed=conf.env.distributed, collate_fn=prep.collate_fn)
    # iid_test_dataset = conf.dataset.train_dataset.make(Split.TRAIN)
    # ood_test_dataset = conf.dataset.ood_detection_dataset.make(Split.TRAIN)
    # DistributedIterableDataset

    prototypes = [torch.nn.parameter.Parameter(conf.env.make(F.normalize(torch.rand(conf.model.projection_size), dim=0))) for _ in range(prep.max_classes)]

    optim = conf.optim.make(chain(model.parameters(), prototypes))
    scheduler = conf.scheduler.make(optim)

    fit(conf, model, prep, tokenizer, train_dataset, optim, scheduler, prototypes=prototypes)


if __name__ == "__main__":
    logging.info("Starting")

    if dist.is_primary():
        nltk.download('averaged_perceptron_tagger')
        nltk.download('wordnet')
        nltk.download('omw-1.4')

    conf = Config.load(Path("configs/base.yml"))
    torch.multiprocessing.set_start_method('spawn')
    dist.launch(
        main,
        conf.env.n_gpu,
        conf.env.n_machine,
        conf.env.machine_rank,
        conf.env.dist_url,
        args=(conf, )
    )