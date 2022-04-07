from __future__ import annotations
from ast import AugAssign

from dataclasses import dataclass
from enum import Enum
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
    lr: float

    #NLP task
    tokenizer_max_length: int

    # Baye sampling method
    method: str

    model: ModelParam
    dataset: MyDataset
    env: EnvironementConfig
    loader: LoaderConfig
    optim: OptimizerConfig
    # scheduler: SchedulerConfig

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
        nn.Linear(get_backbone_model_output_features(backbone_network, conf), conf.model.projection_hidden), # TODO switch to baye
        nn.SiLU(), #TODO param?
        nn.Dropout(p = conf.model.dropout_p),
        nn.Linear(conf.model.projection_hidden, conf.model.projection_size), # TODO switch to baye
    )
    
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


def tokenize_and_make(conf: Config, tokenizer: Any, strs: Union[str, List[str]], **kwargs) -> torch.Tensor:
    if isinstance(strs, str):
        return conf.env.make(utils.to_tensor(tokenizer(strs, truncation=True, **kwargs))) # use return_tensors = pt and unsqueeze
    return conf.env.make(utils.stack_dictionaries([utils.to_tensor(tokenizer(s, truncation=True, **kwargs)) for s in strs]))

def get_augmenter(strength: float, samples: int):
    #! Make general for other tasks, NLPAUG for now
    augmenter1 = naw.AntonymAug(aug_p=strength)
    augmenter2 = naw.SynonymAug(aug_p=strength)    

    def wrap(inp: str):
        return augmenter2.augment(augmenter1.augment(inp), n = samples)

    return wrap


def get_noise_samples(strength: float, batch: List[str], conf: Config) -> List[List[str]]:
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
    return [get_augmenter(strength, conf.noise_samples)(a) for a in batch]


def fit_text_task(conf: Config, model, tokenizer, dataset, optim, scheduler):
    #NOTE Notes on the implementation
    # Base Algorithm
    # pour n n fois le même x
        # 1 Get anchor pools
        # 2 Get displacement pools
        
        # losses
        #   loss d'hypersphere:
        #   loss position relative des samples négatifs
        #   loss position relative nulle des samples positifs 

    #      hypersphere uniform sampling + displacement scaler
    #      mais en fait non car on n'opère pas au niveau des classes mais des features
    #  hypothèse: la displacement loos va forcer les anchors à rester au même emplacement. 

    writer = SummaryWriter()
    model = conf.env.make(model).train()
    pbar = tqdm(dataset, disable=not dist.is_primary())
    for batch_num, x in enumerate(pbar):
        # anchors
        anchors = conf.env.make(torch.zeros(conf.samples, conf.loader.batch_size, conf.model.projection_size))
        inp = tokenize_and_make(conf, tokenizer, x, padding='max_length', max_length=conf.tokenizer_max_length)
        for i in range(conf.samples):
            out = F.normalize(model(**inp), dim=1) #! not general
            anchors[i] = out # swap if necassary

        # displacement
        strength = batch_num/len(pbar) * .9 #! put in config TODO: increase during train
        displacement: torch.Tensor = conf.env.make(torch.zeros(conf.loader.batch_size, conf.noise_samples, conf.model.projection_size))
        noisy_samples: List[List[str]] = get_noise_samples(strength, x, conf) # batch x samples

        for i, inp in enumerate(noisy_samples):
            if len(inp) == 0: continue #! quick fix, check if necessary
            inp = tokenize_and_make(conf, tokenizer, inp, padding='max_length', max_length=conf.tokenizer_max_length) #! tokenize in advance, in dataset!!!
            out = F.normalize(model(**inp), dim=1)
            displacement[i] = out

        # compute anchor loss
        anchor_loss = torch.mean(torch.std(anchors, dim=0)) # BSx

        # compute displacement loss
        mean_anchors = torch.mean(anchors, dim=0, keepdim=True).view((conf.loader.batch_size, 1, conf.model.projection_size))
        mean_anchors = mean_anchors.expand((-1, conf.noise_samples, -1))
        displacement_loss = torch.abs(strength - (1. - torch.clip(torch.cosine_similarity(displacement, mean_anchors, dim=2), 0))).mean(dim=1) # BSx
        
        # average all
        anchor_loss = anchor_loss.mean() # /
        displacement_loss = displacement_loss.mean() # / #displacement_loss.view((conf.loader.batch_size * conf.noise_samples)).mean()

        # if bayse by backprop, inclide lp and lvp
        if conf.method == BayeMethod.BAYE_BY_BACKPROP.value:
            projector = model.module.module.projector if isinstance(model, DistributedDataParallel) else model.projector
            loss_baye = 10e-12 * (projector.log_variational_posterior() - projector.log_prior()) #! fixme scaler
        else:
            loss_baye = ZERO

        # final loss
        loss = anchor_loss + displacement_loss + loss_baye #! add scalers
        loss *= conf.lr 

        #TODO CONTINUE HERE
        # triplet loss ou négative samples ou autr epour empécher collapse

        #TODO: fix distributed
        # if dist.is_primary():
        #     if conf.env.distributed:
        #         gathered_losses = [torch.zeros_like(loss) for _ in range(dist.get_world_size())]
        #         dist.gather(loss, gathered_losses)
        #         print(gathered_losses)
        utils.step(loss, optim, scheduler, clip=conf.clip)
        
        writer.add_scalar('loss/anchor', anchor_loss.detach().item(), batch_num)
        writer.add_scalar('loss/displacement', displacement_loss.detach().item(), batch_num)
        writer.add_scalar('loss/baye', loss_baye.detach().item(), batch_num)
        writer.add_scalar('loss/total', loss.detach().item(), batch_num)
        
        pbar.set_postfix(anchor_loss=anchor_loss.detach().item(), displacement_loss=displacement_loss.detach().item(), loss_baye=loss_baye.detach().item(), strength=strength)
        
        if batch_num % 500 == 0:
            save_path = f'./models/model_{batch_num}.pth'
            logging.info(f'Saving to {save_path}')
            torch.save(model, save_path)
        # else:
        #     dist.gather(loss)

        

def main(conf: Config):
    utils.seed(42)
    utils.boost(not conf.debug)

    logging.info(f'Loading model')
    backbone_network = conf.env.make(torch.hub.load('huggingface/transformers', 'model', conf.model.backbone_network))
    logging.info(f'Loading tokenizer')
    tokenizer = torch.hub.load('huggingface/transformers', 'tokenizer', conf.model.backbone_network) 

    model = conf.env.make(CustomModel(conf, backbone_network, get_projector(backbone_network, conf)))

    logging.info(f'Loading train dataset')
    train_dataset = conf.dataset.train_dataset.make(Split.TRAIN)
    train_dataset = conf.loader.make(train_dataset, shuffle=not conf.env.distributed, distributed=conf.env.distributed)
    # iid_test_dataset = conf.dataset.train_dataset.make(Split.TRAIN)
    # ood_test_dataset = conf.dataset.ood_detection_dataset.make(Split.TRAIN)
    # DistributedIterableDataset

    optim = conf.optim.make(model.parameters())
    scheduler = None #conf.scheduler.make()

    fit_text_task(conf, model, tokenizer, train_dataset, optim, scheduler)


if __name__ == "__main__":
    logging.info("Starting")

    if dist.is_primary():
        nltk.download('averaged_perceptron_tagger')
        nltk.download('wordnet')
        nltk.download('omw-1.4')

    conf = Config.load(Path("configs/base.yml"))

    dist.launch(
        main,
        conf.env.n_gpu,
        conf.env.n_machine,
        conf.env.machine_rank,
        conf.env.dist_url,
        args=(conf, )
    )