from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
from typing import Any, List, Union
from torch.optim import Optimizer
from torch.nn.functional import cross_entropy
from torchbooster.dataset import Split
from torchbooster.config import (
    BaseConfig,
    DatasetConfig,
    EnvironementConfig,
    LoaderConfig,
    OptimizerConfig,
    SchedulerConfig,
)
from tqdm import tqdm
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
    epochs: int
    samples: int
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
    # scheduler: SchedulerConfig

class CustomModel(nn.Module):
    def __init__(self, conf, backbone, projector):
        super().__init__()
        self.conf: Config = conf
        self.backbone = backbone
        self.projector = projector
        self.flat_size = conf.tokenizer_max_length * get_backbone_model_output_features(self.backbone)
        logging.info(f'Flat size = {self.flat_size}')

    def forward(self, *kargs, **kwargs):
        o = self.backbone(**kwargs)

        if hasattr(o, "last_hidden_state"): # transformers/ bert model
            o : torch.FloatTensor = o.last_hidden_state.reshape(conf.loader.batch_size, self.flat_size)
            o = self.projector(o)
        else:
            o = self.projector(o)

        return o

def get_backbone_model_output_features(backbone_model):
    return backbone_model.config.dim

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
        nn.Linear(get_backbone_model_output_features(backbone_network), conf.model.projection_hidden), # TODO switch to baye
        nn.SiLU(), #TODO param?
        nn.Dropout(p = conf.model.dropout_p),
        nn.Linear(conf.model.projection_hidden, conf.model.projection_size), # TODO switch to baye
    )
    
    if conf.method == BayeMethod.FREQUENTIST.value or conf.method == BayeMethod.BAYESIAN_DROPOUT.value: # Train handles baye dropout
        return base_projector
    elif conf.method == BayeMethod.BAYE_BY_BACKPROP.value:
        return bayeformers.to_bayesian(base_projector) # TODO change ð or init better
    else:
        raise ValueError()


def get_model(backbone_network, conf: Config):
    #! Add classification head for testing 
    return torch.nn.Sequential(
        backbone_network,
        get_projector(backbone_network, conf)
    )


def tokenize_and_make(conf: Config, tokenizer: Any, strs: Union[str, List[str]], **kwargs) -> torch.Tensor:
    if isinstance(strs, str):
        return conf.env.make(utils.to_tensor(tokenizer(strs, **kwargs)))
    return conf.env.make(utils.stack_dictionaries([utils.to_tensor(tokenizer(s, **kwargs)) for s in strs]))

def get_augmenter(strength: float, samples: int):
    #! Make general for other tasks, NLPAUG for now
    augmenter1 = naw.AntonymAug(aug_p=strength)
    augmenter2 = naw.SynonymAug(aug_p=strength)    

    def wrap(inp: str):
        return augmenter2.augment(augmenter1.augment(inp), n = samples)

    return wrap


def fit_text_task(conf: Config, model, tokenizer, dataset, optim, scheduler):
    # pour n n fois le même x
        # 1 Get anchor pools
        # 2 Get displacement pools
        
        # losses
        #   loss d'hypersphere:
        #   loss position relative des samples négatifs
        #   loss position relative nulle des samples positifs 

    #TODO hypersphere uniform sampling + displacement scaler

    model = conf.env.make(model).train()
    pbar = tqdm(dataset)
    for x in pbar:
        # anchors
        anchors = conf.env.make(torch.zeros(conf.samples, conf.loader.batch_size, conf.model.projection_size))
        inp = tokenize_and_make(conf, tokenizer, x, padding='max_length', max_length=conf.tokenizer_max_length)
        for i in tqdm(range(conf.samples)):
            out = F.normalize(model(**inp), dim=1) #! not general
            anchors[i] = out # swap if necassary

        # displacement
        strength = .1 #! displacement strength TODO: make random
        displacement = conf.env.make(torch.zeros(conf.samples, conf.loader.batch_size, conf.model.projection_size))
        augmenter = get_augmenter(strength, conf.samples)


        # custom collate 
        noisy_samples: List[List[str]] = [augmenter(a) for a in x] # batch x samples
        #TODO CONTINUE HERE: noisy_samples: tokenize to dataset

        for i, inp in tqdm(enumerate()):
            inp = tokenize_and_make(conf, tokenizer, inp, padding='max_length', max_length=conf.tokenizer_max_length)
            out = F.normalize(model(**inp), dim=1)
            displacement[i] = out 

        anchor_loss = torch.std(anchors, dim=0) #! replace with hypersphere sampling
        displacement_loss = torch.mean(torch.cosine_similarity(displacement, torch.mean(anchors, dim=0), dim=1))
        loss = anchor_loss + displacement_loss #! add scalers

        utils.step(loss, optim, scheduler, clip=conf.clip)
        

def main(conf: Config):
    logging.info(f'Loading model')
    backbone_network = conf.env.make(torch.hub.load('/home/win32gg/Documents/transformers', 'model', conf.model.backbone_network, source="local"))
    logging.info(f'Loading tokenizer')
    tokenizer = torch.hub.load('/home/win32gg/Documents/transformers', 'tokenizer', conf.model.backbone_network, source="local") 

    model = CustomModel(conf, backbone_network, get_projector(backbone_network, conf))

    logging.info(f'Loading train dataset')
    train_dataset = conf.loader.make(conf.dataset.train_dataset.make(Split.TRAIN), shuffle=True)
    # iid_test_dataset = conf.dataset.train_dataset.make(Split.TRAIN)
    # ood_test_dataset = conf.dataset.ood_detection_dataset.make(Split.TRAIN)

    optim = conf.optim.make(model.parameters())
    scheduler = None #conf.scheduler.make()

    fit_text_task(conf, model, tokenizer, train_dataset, optim, scheduler)


if __name__ == "__main__":
    logging.info("Starting")
    utils.seed(42)
    utils.boost()

    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')

    conf = Config.load(Path("configs/base.yml"))

    dist.launch(
        main,
        conf.env.n_gpu,
        conf.env.n_machine,
        conf.env.machine_rank,
        conf.env.dist_url,
        args=(conf, )
    )