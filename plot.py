from __future__ import annotations

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
import umap

from main import Config, CustomModel, MyDataset, ModelParam # used to load model

backbone = "distilbert-base-uncased"

IN_DOMAIN = "The game began development in 2010 , carrying over a large portion of the work done on Valkyria Chronicles II . While it etained the standard features of the series , it also underwent multiple adjustments , such as making the game more for series newcomers"

def process(model, tokenizer, conf, inp: str):
    a = tokenizer(inp, return_tensors='pt', padding="max_length", max_length = conf.tokenizer_max_length, truncation=True)
    a = conf.env.make(a)
    out = model(**a).squeeze(0)
    return out

def main():
    conf = Config.load(Path("configs/base.yml"))
    logging.info("Loading model")
    model = torch.load('models/model_500.pth').eval() #! check if bayesian droupout
    logging.info("Loading tokenizer")
    tokenizer = torch.hub.load('huggingface/transformers', 'tokenizer', backbone)

    test1 = [process(model, tokenizer, conf, "bruh bruh bruh bruh bruh bruh bruh bruh bruh bruh bruh bruh bruh bruh bruh bruh bruh bruh bruh bruh bruh bruh bruh bruh bruh bruh bruh bruh bruh bruh bruh bruh bruh bruh bruh ").detach().cpu() for _ in tqdm(range(5))]
    test1 = torch.stack(test1)
    print(torch.mean(torch.std(test1, 0)))

    test2 = [process(model, tokenizer, conf, "That memory we used to share is like a painted flower it never wilts.").detach().cpu() for _ in tqdm(range(5))]
    test2 = torch.stack(test2)
    print(torch.mean(torch.std(test2, 0)))


    test3 = [process(model, tokenizer, conf, IN_DOMAIN).detach().cpu() for _ in tqdm(range(5))]
    test3 = torch.stack(test3)
    print(torch.mean(torch.std(test3, 0)))
if __name__ == '__main__':
    main()