from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from torch.nn import (Flatten, GELU, Linear, Module, Sequential, Sigmoid, Unflatten)
from torch.optim import Optimizer
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader
from torchbooster.dataset import Split
from torchbooster.config import (
    BaseConfig,
    DatasetConfig,
    EnvironementConfig,
    LoaderConfig,
    OptimizerConfig,
    SchedulerConfig,
)
from torchbooster.scheduler import BaseScheduler
from torchvision.utils import make_grid
from tqdm import tqdm

import torch
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
class Config(BaseConfig):
    epochs: int

    model: str
    task: str
    method: BayeMethod

    dataset: MyDataset
    env: EnvironementConfig
    loader: LoaderConfig
    optim: OptimizerConfig
    scheduler: SchedulerConfig

def get_model(language_model, conf: Config):
    
    return torch.nn.Sequential(
        language_model,
        nn.Linear(language_model.config.dim, language_model.config.dim), # TODO switch to baye
        nn.Linear(language_model.config.dim, 1) # OOD classification head, change for other tasks
        # MLP: possibly bayesian

        # classification head
    )

def main(conf: Config):
    language_model = conf.env.make(torch.hub.load('/home/win32gg/Documents/transformers', 'model', conf.model, source="local"))
    tokenizer = torch.hub.load('/home/win32gg/Documents/transformers', 'tokenizer', conf.model, source="local") 

    train_dataset = conf.dataset.train_dataset.make(Split.TRAIN)
    iid_test_dataset = conf.dataset.train_dataset.make(Split.TEST)
    ood_test_dataset = conf.dataset.ood_detection_dataset.make(Split.TEST)


if __name__ == "__main__":
    utils.seed(42)
    utils.boost()

    conf = Config.load(Path("finetuning.yml"))

    dist.launch(
        main,
        conf.env.n_gpu,
        conf.env.n_machine,
        conf.env.machine_rank,
        conf.env.dist_url,
        args=(conf, )
    )