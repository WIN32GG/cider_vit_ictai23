print("Starting...")
import warnings
from transfo_ood.evaluator import ModelEvaluator
from transfo_ood.nn.model import ModelFactory
from transfo_ood.trainer import ScratchTrainer, CiderTrainer

from transfo_ood.preparator import ImageDataPreparator, TextDataPreparator, dict_hash, log_params, make_ood_dataset
warnings.filterwarnings("ignore")
import dataclasses
import string
import random
import torchinfo
import os
import csv
import sys
import nlpaug.augmenter.sentence as nas
import logging
import torchvision
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
import hashlib
import json
import rich
import transfo_ood.trainer
import argparse
import transfo_ood.similarity as sim

from argparse import ArgumentParser
from transfo_ood.config import Config
from dataclasses import dataclass
from typing import NewType, Tuple
from enum import Enum
from itertools import chain
from pathlib import Path
from typing import Any, List, TypeVar, Union
from torchbooster.dataset import Split
from torch.nn.parallel import DistributedDataParallel
from tensorboard.plugins import projector
from torchbooster.config import (
    BaseConfig,
    DatasetConfig,
    EnvironementConfig,
    LoaderConfig,
    OptimizerConfig,
    SchedulerConfig,
)
from random import random as rand
from torch.utils.data import DataLoader, Dataset
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score, AUROC, CalibrationError
from torchtext.data.functional import to_map_style_dataset
from typing import Dict, Any
from torch.utils.tensorboard import SummaryWriter

try:
    from rich import pretty
    from rich.traceback import install
    install(show_locals=False)
    pretty.install()
    from rich.progress import track
    # from tqdm.rich import tqdm
    from tqdm import tqdm
    
except ImportError:
    from tqdm import tqdm



def main(conf: Config, args) -> None:
    logging.info("Setting base config")
    utils.seed(conf.seed)
    utils.boost(not conf.debug)

    logging.info(f'Loading dataset: Loading raw data')
    train_dataset = conf.dataset.train_dataset.make(Split.TRAIN) # acceptance_fn=lambda x: len(x.strip()) > 0
    test_dataset  = conf.dataset.train_dataset.make(Split.TEST)  # acceptance_fn=lambda x: len(x.strip()) > 0
    if conf.force_make_table_dataset:
        train_dataset = to_map_style_dataset(train_dataset)
        test_dataset  = to_map_style_dataset(test_dataset)

    logging.info("Loading data preparator")
    if conf.data_type == "img":
        prep = ImageDataPreparator(conf, max_classes=conf.dataset.max_classes, target_size=conf.dataset.target_size)
    else:
        logging.info(f'Loading tokenizer')
        tokenizer = torch.hub.load('huggingface/transformers', 'tokenizer', conf.model.backbone_network)
        prep = TextDataPreparator(tokenizer, conf, max_classes=conf.dataset.max_classes) 

    ood_train_dataset = conf.dataset.ood_detection_dataset.make(Split.TRAIN)
    ood_test_dataset  = conf.dataset.ood_detection_dataset.make(Split.TEST)

    if args.dry_run:
        logging.info("======================= TEST DATASET INPUT SAMPLE =======================")
        print(prep.get_element(test_dataset[0], conf.dataset.input_position))
        logging.info("======================= OOD DATASET INPUT SAMPLE =======================")
        print(prep.get_element(ood_test_dataset[0], conf.dataset.ood_input_position))
        logging.info("=============================================")


    logging.info(f'Loading dataset: Making Data Loaders')
    # combined_ood_train_dataset =  conf.loader.make(make_ood_dataset(conf, train_dataset, ood_train_dataset), shuffle=True,  distributed=conf.env.distributed, collate_fn=prep.collate_fn)
    # combined_ood_test_dataset  =  conf.loader.make(make_ood_dataset(conf, test_dataset, ood_test_dataset),   shuffle=False, distributed=conf.env.distributed, collate_fn=prep.collate_fn)
    
    train_dataset = conf.loader.make(train_dataset, shuffle=not conf.env.distributed, distributed=conf.env.distributed, collate_fn=prep.collate_fn)
    test_dataset  = conf.loader.make(test_dataset, shuffle=False, distributed=conf.env.distributed, collate_fn=prep.collate_fn)
    # iid_test_dataset = test_dataset
    # DistributedIterableDataset

    model = ModelFactory(conf, prep)()

    print(conf.hp())
    if args.dry_run:
        torchinfo.summary(model)
        logging.warning("Dry run: exiting")
        exit(0)

    

    # CIDER prototypes TODO: move to loss
    prototypes = [torch.nn.parameter.Parameter(conf.env.make(F.normalize(torch.rand(conf.model.projection_size), dim=0))) for _ in range(prep.max_classes)]

    optim = conf.optim.make(chain(model.parameters(), prototypes))
    scheduler = conf.scheduler.make(optim)

    run_name = dict_hash(dataclasses.asdict(conf))
    print(f'Run Name {run_name}')
    
    rich.print_json(data=dataclasses.asdict(conf))
    writer = SummaryWriter(f'runs/{run_name}') if dist.is_primary() else None
    evaluator = ModelEvaluator(conf, model, train_dataset, test_dataset, combined_ood_train_dataset, combined_ood_test_dataset, prep, writer)

    if args.similarity:
        print(f'Running similarity')
        
        if conf.data_type == "img":
            val = sim.ImageVGG16FeaturesDistanceSimilarity(prep, test_dataset, ood_test_dataset)()
        else:
            val = sim.TextBertFeaturesDistanceSimilarity(prep, test_dataset, ood_test_dataset, conf)()
        print(f'Similarity is: {val}')
        writer.add_scalar("similarity", val)
        return

    trainer_class = getattr(transfo_ood.trainer, conf.trainer)
    trainer = trainer_class(conf, model, prep, train_dataset, test_dataset, optim, scheduler, writer, evaluator)

    trainer()
    log_params(conf, writer, evaluator(steps=0, epoch_fraction=conf.eval_train_epoch_fraction, iid=False, ood=True))


if __name__ == "__main__":
    parser = ArgumentParser()
    
    parser.add_argument("-d", "--dry_run", action="store_true", help="Dry run, print model architecture, dataset sample and exit")
    parser.add_argument("-c", "--config", required=True, help="Config file path to use (e.g: configs/txt/reviews.yml)")
    parser.add_argument("-s", "--similarity", action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    logging.info("Starting")
    if args.dry_run:
        logging.warning("Dry run ON")
    csv.field_size_limit(sys.maxsize)
    torch.multiprocessing.set_start_method('spawn')

    if dist.is_primary():
        nltk.download('averaged_perceptron_tagger')
        nltk.download('wordnet')
        nltk.download('omw-1.4')

    conf_gen = Config.load(Path(f"{args.config}"), hyperparams=True)
    for conf in conf_gen:
        conf: Config
        dist.launch(
            main,
            conf.env.n_gpu,
            conf.env.n_machine,
            conf.env.machine_rank,
            conf.env.dist_url,
            args=(conf, args)
        )