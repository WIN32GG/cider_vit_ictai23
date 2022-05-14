from __future__ import annotations
from argparse import ArgumentParser
import dataclasses
import string
from turtle import forward
from cv2 import transform

import tensorboard

try:
    from rich import pretty
    from rich.traceback import install
    install(show_locals=True)
    pretty.install()
    from rich.progress import track
    from tqdm.rich import tqdm
    
except ImportError:
    from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")
import rich
from dataclasses import dataclass
from typing import NewType, Tuple
from enum import Enum
from itertools import chain
from pathlib import Path
import random
import torchinfo
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


from torch.utils.tensorboard import SummaryWriter
from random import random as rand
from torch.utils.data import DataLoader, Dataset
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score, AUROC, CalibrationError
from torchtext.data.functional import to_map_style_dataset

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

TOKENIZER_MASK = '[MASK]'
ZERO = torch.tensor(0.)
LOG_DIR = "./runs"

HyperParam = NewType('HyperParam', Any)

class BayeMethod(str, Enum):
    BAYE_BY_BACKPROP = "baye_by_backprop"
    BAYESIAN_DROPOUT = "dropout"
    FREQUENTIST      = "frequentist"

@dataclass
class MyDataset(BaseConfig):
    train_dataset: DatasetConfig
    ood_detection_dataset: DatasetConfig

    input_features: int = 1 # used for CNN models
    
    input_position: int = 1
    label_position: int = 0

    ood_input_position: int = 1
    ood_label_position: int = 0

    target_size: list(int, int) = None

    max_classes: int = 2
    

@dataclass
class ModelParam(BaseConfig):
    backbone_network: str
    projection_size: int
    projection_hidden: int
    dropout_p: float
    projector: str = "mlp"
    output_features: int = -1

@dataclass
class Config(BaseConfig):
    seed: int  # seed for reproductibility
    freeze_backbone: bool
    debug: bool
    epochs: int
    samples: int
    noise_samples: int
    clip: float
    data_type: str
    force_make_table_dataset: bool

    #Eval hp
    eval_train_epoch_fraction: float

    #Cider param
    text_shift_stength: float
    alpha: float
    temp: float
    lambda_c: float
    lambda_d: float

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

    def hp(self, prefix: str = "hparams"):
        return {
            f'{prefix}/dataset/id_train': self.dataset.train_dataset.name + ("" if self.dataset.train_dataset.task is None else ':' + self.dataset.train_dataset.task),
            f'{prefix}/dataset/ood_detection': self.dataset.ood_detection_dataset.name + ("" if self.dataset.ood_detection_dataset.task is None else ':' + self.dataset.ood_detection_dataset.task),
            f'{prefix}/dataset/tokenizer_max_length': self.tokenizer_max_length,

            f'{prefix}/trainer/batch_size': self.loader.batch_size,
            f'{prefix}/trainer/epochs': self.epochs,
            f'{prefix}/trainer/seed': self.seed,

            f'{prefix}/scheduler/name': self.scheduler.name if self.scheduler is not None else "None",

            f'{prefix}/eval_train_epoch_fraction': self.eval_train_epoch_fraction,
            f'{prefix}/text_shift_stength': self.text_shift_stength,
            f'{prefix}/alpha': self.alpha,
            f'{prefix}/lambda_c': self.lambda_c,
            f'{prefix}/lambda_d': self.lambda_d,
            f'{prefix}/temp': self.temp,
            f'{prefix}/clip': self.clip,

            f'{prefix}/optim/learning_rate': self.optim.lr,
            f'{prefix}/optim/name': self.optim.name,

            f'{prefix}/model/freeze_backbone': self.freeze_backbone,
            f'{prefix}/model/backbone_network': self.model.backbone_network,
            f'{prefix}/model/projection_hidden': self.model.projection_hidden,
            f'{prefix}/model/projection_size': self.model.projection_size,
            f'{prefix}/model/dropout_p': self.model.dropout_p,
        }

class GeneralSequential(nn.Sequential):
    """GeneralSequential

    A Sequential that works, among other things, with HuggingFace way of passing inputs
    """

    def forward(self, *kargs, **kwargs):
        for i, module in enumerate(self):
            input = module(*kargs, **kwargs) if i == 0 else module(input)
        return input

class CustomModel(nn.Module):
    def __init__(self, conf: Config, backbone, projector, preparator: DataPreparator):
        super().__init__()
        self.conf: Config = conf
        self.backbone = backbone
        self.projector = projector
        self.preparator: DataPreparator = preparator
        if conf.freeze_backbone:
            logging.info(f'Backbone model is frozen')
            for param in self.backbone.parameters():
                param.requires_grad = False
        logging.info(f'Flat size = {get_backbone_model_output_features(backbone, conf)}')

    def forward(self, *kargs, **kwargs):
        o = self.backbone(*kargs, **kwargs)

        if hasattr(o, "last_hidden_state"): # transformers/ bert model
            o : torch.FloatTensor = o.last_hidden_state#.view(conf.loader.batch_size, self.flat_size)
            o = self.projector(o.flatten(1))
        else:
            o = self.projector(o)

        return F.normalize(o, dim=1)

def get_backbone_model_output_features(backbone_model, conf: Config):
    if conf.model.output_features > -1:
        return conf.model.output_features
    if isinstance(backbone_model, DistributedDataParallel):
        backbone_model = backbone_model.module
    if hasattr(backbone_model, "config"): # probably a HuggingFace model
        if hasattr(backbone_model.config, 'dim'):
            return backbone_model.config.dim * conf.tokenizer_max_length #! make gen 
        if hasattr(backbone_model.config, 'hidden_size'):
            return backbone_model.config.hidden_size * conf.tokenizer_max_length #! make gen 
        raise RuntimeError()
    return list(backbone_model.modules())[-1].out_features # for most torchvision models

def load_cnn_backbone(model_name):
    if model_name == "raw_small":
        return nn.Sequential(
            nn.Conv2d(conf.dataset.input_features, 50, (3, 3)),
            nn.ReLU(inplace=True),
            nn.Conv2d(50, 50, (3, 3)),
            nn.ReLU(inplace=True),
            nn.Conv2d(50, 50, (3, 3)),
            nn.ReLU(inplace=True)
        )
    if model_name == "raw_large":
        return nn.Sequential( #! add MaxPool
            nn.Conv2d(conf.dataset.input_features, 256, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU()
        )

    if hasattr(torchvision.models, model_name):
        #torchvision.models.resnet152()
        return getattr(torchvision.models, model_name)(pretrained=True, progress=True)
    raise RuntimeError("CNN model not found")


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

    # base_projector = nn.Sequential(
    #     nn.SiLU(),
    #     nn.Linear(get_backbone_model_output_features(backbone_network, conf), conf.model.projection_size), # TODO switch to baye
    #     # nn.SiLU(), #TODO param?
    #     # nn.Dropout(p = conf.model.dropout_p),
    #     # nn.Linear(conf.model.projection_hidden, conf.model.projection_size), # TODO switch to baye
    # )

    if conf.model.projector == "identity":
        conf.model.projection_size = get_backbone_model_output_features(backbone_network, conf) # auto set
        return nn.Identity()
    elif conf.model.projector == "mlp":
        base_projector = nn.Sequential(
            nn.Linear(get_backbone_model_output_features(backbone_network, conf), conf.model.projection_hidden), # TODO switch to baye
            nn.SiLU(), #TODO param?
            nn.Dropout(p = conf.model.dropout_p),
            nn.Linear(conf.model.projection_hidden, conf.model.projection_size), # TODO switch to baye
            nn.SiLU()
        )
    elif conf.model.projector == 'simple':
        base_projector = nn.Sequential(
            nn.Linear(get_backbone_model_output_features(backbone_network, conf), conf.model.projection_size), # TODO switch to baye
            nn.SiLU(),
        )
    else:
        raise RuntimeError("Unknown projector")
    
    return base_projector # Bayesian projectors in a second time
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

class DataPreparator():
    def __init__(self, dataset_len, conf, max_classes = -1) -> None:
        self.batch_num = 0
        self.conf: Config = conf
        self.dataset_len = dataset_len
        self.max_classes = max_classes

    @staticmethod
    def get_element(elem, pos):
        if isinstance(elem, dict):
            return elem[list(elem.keys())[pos]]
        else:
            return elem[pos]

    def augment_and_prepare_batch(self, batch, augment=True):
        """augment_and_prepare_batch

        Prepares a given batch to be processed in a model, the returned value should be immediatly usable
        by the model.

        Parameters
        ----------
        batch : Any
            The batch, usually a list of size batch_size

        Raises
        ------
        NotImplementedError
            This must be implemented in others Preparator
        """
        raise NotImplementedError("Not Implemented")

    def forward(self, model: torch.Module, data: Any):
        return model(data)

    def collate_fn(self, data):
        return data

class ImageDataPreparator(DataPreparator):
    def __init__(self, dataset_len, conf, max_classes=-1, target_size: Tuple[int] = None) -> None:
        super().__init__(dataset_len, conf, max_classes)
        self.resize_augmenter = [T.Resize(target_size)] if target_size is not None else []
        self.augmenter = self.get_augmenter()
        self.id_augmenter = T.Compose( [ *self.resize_augmenter, T.ToTensor()])
        if target_size is not None:
            logging.info(f"Target size for image input is {target_size}")

    def get_augmenter(self):
        return T.Compose([
            *self.resize_augmenter,
            T.RandomRotation(45),
            # T.RandomCrop(10),
            T.ToTensor()
        ])

    def forward(self, model: torch.Module, data: Any):
        return model(data)

    def augment_and_prepare_batch(self, batch, augment=True):
        if augment:
            new_batch = [ (conf.env.make(self.augmenter(self.get_element(a, conf.dataset.input_position))), 
                            self.get_element(a, conf.dataset.label_position)) for a in batch ]
        else:
            new_batch = [ (conf.env.make(self.id_augmenter(self.get_element(a, conf.dataset.input_position))), 
                            self.get_element(a, conf.dataset.label_position)) for a in batch ]

        X, Y = [], []
        for e in new_batch:
            X.append(e[0])
            Y.append(e[1])
        
        return conf.env.make(torch.stack(X)), conf.env.make(utils.to_tensor(Y)) 

class TextDataPreparator(DataPreparator):
    def __init__(self, dataset_len, tokenizer, conf, max_classes = -1) -> None:
        self.batch_num = 0
        self.dataset_len: int = dataset_len
        self.tokenizer = tokenizer
        self.conf: Config = conf
        self.max_classes = max_classes

    def tokenize_and_make(self, strs: Union[str, List[str]], **kwargs) -> torch.Tensor:
        if isinstance(strs, str):
            return self.conf.env.make(utils.to_tensor(self.tokenizer(strs, truncation=True, padding='max_length', max_length=self.conf.tokenizer_max_length, **kwargs))) # use return_tensors = pt and unsqueeze
        return self.conf.env.make(
                utils.stack_dictionaries(
                        [self.conf.env.make(utils.to_tensor(self.tokenizer(s, truncation=True, padding='max_length', max_length=self.conf.tokenizer_max_length, **kwargs))) for s in strs]))

    def get_augmenter(self, samples: int):
        strength = self.conf.text_shift_stength #rand() #! change me    
        # augmenter1 = naw.AntonymAug(aug_p=strength/2)
        augmenter2 = naw.SynonymAug(aug_p=strength)    
            
        def mask_augment(inp: str, strength: float):
            tok = inp.split()
            if len(tok) == 0:
                return ''
            for i in random.sample(range(len(tok)), int(strength*len(tok))+1):
                tok[i] = TOKENIZER_MASK
            return ' '.join(tok)

        def wrap(inp: str): #! implement n_samples
            x = mask_augment(inp, strength)
            # print(x)
            return x
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

    def augment_and_prepare_batch(self, batch, augment=True):
        augmenter = self.get_augmenter(1) if augment else lambda x: x
        new_batch = []
        for elem in batch:
            new_batch.append([ augmenter(self.get_element(elem, self.conf.dataset.input_position)), 
                               self.get_element(elem, self.conf.dataset.label_position)
                            ])

        x, y = [], []
        for elem in new_batch:
            x.append(self.get_element(elem, 0)) #self.conf.dataset.input_position))
            y.append(self.get_element(elem, 1)) #self.conf.dataset.label_position))

        return self.tokenize_and_make(x), conf.env.make(utils.to_tensor(y)) 

    def forward(self, model: torch.Module, data: Any):
        return model(**data)

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


class ModelEvaluator():

    @staticmethod
    def unfreeze(module: torch.Module) -> torch.Module:
        for param in module.parameters():
            param.requires_grad = True
        return module

    def __init__(self, conf: Config, model: torch.Module, train_dataset: DataLoader, test_dataset: DataLoader, combined_ood_train_dataset: DataLoader, combined_ood_test_dataset: Dataset, preparator: DataPreparator, writer: SummaryWriter) -> None:
        self.conf: Config = conf
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.combined_ood_train_dataset = combined_ood_train_dataset
        self.combined_ood_test_dataset = combined_ood_test_dataset
        self.preparator: DataPreparator = preparator
        self.writer: SummaryWriter = writer

        self.crit = nn.CrossEntropyLoss()

        self.metrics_id = MetricCollection([
            Accuracy( num_classes=preparator.max_classes, average='macro'),
            Precision(num_classes=preparator.max_classes, average='macro'),
            Recall(   num_classes=preparator.max_classes, average='macro'),
            F1Score(  num_classes=preparator.max_classes, average='macro'), 
            AUROC(    num_classes=preparator.max_classes, average='macro'), 
            # CalibrationError(n_bins=10, norm='l2')
        ])

        self.metrics_ood = MetricCollection([
            Accuracy( num_classes=2, average='macro'),
            Precision(num_classes=2, average='macro'),
            Recall(   num_classes=2, average='macro'),
            F1Score(  num_classes=2, average='macro'), 
            AUROC(    num_classes=2, average='macro'), 
            # CalibrationError(n_bins=10, norm='l2')
        ])

    def __call__(self, steps: int = 0, epoch_fraction: float = .1, iid: bool = True, ood: bool = True) -> Any:
        id_metrics = {}
        ood_metrics = {}

        utils.freeze(self.model)

        if iid:
            # ID EVALUATION
            id_model = self.get_model_for_id_classification()
            optim = torch.optim.AdamW(id_model.parameters())
            self.train_model_for_id_classification(id_model, optim, epoch_fraction)
            id_metrics = self.evaluate_id_performance(id_model, steps)
            print(id_metrics)

        if ood:
            # OOD evaluation
            ood_model = self.get_model_for_ood_classification()
            optim = torch.optim.AdamW(ood_model.parameters())
            self.train_model_for_ood_classification(ood_model, optim, epoch_fraction)
            ood_metrics = self.evaluate_ood_performance(ood_model, steps)
            print(ood_metrics)
            print_projector(conf, self.model, self.combined_ood_test_dataset, self.preparator, self.writer, steps, "OOD_Projector")

        if steps >= 0:
            if iid:
                self.writer.add_scalar(f'metrics/id_Accuracy', id_metrics['Accuracy'], global_step=steps)
                self.writer.add_scalar(f'metrics/id_AUROC',    id_metrics['AUROC'],    global_step=steps)
                self.writer.add_scalar(f'metrics/id_F1',       id_metrics['F1Score'],  global_step=steps)

            if ood:
                self.writer.add_scalar(f'metrics/ood_Accuracy', ood_metrics['Accuracy'], global_step=steps)
                self.writer.add_scalar(f'metrics/ood_AUROC',    ood_metrics['AUROC'],    global_step=steps)
                self.writer.add_scalar(f'metrics/ood_F1',       ood_metrics['F1Score'],  global_step=steps)
                

        # Unfreeze base model
        ModelEvaluator.unfreeze(self.model)

        return {
            'id_metrics': id_metrics,
            'ood_metrics': ood_metrics
        }

    def _evaluate(self, model: torch.Module, dataset: DataLoader, msg: str, metrics: MetricCollection):
        if not dist.is_primary(): return
        metrics.reset()
        model = model.eval()
        logging.info(msg)

        with torch.no_grad():
            pbar = tqdm(dataset)
            for batch in pbar:
                x, y = self.preparator.augment_and_prepare_batch(batch)
                out  = self.preparator.forward(model, x) # bs x max_classes
                # out  = F.softmax(out) # necessary?

                m = metrics(out.detach().cpu(), y.detach().cpu())
                pbar.set_postfix(acc=f'{m["Accuracy"]}')
        return metrics.compute()

    def evaluate_ood_performance(self, model: torch.Module, steps: int = 0) -> dict[str, float]:
        return self._evaluate(model, self.combined_ood_test_dataset, "Evaluating for OOD classificaiton", self.metrics_ood)

    def evaluate_id_performance(self, model: torch.Module, steps: int = 0):
        return self._evaluate(model, self.test_dataset, "Evaluating for ID task", self.metrics_id)

    def _train(self, model, optim, metrics: MetricCollection, dataset: DataLoader, desc: str, msg: str, epoch_fraction: float = .1):
        metrics.reset()
        model = model.train()

        pbar = tqdm(dataset, desc=desc)
        logging.info(msg)
        for i, batch in enumerate(pbar):
            if i >= len(pbar) * epoch_fraction: return # early stop
            x, y = self.preparator.augment_and_prepare_batch(batch, augment=False)
            out  = self.preparator.forward(model, x) # bs x max_classes
    
            loss = self.crit(out, y)
            m = metrics(out.detach().cpu(), y.detach().cpu())
            pbar.set_postfix(acc=f'{m["Accuracy"]}')
            utils.step(loss, optim)

    def train_model_for_id_classification(self, model: torch.Module, optim, epoch_fraction: float = .1):
        return self._train(model, optim, self.metrics_id, self.train_dataset, "ID_FT", "Train head for ID metrics", epoch_fraction)

    def train_model_for_ood_classification(self, model: torch.Module, optim, epoch_fraction: float = .1):
        return self._train(model, optim, self.metrics_ood, self.combined_ood_train_dataset, "OOD_FT", "Train head for OOD task", epoch_fraction)
       
    def get_model_for_ood_classification(self) -> torch.Module:
        return self.conf.env.make(GeneralSequential(
            self.model,
            nn.SiLU(inplace=True),
            nn.Linear(self.conf.model.projection_size, 2) # 0: ID 1: OOD
        ))

    def get_model_for_id_classification(self) -> torch.Module:
        return self.conf.env.make(GeneralSequential(
            self.model,
            nn.SiLU(inplace=True),
            nn.Linear(self.conf.model.projection_size, self.preparator.max_classes)
        ))

def print_projector(conf: Config, model: torch.Module, test_dataset, preparator: DataPreparator, writer: SummaryWriter, steps: int = 0, tag_name = "Projector"):
    if not dist.is_primary(): return

    logging.info(f"Printing projector state to tensoboard with tag \"{tag_name}\"")
    model = model.eval()
    with torch.no_grad():

        images = {}
        data_embeddings, labels = [], []
        inp = []
        pbar = tqdm(test_dataset)
        for _, batch in enumerate(pbar):
            x, y = preparator.augment_and_prepare_batch(batch, augment=False) 
            inp.append(x)
            out = F.normalize(preparator.forward(model, x), dim=1)

            # Separate images by labels 
            for i in range(len(y)):
                emb, lab = out[i], y[i]
                lab = lab.detach().cpu().item()
                if lab not in images:
                    images[lab] = []
                images[lab].append(emb)

            data_embeddings += [out]
            labels += y

        # Compute mean of activation by label and plot it
        imgs = []
        for l in images:
            stacked = torch.stack(images[l])
            imgs.append(torch.mean(stacked, dim=0))
        print_last_layer_image(torch.stack(imgs), writer, conf)
        
        mat = torch.cat(data_embeddings, dim=0)
        img_labels = None# torch.cat(inp) if isinstance(preparator, ImageDataPreparator) else None
        writer.add_embedding(mat, labels, tag=tag_name, global_step=steps, label_img=img_labels)

def fit(conf: Config, model, preparator: DataPreparator, dataset, test_dataset, optim, scheduler, prototypes, writer: SummaryWriter, evaluator: ModelEvaluator):
    logging.info("Start training")
    temp = conf.temp
    model = conf.env.make(model)
    C = preparator.max_classes
    alpha = conf.alpha

    print_projector(conf, model, test_dataset, preparator, writer, steps=0)
    evaluator(steps = 0, epoch_fraction = conf.eval_train_epoch_fraction)
    
    for epoch in range(conf.epochs):
        pbar = tqdm(dataset, disable=not dist.is_primary())
        model = model.train()
        logging.info(f'Epoch {epoch}')
        for batch_num, batch in enumerate(pbar):
            step = len(dataset)*epoch + batch_num
            x, y = preparator.augment_and_prepare_batch(batch)
            out  = preparator.forward(model, x) # bs x ProjectorSize

            ### Compute CIDER Losses

            for i, label in enumerate(y):
                # Update class prototype
                l =  label - 1
                prototypes[l].data = F.normalize(prototypes[l] * alpha + (1 - alpha) * out.select(0, i), dim=0)

            # compute L_compactness
            l_compactness = 0.
            for i in range(len(x)):
                l_compactness += torch.log(torch.exp(torch.dot(out.select(0, i), prototypes[y[i] - 1].detach())/temp) / torch.sum(torch.stack([torch.exp(torch.dot(out.select(0, i), prototypes[j])/temp) for j in range(C)]), dim=0))
            l_compactness *= -1.
            
            # compute L_dispersion
            l_dispersion = 1/C * torch.sum(
                torch.stack([ torch.log(1/(C-1) * torch.sum(
                    torch.stack([ torch.exp(
                            torch.dot(prototypes[i], prototypes[j]) /temp
                        ) for j in range(C) if i != j ]
                    )
                , dim=0)
                )  for i in range(C)])
            , dim = 0)

            loss = conf.lambda_d * l_dispersion + conf.lambda_c * l_compactness

            utils.step(loss, optim, scheduler, clip=conf.clip)
            pbar.set_postfix(l_d=l_dispersion.detach().item(), l_c=l_compactness.detach().item())

            if dist.is_primary():
                writer.add_scalar("loss/L_disper",  l_dispersion.detach().item(), step)
                writer.add_scalar("loss/L_compact", l_compactness.detach().item(), step)
                writer.add_scalar("loss/L_total",   loss.detach().item(), step)
                writer.add_scalar("proto/std", torch.stack(prototypes).std(0).mean().detach().item(), step)

        # Epoch finished, test it
        print_projector(conf, model, test_dataset, preparator, writer, steps=step)
        evaluator(steps = step, epoch_fraction = conf.eval_train_epoch_fraction)


class ListDataset(Dataset):

    def __init__(self, elems) -> None:
        super().__init__()
        self.list = elems

    def __len__(self):
        return len(self.list)
    
    def __getitem__(self, key):
        return self.list[key]

def make_ood_dataset(conf: Config, id_dataset, ood_dataset):
    def elem_with_place(x, y):
        return (x, y) if conf.dataset.input_position == 0 else (y, x)
    logging.info('Building OOD classifier dataset')

    dataset = [ elem_with_place(DataPreparator.get_element(e, conf.dataset.input_position), 0) for e in tqdm(id_dataset)] + \
              [ elem_with_place(DataPreparator.get_element(e, conf.dataset.ood_input_position), 1) for e in tqdm(ood_dataset)]
    random.shuffle(dataset)
    dataset = ListDataset(dataset)
    return dataset

def log_params(conf: Config, writer: SummaryWriter, metrics: dict[str, float]):
    def flatten_dict(d: dict, out = {}, path = 'hparams/'):
        for k in d:
            if isinstance(d[k], dict):
                flatten_dict(d[k], out, f'{path}{k}/')
            else:
                out[f'{path}{k}'] = d[k]
        return out

    writer.add_hparams(conf.hp(), flatten_dict(metrics))

from typing import Dict, Any
import hashlib
import json

def print_last_layer_image(data: torch.Tensor, writer: SummaryWriter, conf: Config, steps: int = 0) -> None:
    logging.info("Plotting last activation image")
    # data: batch_size x projector_size
    if len(data.shape) != 2:
        raise RuntimeError()
    d = data.shape[1]
    w = int(d**.5) + 1
    if d%w != 0:
        data = F.pad(data, (0, w**2 - d), 'constant', 0)
    data = data.view((-1, 1, w, w))
    writer.add_image("last_activation", torchvision.utils.make_grid(data, normalize=True), global_step=steps)    
    

def dict_hash(dictionary: Dict[str, Any]) -> str:
    """MD5 hash of a dictionary."""
    dhash = hashlib.md5()
    # We need to sort arguments so {'a': 1, 'b': 2} is
    # the same as {'b': 2, 'a': 1}
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()


def main(conf: Config, args):
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
        prep = ImageDataPreparator(len(train_dataset), conf, max_classes=conf.dataset.max_classes, target_size=conf.dataset.target_size)
    else:
        logging.info(f'Loading tokenizer')
        tokenizer = torch.hub.load('huggingface/transformers', 'tokenizer', conf.model.backbone_network)
        prep = TextDataPreparator(len(train_dataset), tokenizer, conf, max_classes=conf.dataset.max_classes) 

    ood_train_dataset = conf.dataset.ood_detection_dataset.make(Split.TRAIN)
    ood_test_dataset  = conf.dataset.ood_detection_dataset.make(Split.TEST)

    if args.dry_run:
        logging.info("======================= TEST DATASET INPUT SAMPLE =======================")
        print(prep.get_element(test_dataset[0], conf.dataset.input_position))
        logging.info("======================= OOD DATASET INPUT SAMPLE =======================")
        print(prep.get_element(ood_test_dataset[0], conf.dataset.ood_input_position))
        logging.info("=============================================")


    logging.info(f'Loading dataset: Making Data Loaders')
    combined_ood_train_dataset =  conf.loader.make(make_ood_dataset(conf, train_dataset, ood_train_dataset), shuffle=True,  distributed=conf.env.distributed, collate_fn=prep.collate_fn)
    combined_ood_test_dataset  =  conf.loader.make(make_ood_dataset(conf, test_dataset, ood_test_dataset),   shuffle=False, distributed=conf.env.distributed, collate_fn=prep.collate_fn)
    
    train_dataset = conf.loader.make(train_dataset, shuffle=not conf.env.distributed, distributed=conf.env.distributed, collate_fn=prep.collate_fn)
    test_dataset  = conf.loader.make(test_dataset, shuffle=False, distributed=conf.env.distributed, collate_fn=prep.collate_fn)
    # iid_test_dataset = test_dataset
    # DistributedIterableDataset

    logging.info(f'Loading model')
    # if conf.data_type == "img":
    try:
        logging.info("Trying: Loading CNN based backbone")
        backbone_network = load_cnn_backbone(conf.model.backbone_network)
    except:
        logging.info("Trying: Loading Transformer based backbone")
        backbone_network = torch.hub.load('huggingface/transformers', 'model', conf.model.backbone_network)

    logging.info("Loading Full Model")
    backbone_network = conf.env.make(backbone_network)
    model = conf.env.make(CustomModel(conf, backbone_network, get_projector(backbone_network, conf), preparator=prep))

    if args.dry_run:
        print(conf.hp())
        torchinfo.summary(model)
        logging.warning("Dry run: exiting")
        exit(0)

    # CIDER prototypes
    prototypes = [torch.nn.parameter.Parameter(conf.env.make(F.normalize(torch.rand(conf.model.projection_size), dim=0))) for _ in range(prep.max_classes)]

    optim = conf.optim.make(chain(model.parameters(), prototypes))
    scheduler = conf.scheduler.make(optim)

    run_name = dict_hash(dataclasses.asdict(conf))
    print(f'Run Name {run_name}')
    
    rich.print_json(data=dataclasses.asdict(conf))
    writer = SummaryWriter(f'runs/{run_name}') if dist.is_primary() else None
    evaluator = ModelEvaluator(conf, model, train_dataset, test_dataset, combined_ood_train_dataset, combined_ood_test_dataset, prep, writer)

    # print_projector(conf, model, combined_ood_test_dataset, prep, writer, 0, "OOD_Projector")
    # fit(conf, model, prep, train_dataset, test_dataset, optim, scheduler, prototypes, writer, evaluator)
    log_params(conf, writer, evaluator(steps=0, epoch_fraction=conf.eval_train_epoch_fraction, iid=False, ood=True))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--dry_run", action="store_true", help="Dry run, print model architecture, dataset sample and exit")
    parser.add_argument("-c", "--config", required=True, help="Config file path to use (e.g: configs/txt/reviews.yml)")

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