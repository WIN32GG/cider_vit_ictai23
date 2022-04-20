from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import chain
from pathlib import Path
import random
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

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from random import random as rand
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

class BayeMethod(str, Enum):
    BAYE_BY_BACKPROP = "baye_by_backprop"
    BAYESIAN_DROPOUT = "dropout"
    FREQUENTIST      = "frequentist"

@dataclass
class MyDataset(BaseConfig):
    train_dataset: DatasetConfig
    ood_detection_dataset: DatasetConfig

    input_features: int = 1 # used for CNN models
    label_position: int = 0
    input_position: int = 1
    max_classes: int = 2
    

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
    data_type: str
    force_make_table_dataset: bool

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

        return o

def get_backbone_model_output_features(backbone_model, conf):
    if isinstance(backbone_model, DistributedDataParallel):
        backbone_model = backbone_model.module
    if hasattr(backbone_model, "config"): # probably a HuggingFace model
        return backbone_model.config.dim * conf.tokenizer_max_length #! make gen 
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

    base_projector = nn.Sequential(
        # nn.SiLU(),
        nn.Linear(get_backbone_model_output_features(backbone_network, conf), conf.model.projection_hidden), # TODO switch to baye
        nn.SiLU(), #TODO param?
        nn.Dropout(p = conf.model.dropout_p),
        nn.Linear(conf.model.projection_hidden, conf.model.projection_size), # TODO switch to baye
    )
    
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
    def __init__(self, dataset_len, conf, max_classes=-1) -> None:
        super().__init__(dataset_len, conf, max_classes)
        self.augmenter = self.get_augmenter()
        self.id_augmenter = T.ToTensor()

    def get_augmenter(self):
        return T.Compose([
            T.RandomRotation(45),
            # T.RandomCrop(10),
            T.ToTensor()
        ])

    def forward(self, model: torch.Module, data: Any):
        return model(data)

    def augment_and_prepare_batch(self, batch, augment=True):
        if augment:
            new_batch = [ (conf.env.make(self.augmenter(a[conf.dataset.input_position])), a[conf.dataset.label_position]) for a in batch ]
        else:
            new_batch = [ (conf.env.make(self.id_augmenter(a[conf.dataset.input_position])), a[conf.dataset.label_position]) for a in batch ]

        X, Y = [], []
        for e in new_batch:
            X.append(e[0])
            Y.append(e[1])
        
        return conf.env.make(torch.stack(X)), Y

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
        return self.conf.env.make(utils.stack_dictionaries([conf.env.make(utils.to_tensor(self.tokenizer(s, truncation=True, padding='max_length', max_length=self.conf.tokenizer_max_length, **kwargs))) for s in strs]))

    def get_augmenter(self, samples: int):
        strength = self.conf.text_shift_stength #rand() #! change me    
        # augmenter1 = naw.AntonymAug(aug_p=strength/2)
        augmenter2 = naw.SynonymAug(aug_p=strength)    
            
        def mask_augment(inp: str, strength: float):
            tok = inp.split()
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
            new_batch.append([elem[0], augmenter(elem[1])])

        x, y = [], []
        for elem in new_batch:
            x.append(elem[conf.dataset.input_position])
            y.append(elem[conf.dataset.label_position])

        return self.tokenize_and_make(x), y

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

def print_projector(conf: Config, model: torch.Module, test_dataset, preparator: DataPreparator, writer: SummaryWriter, steps: int = 0):
    if not dist.is_primary(): return

    logging.info("Printing projector state to tensoboard")
    model = model.eval()
    with torch.no_grad():

        data_embeddings, labels = [], []
        inp = []
        pbar = tqdm(test_dataset)
        for _, batch in enumerate(pbar):
            x, y = preparator.augment_and_prepare_batch(batch, augment=False) 
            inp.append(x)
            data_embeddings += [F.normalize(preparator.forward(model, x), dim=1)]
            labels += y

        mat = torch.cat(data_embeddings, dim=0)

        img_labels = None# torch.cat(inp) if isinstance(preparator, ImageDataPreparator) else None
        writer.add_embedding(mat, labels, tag="Projector", global_step=steps, label_img=img_labels)

    # os.makedirs(f'{LOG_DIR}/embeddings', exist_ok=True)
    # with open(f'{LOG_DIR}/embeddings/feature_vecs.tsv', 'w') as fw:
    #     csv_writer = csv.writer(fw, delimiter='\t')
    #     csv_writer.writerows(tqdm(data_embeddings, desc="write::embedding"))
    # with open(f'{LOG_DIR}/embeddings/metadata.tsv', 'w') as file: 
    #     for label in tqdm(labels, desc="write::labels"):
    #         file.write(f"{label}\n")


    # pc = projector.ProjectorConfig()
    # embedding = pc.embeddings.add()


def fit(conf: Config, model, preparator: DataPreparator, dataset, test_dataset, optim, scheduler, prototypes):
    logging.info("Start training")

    temp = conf.temp #.1 #! TODO config

    writer = SummaryWriter() if dist.is_primary() else None
    model = conf.env.make(model)

    C = preparator.max_classes

    alpha = conf.alpha #0.95 #! TODO conf.prototype_shift
    print_projector(conf, model, test_dataset, preparator, writer, steps=0)
    for epoch in range(conf.epochs):
        pbar = tqdm(dataset, disable=not dist.is_primary())
        model = model.train()
        for batch_num, batch in enumerate(pbar):
            step = len(dataset)*epoch + batch_num
            x,y = preparator.augment_and_prepare_batch(batch)
            out = F.normalize(preparator.forward(model, x), dim=1) # bs x ProjectorSize

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
                        torch.dot(prototypes[i], prototypes[j])/temp)
                            for j in range(C) if i != j ]
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

def main(conf: Config):
    logging.info("Setting base config")
    utils.seed(42)
    utils.boost(not conf.debug)

    logging.info(f'Loading dataset: Loading raw data')
    train_dataset = conf.dataset.train_dataset.make(Split.TRAIN) # acceptance_fn=lambda x: len(x.strip()) > 0
    test_dataset  = conf.dataset.train_dataset.make(Split.TEST)  # acceptance_fn=lambda x: len(x.strip()) > 0
    if conf.force_make_table_dataset:
        train_dataset = to_map_style_dataset(train_dataset)
        test_dataset  = to_map_style_dataset(test_dataset)

    logging.info("Loading data preparator")
    if conf.data_type == "img":
        prep = ImageDataPreparator(len(train_dataset), conf, max_classes=conf.dataset.max_classes)
    else:
        logging.info(f'Loading tokenizer')
        tokenizer = torch.hub.load('huggingface/transformers', 'tokenizer', conf.model.backbone_network)
        prep = TextDataPreparator(len(train_dataset), tokenizer, conf, max_classes=conf.dataset.max_classes) 

    logging.info(f'Loading dataset: Making Data Loaders')
    train_dataset = conf.loader.make(train_dataset, shuffle=not conf.env.distributed, distributed=conf.env.distributed, collate_fn=prep.collate_fn)
    test_dataset  = conf.loader.make(test_dataset, shuffle=False, distributed=conf.env.distributed, collate_fn=prep.collate_fn)
    # iid_test_dataset = conf.dataset.train_dataset.make(Split.TRAIN)
    # ood_test_dataset = conf.dataset.ood_detection_dataset.make(Split.TRAIN)
    # DistributedIterableDataset

    logging.info(f'Loading model')
    if conf.data_type == "img":
        logging.info("Loading CNN based backbone")
        backbone_network = load_cnn_backbone(conf.model.backbone_network.split("cnn/")[1])
    else:
        logging.info("Loading Transformer based backbone")
        backbone_network = torch.hub.load('huggingface/transformers', 'model', conf.model.backbone_network)

    backbone_network = conf.env.make(backbone_network)
    model = conf.env.make(CustomModel(conf, backbone_network, get_projector(backbone_network, conf), preparator=prep))

    prototypes = [torch.nn.parameter.Parameter(conf.env.make(F.normalize(torch.rand(conf.model.projection_size), dim=0))) for _ in range(prep.max_classes)]

    optim = conf.optim.make(chain(model.parameters(), prototypes))
    scheduler = conf.scheduler.make(optim)

    fit(conf, model, prep, train_dataset, test_dataset, optim, scheduler, prototypes=prototypes)


if __name__ == "__main__":
    logging.info("Starting")

    if dist.is_primary():
        nltk.download('averaged_perceptron_tagger')
        nltk.download('wordnet')
        nltk.download('omw-1.4')

    conf = Config.load(Path(f"configs/{sys.argv[1]}.yml"))
    torch.multiprocessing.set_start_method('spawn')
    dist.launch(
        main,
        conf.env.n_gpu,
        conf.env.n_machine,
        conf.env.machine_rank,
        conf.env.dist_url,
        args=(conf, )
    )