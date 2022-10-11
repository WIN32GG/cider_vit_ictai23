import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parallel import DistributedDataParallel

import torchvision

from transfo_ood.config import Config
from transfo_ood.preparator import DataPreparator


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
        logging.info(f'Flat size = {ModelFactory.get_backbone_model_output_features(backbone, conf)}')

    def forward(self, *kargs, **kwargs):
        o = self.backbone(*kargs, **kwargs)
        if hasattr(o, "last_hidden_state"): # transformers/ bert model
            o : torch.FloatTensor = o.last_hidden_state#.view(conf.loader.batch_size, self.flat_size)
            o = self.projector(o.flatten(1))
        else:
            o = self.projector(o)
        return F.normalize(o, dim=1)

class ModelFactory():

    @staticmethod
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

    def __init__(self, conf: Config, preparator: DataPreparator) -> None:
        self.conf: Config                = conf
        self.preparator: DataPreparator  = preparator

    def __call__(self) -> CustomModel:
        logging.info(f'Loading model')
        try:
            logging.info("Trying: Loading CNN based backbone")
            backbone_network = self.load_cnn_backbone(self.conf.model.backbone_network)
        except:
            logging.info("Trying: Loading Transformer based backbone")
            backbone_network = torch.hub.load('huggingface/transformers', 'model', self.conf.model.backbone_network)

        logging.info("Loading Full Model")
        backbone_network = self.conf.env.make(backbone_network)
        return self.conf.env.make(CustomModel(self.conf, backbone_network, self.get_projector(backbone_network), preparator=self.preparator))

    def load_cnn_backbone(self, model_name: str) -> nn.Module:
        if model_name == "raw_small":
            return nn.Sequential(
                nn.Conv2d(self.conf.dataset.input_features, 50, (3, 3)),
                nn.ReLU(inplace=True),
                nn.Conv2d(50, 50, (3, 3)),
                nn.ReLU(inplace=True),
                nn.Conv2d(50, 50, (3, 3)),
                nn.ReLU(inplace=True)
            )
        if model_name == "raw_large":
            return nn.Sequential( #! add MaxPool
                nn.Conv2d(self.conf.dataset.input_features, 256, (3, 3)),
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


    def get_projector(self, backbone_network: nn.Module) -> nn.Module:
        """get_projector

        Return untrained projector with the appropriate method

        Parameters
        ----------
        backbone_network : nn.Module
            The base backbone model that will be used, passed to return a matching projector
        conf : Config
            The torchbooster config

        Returns
        -------
        nn.Module
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

        if self.conf.model.projector == "identity":
            self.conf.model.projection_size = ModelFactory.get_backbone_model_output_features(backbone_network, self.conf) # auto set
            return nn.Identity()
        elif self.conf.model.projector == "mlp":
            base_projector = nn.Sequential(
                nn.Linear(ModelFactory.get_backbone_model_output_features(backbone_network, self.conf), self.conf.model.projection_hidden), # TODO switch to baye
                nn.SiLU(), #TODO param?
                nn.Dropout(p = self.conf.model.dropout_p),
                nn.Linear(self.conf.model.projection_hidden, self.conf.model.projection_size), # TODO switch to baye
                nn.SiLU()
            )
        elif self.conf.model.projector == 'simple':
            base_projector = nn.Sequential(
                nn.Linear(ModelFactory.get_backbone_model_output_features(backbone_network, self.conf), self.conf.model.projection_size), # TODO switch to baye
                nn.SiLU(),
            )
        else:
            raise RuntimeError("Unknown projector")
        
        return base_projector # Bayesian projectors later
        if conf.method == BayeMethod.FREQUENTIST.value or conf.method == BayeMethod.BAYESIAN_DROPOUT.value: # Train handles baye dropout
            return base_projector
        elif conf.method == BayeMethod.BAYE_BY_BACKPROP.value:
            logging.info("Using Baye by Backprop")
            return bayeformers.to_bayesian(base_projector) # TODO change ð or init better
        else:
            raise ValueError()
