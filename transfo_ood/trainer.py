import logging
import torchbooster.utils as utils
import torchbooster.distributed as dist
import torch

from typing import Any
from transfo_ood.cider import Cider
from transfo_ood.config import Config
from transfo_ood.evaluator import ModelEvaluator
from transfo_ood.preparator import DataPreparator
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer


"""
Trainer is the strategy and what we want to test

CiderTrainer: Test cider loss for OOD detection improvment on textual/visual data
ScratchTrainer: Test LM OOD detection capabilities over time of training
"""

class Trainer:

    def __init__(self, conf: Config, model, preparator: DataPreparator, dataset, test_dataset, optim, scheduler, writer: SummaryWriter, evaluator: ModelEvaluator) -> None:
        self.conf: Config                    = conf
        self.model: nn.Module                = model
        self.preparator: DataPreparator      = preparator
        self.dataset: DataLoader             = dataset
        self.test_dataset: DataLoader        = test_dataset
        self.optimizer: Optimizer            = optim
        self.scheduler                       = scheduler
        self.writer: SummaryWriter           = writer
        self.evaluator: ModelEvaluator       = evaluator
        self.setup()

    def reset_model(self) -> "Trainer":
        def _reset_model(m):
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
        self.model.apply(_reset_model)
        return self

    def setup(self) -> None:
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        self.fit()

    def fit(self) -> None:
        raise NotImplementedError()


class IdentityTrainer(Trainer):
    """IdentityTrainer
    Keep the models as is and perform OOD evaluation_
    """

    def fit(self) -> None:
        pass

class ScratchTrainer(Trainer):
    """ScratchTrainer

    Train the model from scratch by reseting the parameters
    Traininig objective is wordmasking on "dataset" 
    TODO: implement other strategies? 
    """
    def setup(self):
        self.reset_model()

    def fit(self) -> None:
        logging.info("Start training")
        model = self.conf.env.make(model)
        
        for epoch in range(self.conf.epochs):
            pbar = tqdm(self.dataset, disable=not dist.is_primary())
            logging.info(f'Epoch {epoch}')
            
            for batch_num, batch in enumerate(pbar):

                step = len(self.dataset)*epoch + batch_num
                x, y = self.preparator.augment_and_prepare_batch(batch)
                out  = self.preparator.forward(model, x) # bs x ProjectorSize

class CiderTrainer(Trainer):
    """CiderTrainer

    FineTune model with the cider loss
    """

    def fit(self):
        logging.info("Start training")
        model = self.conf.env.make(model)

        cider_loss = Cider(self.conf)
        self.evaluator(steps = 0, epoch_fraction = self.conf.eval_train_epoch_fraction) # FIXME: move out
        
        for epoch in range(self.conf.epochs):
            pbar = tqdm(self.dataset, disable=not dist.is_primary())
            model = ModelEvaluator.unfreeze(model.train())
            logging.info(f'Epoch {epoch}')
            
            for batch_num, batch in enumerate(pbar):

                step = len(self.dataset)*epoch + batch_num
                x, y = self.preparator.augment_and_prepare_batch(batch)
                out  = self.preparator.forward(model, x) # bs x ProjectorSize

                l_dispersion, l_compactness = cider_loss(out, y)
                loss = self.conf.lambda_d * l_dispersion + self.conf.lambda_c * l_compactness

                utils.step(loss, self.optim, self.scheduler, clip=self.conf.clip)
                pbar.set_postfix(l_d=l_dispersion.detach().item(), l_c=l_compactness.detach().item())

                if dist.is_primary():
                    self.writer.add_scalar("loss/L_disper",  l_dispersion.detach().item(), step)
                    self.writer.add_scalar("loss/L_compact", l_compactness.detach().item(), step)
                    self.writer.add_scalar("loss/L_total",   loss.detach().item(), step)
                    self.writer.add_scalar("proto/std", torch.stack(cider_loss.prototypes).std(0).mean().detach().item(), step)

        # print_projector(conf, model, test_dataset, preparator, writer, steps=step)
        self.evaluator(steps = step, epoch_fraction = self.conf.eval_train_epoch_fraction) # FIXME: mlive out


