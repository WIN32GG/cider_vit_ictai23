from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchbooster.utils as utils
import torchbooster.distributed as dist
import logging

from tqdm import tqdm
from transfo_ood.config import Config
from transfo_ood.nn.model import GeneralSequential
from transfo_ood.preparator import DataPreparator

from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import (
    MetricCollection,
    Accuracy, 
    Precision, 
    Recall, 
    F1Score, 
    AUROC, 
    CalibrationError
)


class ModelEvaluator():
    """
    V2: Use ood_evaluators
    """

    @staticmethod
    def unfreeze(module: nn.Module) -> nn.Module:
        for param in module.parameters():
            param.requires_grad = True
        return module

    def __init__(self, conf: Config, model: nn.Module, train_dataset: DataLoader, test_dataset: DataLoader, preparator: DataPreparator, writer: SummaryWriter) -> None:

        self.conf: Config                               = conf
        self.model: nn.Module                        = model
        self.train_dataset: DataLoader                  = train_dataset
        self.test_dataset: DataLoader                   = test_dataset
        # self.combined_ood_train_dataset: DataLoader     = combined_ood_train_dataset
        # self.combined_ood_test_dataset: DataLoader      = combined_ood_test_dataset
        self.preparator: DataPreparator                 = preparator
        self.writer: SummaryWriter                      = writer
        self.reducer: int                               = 64
        self.crit: nn.Module                            = nn.CrossEntropyLoss()

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

    def __call__(self, steps: int = 0, epoch_fraction: float = .1) -> dict[str, float]:
        id_metrics = {}
        ood_metrics = {}

        # TODO test if re-training setting
        # utils.freeze(self.model)
        
        # train on id data to get task perforamance statistics
        id_model    = self.get_model_for_id_classification()
        optim       = torch.optim.AdamW(id_model.parameters())

        self.train_model_for_id_classification(id_model, optim, epoch_fraction)
        id_metrics = self.evaluate_id_performance(id_model, steps)
        print(id_metrics)

        #NOTE: wrong: train on joint id/ood data to evaluate calssifier
        #      this is incorrect, we use ood_classifier now
        # if ood:            
        #     ood_model   = self.get_model_for_ood_classification()
        #     optim       = torch.optim.AdamW(ood_model.parameters())

        #     self.train_model_for_ood_classification(ood_model, optim, epoch_fraction)
        #     ood_metrics = self.evaluate_ood_performance(ood_model, steps)
        #     print(ood_metrics)
        #     self.print_projector(steps, "OOD_Projector")

        ood_model    = self.get_model_for_id_classification() # same model as ID classification

            

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

    def _evaluate(self, model: nn.Module, dataset: DataLoader, msg: str, metrics: MetricCollection) -> float:
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

    def evaluate_ood_performance(self, model: nn.Module, steps: int = 0) -> dict[str, float]:
        return self._evaluate(model, self.combined_ood_test_dataset, "Evaluating for OOD classificaiton", self.metrics_ood)

    def evaluate_id_performance(self, model: nn.Module, steps: int = 0) -> dict[str, float]:
        return self._evaluate(model, self.test_dataset, "Evaluating for ID task", self.metrics_id)

    def _train(self, model: nn.Module, optim, metrics: MetricCollection, dataset: DataLoader, desc: str, msg: str, epoch_fraction: float = .1) -> None:
        metrics.reset()
        model = model.train()

        logging.info(msg)
        logging.warn("Epoch fraction works as epoch now, floating <1 will do a whole epoch")
        epochs = int(epoch_fraction) #FIXME wrong logic
        for e in range(epochs):
            logging.info(f'{desc} Epoch #{e}')
            pbar = tqdm(dataset, desc=desc)
            for i, batch in enumerate(pbar):
                # if i >= len(pbar) * epoch_fraction: return # early stop
                x, y = self.preparator.augment_and_prepare_batch(batch, augment=False)
                out  = self.preparator.forward(model, x) # bs x max_classes
                # print(out); print(y)
                loss = self.crit(out, y)
                m = metrics(out.detach().cpu(), y.detach().cpu())
                pbar.set_postfix(acc=f'{m["Accuracy"]:.2f}')
                utils.step(loss, optim)

    def train_model_for_id_classification(self, model: nn.Module, optim, epoch_fraction: float = .1) -> None:
        return self._train(model, optim, self.metrics_id, self.train_dataset, "ID_FT", "Train head for ID metrics", epoch_fraction)

    def train_model_for_ood_classification(self, model: nn.Module, optim, epoch_fraction: float = .1) -> None:
        raise NotImplementedError() #NOTE no longer train model on ood data
        return self._train(model, optim, self.metrics_ood, self.combined_ood_train_dataset, "OOD_FT", "Train head for OOD task", epoch_fraction)
       
    def get_model_for_ood_classification(self) -> nn.Module:
        # return self.conf.env.make(GeneralSequential(
        #     self.model,
        #     nn.SiLU(inplace=True),
        #     nn.Linear(self.conf.model.projection_size, 2) # 0: ID 1: OOD
        # ))
        s = self.conf.model.projection_size//self.reducer
        return self.conf.env.make(GeneralSequential(
            self.model,
            nn.SiLU(inplace=True),
            nn.Dropout(self.conf.model.dropout_p),
            nn.Linear(self.conf.model.projection_size, s),
            nn.SiLU(inplace=True),
            nn.Dropout(self.conf.model.dropout_p),
            nn.Linear(s, s),
            nn.SiLU(inplace=True),
            nn.Dropout(self.conf.model.dropout_p),
            nn.Linear(s, 2) # 0: ID, 1: OOD
        ))

    def get_model_for_id_classification(self) -> nn.Module:
        # print(self.conf.model.projection_size//4); exit();
        s = self.conf.model.projection_size//self.reducer
        return self.conf.env.make(GeneralSequential(
            self.model,
            nn.SiLU(inplace=True),
            nn.Dropout(self.conf.model.dropout_p),
            nn.Linear(self.conf.model.projection_size, s),
            nn.SiLU(inplace=True),
            nn.Dropout(self.conf.model.dropout_p),
            nn.Linear(s, s),
            nn.SiLU(inplace=True),
            nn.Dropout(self.conf.model.dropout_p),
            nn.Linear(s, self.preparator.max_classes)
        ))

    def print_projector(self, steps: int = 0, tag_name = "Projector") -> None:
        if not dist.is_primary(): return

        logging.info(f"Printing projector state to tensoboard with tag \"{tag_name}\"")
        model = self.model.eval()
        with torch.no_grad():

            images = {}
            data_embeddings, labels = [], []
            inp = []
            pbar = tqdm(self.test_dataset)
            for _, batch in enumerate(pbar):
                x, y = self.preparator.augment_and_prepare_batch(batch, augment=False) 
                inp.append(x)
                out = F.normalize(self.preparator.forward(model, x), dim=1)

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
            # imgs = []
            # for l in images:
            #     stacked = torch.stack(images[l])
            #     imgs.append(torch.mean(stacked, dim=0))
            # print_last_layer_image(torch.stack(imgs), writer, conf)
            
            mat = torch.cat(data_embeddings, dim=0)
            img_labels = None# torch.cat(inp) if isinstance(preparator, ImageDataPreparator) else None
            self.writer.add_embedding(mat, labels, tag=tag_name, global_step=steps, label_img=img_labels)