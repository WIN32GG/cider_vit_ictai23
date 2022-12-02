import torch.nn as nn
import torch

from transfo_ood.config import Config
from transfo_ood.preparator import DataPreparator
from torch.utils.data import DataLoader
from typing import Any, Generator
from tqdm import tqdm

class OODEvaluator():
    """
        Base Class for OOD Evaluators
    """
    def __init__(self, config: Config, preparator: DataPreparator, model: nn.Model, id_dataset: DataLoader, ood_dataset: DataLoader, **kwargs) -> None:
        self.model: nn.Module = model
        self.preparator: DataPreparator = preparator
        self.id_dataset: DataLoader = id_dataset
        self.ood_dataset: DataLoader = ood_dataset
        self.config: Config = config

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass

    def _iter_dataaset(self, dataset) -> Generator[torch.Tensor, torch.Tensor]:
        for x, y in tqdm(dataset):
            probs = self.preparator.forward(self.model, x)
            yield probs, y
    
    def _iter_id_probs(self) -> Generator[torch.Tensor, torch.Tensor]:
        return self._iter_dataaset(self.id_dataset)

    def _iter_ood_probs(self) -> Generator[torch.Tensor, torch.Tensor]:
        return self._iter_dataaset(self.ood_dataset)
        
    def compute_ood_stats(self) -> tuple[int, int, int, int]: # tp, fp, tn, fn
        tp, fp, tn, fn = 0
        for p, y in self._iter_id_probs():
            #p-> B,C
            c = self.count_ood(p)
            fp += c; tn += p.size()[0] - c
        for p, y in self._iter_ood_probs():
            #p-> B,C
            c = self.count_ood(p)
            tp += c; fp += p.size()[0] - c
        return tp, fp, tn, fn
    
    def count_ood(self, sample: torch.Tensor) -> int:
        raise NotImplementedError()

class EVM(OODEvaluator):
    pass #TODO

class LLR(OODEvaluator):
    """Likelyhood ratio
        https://arxiv.org/abs/1906.02845

        Using the finedtuned model as the base and original model as background
    """
    def __init__(self, config: Config, preparator: DataPreparator, model: nn.Model, model_background: nn.Model, id_dataset: DataLoader, ood_dataset: DataLoader, **kwargs) -> None:
        super().__init__(config, preparator, model, id_dataset, ood_dataset, **kwargs)
        self.model_background = model_background
    
    
    def _iter_dataaset(self, dataset) -> Generator[torch.Tensor, torch.Tensor]:
        # NOTE Redefinition to yield probas for model1 and model_background
        for x, y in tqdm(dataset):
            probs_model      = self.preparator.forward(self.model, x)
            probs_background = self.preparator.forward(self.model_background, x)
            yield probs_model, probs_background

    def compute_ood_stats(self) -> tuple[int, int, int, int]: # tp, fp, tn, fn
        tp, fp, tn, fn = 0
        for p_m, p_back in self._iter_id_probs():
            #p_m, p_back-> B,C
            ratios = torch.log(torch.divide(p_m, p_back)).mean(1)
            c = self.count_ood(ratios)
            fp += c; tn += ratios.size(0) - c
        for p_m, p_back in self._iter_ood_probs():
            #p-> B,C
            ratios = torch.log(torch.divide(p_m, p_back)).mean(1)
            c = self.count_ood(ratios)
            tp += c; fp += ratios.size(0) - c
        return tp, fp, tn, fn

class MaxProb(OODEvaluator):
    def __init__(self, config: Config, preparator: DataPreparator, model: nn.Model, id_dataset: DataLoader, ood_dataset: DataLoader, **kwargs) -> None:
        super().__init__(config, preparator, model, id_dataset, ood_dataset, **kwargs)
        self.threshold = .7

    def count_ood(self, p: torch.Tensor) -> int:
        return (p.amax(dim=1) < self.threshold).sum()
            

class Entropy(OODEvaluator):
    def __init__(self, config: Config, preparator: DataPreparator, model: nn.Model, id_dataset: DataLoader, ood_dataset: DataLoader, **kwargs) -> None:
        super().__init__(config, preparator, model, id_dataset, ood_dataset, **kwargs)
        self.threshold = 1.

    def count_ood(self, p: torch.Tensor) -> int:
        return (torch.sum(-p*p.log(), dim=1) > self.threshold).sum()

        
class Mahalanobis(OODEvaluator):
    pass #TODO