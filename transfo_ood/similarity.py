"""
similarity.py
Compute similaritues between datasets for comparison purposes
"""
import torch

from torch.utils.data import DataLoader
from transfo_ood.config import Config
from transfo_ood.preparator import DataPreparator, TextDataPreparator
from transfo_ood.utils import Modality
from typing import Any
from tqdm import tqdm

from torch import nn

from torchvision.models import vgg16
from torchvision.models import VGG16_Weights

class Similarity:

    def __init__(self, modality: Modality, preparator: DataPreparator, dataset1: DataLoader, dataset2: DataLoader) -> None:
        self.preparator: DataPreparator = preparator
        self.dataset1: DataLoader = dataset1
        self.dataset2: DataLoader = dataset2
        self.modality: Modality = modality

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.similarity()

    def similarity(self) -> torch.Tensor:
        raise NotImplementedError()


class FeatureDistanceSimilarity(Similarity):

    def __init__(self, modality: Modality, preparator: DataPreparator, dataset1: DataLoader, dataset2: DataLoader, comparator: nn.Module = None) -> None:
        super().__init__(modality, preparator, dataset1, dataset2)
        if comparator is not None:
            self.comparator = comparator

    def set_comparator(self, comparator: nn.Module) -> Similarity:
        self.comparator = comparator.eval()
        return self

    def _dataset_high_level_stats(self, dataset: DataLoader, model: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = torch.stack([self.preparator(batch, model, augment=False)[0].detach().cpu() for batch in tqdm(dataset)])
        return outputs.mean(0), outputs.std(0)
    
    def similarity(self):
        assert self.comparator is not None, "Comparator undefined"
        m1, s1  = self._dataset_high_level_stats(self.dataset1, self.comparator) #FIXME: do something with s1
        m2, s2  = self._dataset_high_level_stats(self.dataset2, self.comparator) #FIXME: do something with s2

        return torch.sqrt(torch.pow(m1 - m2, 2)).mean()

class TextBertFeaturesDistanceSimilarity(FeatureDistanceSimilarity):
    def __init__(self, prep: DataPreparator, dataset1: DataLoader, dataset2: DataLoader, conf: Config) -> None:
        super().__init__(Modality.TEXTUAL, prep, dataset1, dataset2)

        comparator_backbone = "bert-base-uncased"
        self.preparator = TextDataPreparator(torch.hub.load('huggingface/transformers', 'tokenizer', comparator_backbone), conf) 
        self.set_comparator(torch.hub.load('huggingface/transformers', 'model', comparator_backbone))

class ImageVGG16FeaturesDistanceSimilarity(FeatureDistanceSimilarity):
    def __init__(self, prep: DataPreparator, dataset1: DataLoader, dataset2: DataLoader) -> None:
        super().__init__(Modality.VISUAL, prep, dataset1, dataset2, vgg16(weights = VGG16_Weights.DEFAULT).features)
