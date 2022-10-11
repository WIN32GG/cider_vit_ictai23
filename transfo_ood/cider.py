import torch

from torch import Tensor
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F

from transfo_ood.config import Config

### Compute CIDER Losses

class Cider():

    def __init__(self, conf: Config) -> None:
        self.alpha: float             = conf.alpha
        self.temp: float              = conf.temp
        self.max_classes: int         = conf.dataset.max_classes
        self.prototypes: list[Tensor] = [torch.nn.parameter.Parameter(conf.env.make(F.normalize(torch.rand(conf.model.projection_size), dim=0))) for _ in range(self.max_classes)]

    def __call__(self, out: Tensor, y: list[Tensor]) -> tuple[float, float]:

        for i, label in enumerate(y):
            # Update class prototype
            l =  label - 1
            self.prototypes[l].data = F.normalize(self.prototypes[l] * self.alpha + (1 - self.alpha) * out.select(0, i), dim=0)

        # compute L_compactness
        l_compactness = 0.
        for i in range(len(out)):
            l_compactness += torch.log(torch.exp(torch.dot(out.select(0, i), self.prototypes[y[i] - 1].detach())/self.temp) / torch.sum(torch.stack([torch.exp(torch.dot(out.select(0, i), self.prototypes[j])/self.temp) for j in range(self.max_classes)]), dim=0))
        l_compactness *= -1.
        
        # compute L_dispersion
        l_dispersion = 1/self.max_classes * torch.sum(
            torch.stack([ torch.log(1/(self.max_classes-1) * torch.sum(
                torch.stack([ torch.exp(
                        torch.dot(self.prototypes[i], self.prototypes[j]) / self.temp
                    ) for j in range(self.max_classes) if i != j ]
                )
            , dim=0)
            )  for i in range(self.max_classes)])
        , dim = 0)

        return l_dispersion, l_compactness