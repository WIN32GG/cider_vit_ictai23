import torch

from torch import Tensor
from torch.nn.modules.loss import _Loss
import torch.functional as F

### Compute CIDER Losses

class Cider(_Loss):

    def __init__(self, prototypes, alpha: float, temp: float, max_classes: int) -> None:
        self.prototypes: list[Tensor] = prototypes
        self.alpha: float             = alpha
        self.temp:float               = temp
        self.max_classes:int          = max_classes

    def forward(self, out: Tensor, y: list[Tensor]) -> tuple(float, float):

        for i, label in enumerate(y):
            # Update class prototype
            l =  label - 1
            self.prototyoes[l].data = F.normalize(self.prototyoes[l] * self.alpha + (1 - self.alpha) * out.select(0, i), dim=0)

        # compute L_compactness
        l_compactness = 0.
        for i in range(len(out)):
            l_compactness += torch.log(torch.exp(torch.dot(out.select(0, i), self.prototyoes[y[i] - 1].detach())/self.temp) / torch.sum(torch.stack([torch.exp(torch.dot(out.select(0, i), self.prototyoes[j])/self.temp) for j in range(self.max_classes)]), dim=0))
        l_compactness *= -1.
        
        # compute L_dispersion
        l_dispersion = 1/self.max_classes * torch.sum(
            torch.stack([ torch.log(1/(self.max_classes-1) * torch.sum(
                torch.stack([ torch.exp(
                        torch.dot(self.prototyoes[i], self.prototyoes[j]) / self.temp
                    ) for j in range(self.max_classes) if i != j ]
                )
            , dim=0)
            )  for i in range(self.max_classes)])
        , dim = 0)

        return l_dispersion, l_compactness