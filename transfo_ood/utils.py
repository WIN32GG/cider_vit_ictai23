import torch

from typing import NewType, Any
from enum import Enum

TOKENIZER_MASK = '[MASK]'
ZERO = torch.tensor(0.)
LOG_DIR = "./runs"

HyperParam = NewType('HyperParam', Any)

class Modality(Enum):
    TEXTUAL = "txt"
    VISUAL  = "img"