import torch

from typing import NewType, Any

TOKENIZER_MASK = '[MASK]'
ZERO = torch.tensor(0.)
LOG_DIR = "./runs"

HyperParam = NewType('HyperParam', Any)