from dataclasses import dataclass
from enum import Enum
from torchbooster.config import (
    BaseConfig,
    DatasetConfig,
    EnvironementConfig,
    LoaderConfig,
    OptimizerConfig,
    SchedulerConfig,
)

class BayeMethod(str, Enum):
    BAYE_BY_BACKPROP = "baye_by_backprop"
    BAYESIAN_DROPOUT = "dropout"
    FREQUENTIST      = "frequentist"

@dataclass
class MyDataset(BaseConfig):
    train_dataset: DatasetConfig
    ood_detection_dataset: DatasetConfig

    input_features: int = 1 # used for CNN models
    
    input_position: int = 1
    label_position: int = 0

    ood_input_position: int = 1
    ood_label_position: int = 0

    target_size: list[int, int] = None

    max_classes: int = 2
    



@dataclass
class ModelParam(BaseConfig):
    backbone_network: str
    projection_size: int
    projection_hidden: int
    dropout_p: float
    projector: str = "mlp"
    output_features: int = -1

@dataclass
class Config(BaseConfig):
    seed: int  # seed for reproductibility
    freeze_backbone: bool
    debug: bool
    epochs: int
    samples: int
    noise_samples: int
    clip: float
    data_type: str
    force_make_table_dataset: bool
    trainer: str

    #Eval hp
    eval_train_epoch_fraction: float

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

    def hp(self, prefix: str = "hparams"):
        return {
            f'{prefix}/dataset/id_train': self.dataset.train_dataset.name + ("" if self.dataset.train_dataset.task is None else ':' + self.dataset.train_dataset.task),
            f'{prefix}/dataset/ood_detection': self.dataset.ood_detection_dataset.name + ("" if self.dataset.ood_detection_dataset.task is None else ':' + self.dataset.ood_detection_dataset.task),
            f'{prefix}/dataset/tokenizer_max_length': self.tokenizer_max_length,

            f'{prefix}/trainer/batch_size': self.loader.batch_size,
            f'{prefix}/trainer/epochs': self.epochs,
            f'{prefix}/trainer/seed': self.seed,
            f'{prefix}/trainer/name': self.trainer,

            f'{prefix}/scheduler/name': self.scheduler.name if self.scheduler is not None else "None",

            f'{prefix}/eval_train_epoch_fraction': self.eval_train_epoch_fraction,
            f'{prefix}/text_shift_stength': self.text_shift_stength,
            f'{prefix}/alpha': self.alpha,
            f'{prefix}/lambda_c': self.lambda_c,
            f'{prefix}/lambda_d': self.lambda_d,
            f'{prefix}/temp': self.temp,
            f'{prefix}/clip': self.clip,

            f'{prefix}/optim/learning_rate': self.optim.lr,
            f'{prefix}/optim/name': self.optim.name,

            f'{prefix}/model/freeze_backbone': self.freeze_backbone,
            f'{prefix}/model/backbone_network': self.model.backbone_network,
            f'{prefix}/model/projection_hidden': self.model.projection_hidden,
            f'{prefix}/model/projection_size': self.model.projection_size,
            f'{prefix}/model/dropout_p': self.model.dropout_p,
        }


