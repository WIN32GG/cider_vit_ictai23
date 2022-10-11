
import torch
import hashlib
import json
import logging
import torchvision
import torch.nn as nn
import random
import torch.functional as F
import torchvision.transforms as T
import nlpaug.augmenter.word as naw

from torchbooster import utils
from random import random as rand
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Any, Tuple, List, Union
from torch.utils.tensorboard import SummaryWriter
from transfo_ood.config import Config
from transfo_ood.utils import TOKENIZER_MASK



def dict_hash(dictionary: dict[str, Any]) -> str:
    """MD5 hash of a dictionary."""
    dhash = hashlib.md5()
    # We need to sort arguments so {'a': 1, 'b': 2} is
    # the same as {'b': 2, 'a': 1}
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()


def log_params(conf: Config, writer: SummaryWriter, metrics: dict[str, float]):
    def flatten_dict(d: dict, out = {}, path = 'hparams/'):
        for k in d:
            if isinstance(d[k], dict):
                flatten_dict(d[k], out, f'{path}{k}/')
            else:
                out[f'{path}{k}'] = d[k]
        return out

    writer.add_hparams(conf.hp(), flatten_dict(metrics))


# def print_last_layer_image(data: torch.Tensor, writer: SummaryWriter, conf: Config, steps: int = 0) -> None:
#     logging.info("Plotting last activation image")
#     # data: batch_size x projector_size
#     if len(data.shape) != 2:
#         raise RuntimeError()
#     d = data.shape[1]
#     w = int(d**.5) + 1
#     if d%w != 0:
#         data = F.pad(data, (0, w**2 - d), 'constant', 0)
#     data = data.view((-1, 1, w, w))
#     writer.add_image("last_activation", torchvision.utils.make_grid(data, normalize=True), global_step=steps)    
    

def make_ood_dataset(conf: Config, id_dataset, ood_dataset):
    def elem_with_place(x, y):
        return (x, y) if conf.dataset.input_position == 0 else (y, x)
    logging.info('Building OOD classifier dataset')

    dataset = [ elem_with_place(DataPreparator.get_element(e, conf.dataset.input_position), 0) for e in tqdm(id_dataset)] + \
              [ elem_with_place(DataPreparator.get_element(e, conf.dataset.ood_input_position), 1) for e in tqdm(ood_dataset)]
    random.shuffle(dataset)
    dataset = ListDataset(dataset)
    return dataset


class ListDataset(Dataset):

    def __init__(self, elems) -> None:
        super().__init__()
        self.list = elems

    def __len__(self):
        return len(self.list)
    
    def __getitem__(self, key):
        return self.list[key]


class DataPreparator():
    def __init__(self, dataset_len, conf, max_classes = -1) -> None:
        self.batch_num = 0
        self.conf: Config = conf
        self.dataset_len = dataset_len
        self.max_classes = max_classes

    @staticmethod
    def get_element(elem, pos):
        if isinstance(elem, dict):
            return elem[list(elem.keys())[pos]]
        else:
            return elem[pos]

    def augment_and_prepare_batch(self, batch, augment=True):
        """augment_and_prepare_batch

        Prepares a given batch to be processed in a model, the returned value should be immediatly usable
        by the model.

        Parameters
        ----------
        batch : Any
            The batch, usually a list of size batch_size

        Raises
        ------
        NotImplementedError
            This must be implemented in others Preparator
        """
        raise NotImplementedError("Not Implemented")

    def forward(self, model: nn.Module, data: Any):
        return model(data)

    def collate_fn(self, data):
        return data

class ImageDataPreparator(DataPreparator):
    def __init__(self, dataset_len, conf, max_classes=-1, target_size: Tuple[int] = None) -> None:
        super().__init__(dataset_len, conf, max_classes)
        self.resize_augmenter = [T.Resize(target_size)] if target_size is not None else []
        self.augmenter = self.get_augmenter()
        self.id_augmenter = T.Compose( [ *self.resize_augmenter, T.ToTensor()])
        if target_size is not None:
            logging.info(f"Target size for image input is {target_size}")

    def get_augmenter(self):
        return T.Compose([
            *self.resize_augmenter,
            T.RandomRotation(45),
            # T.RandomCrop(10),
            T.ToTensor()
        ])

    def forward(self, model: nn.Module, data: Any):
        return model(data)

    def augment_and_prepare_batch(self, batch, augment=True):
        if augment:
            new_batch = [ (self.conf.env.make(self.augmenter(self.get_element(a, self.conf.dataset.input_position))), 
                            self.get_element(a, self.conf.dataset.label_position)) for a in batch ]
        else:
            new_batch = [ (self.conf.env.make(self.id_augmenter(self.get_element(a, self.conf.dataset.input_position))), 
                            self.get_element(a, self.conf.dataset.label_position)) for a in batch ]

        X, Y = [], []
        for e in new_batch:
            X.append(e[0])
            Y.append(e[1])
        
        return self.conf.env.make(torch.stack(X)), self.conf.env.make(utils.to_tensor(Y)) 

class TextDataPreparator(DataPreparator):
    def __init__(self, dataset_len, tokenizer, conf, max_classes = -1) -> None:
        self.batch_num = 0
        self.dataset_len: int = dataset_len
        self.tokenizer = tokenizer
        self.conf: Config = conf
        self.max_classes = max_classes

    def tokenize_and_make(self, strs: Union[str, List[str]], **kwargs) -> torch.Tensor:
        if isinstance(strs, str):
            return self.conf.env.make(utils.to_tensor(self.tokenizer(strs, truncation=True, padding='max_length', max_length=self.conf.tokenizer_max_length, **kwargs))) # use return_tensors = pt and unsqueeze
        return self.conf.env.make(
                utils.stack_dictionaries(
                        [self.conf.env.make(utils.to_tensor(self.tokenizer(s, truncation=True, padding='max_length', max_length=self.conf.tokenizer_max_length, **kwargs))) for s in strs]))

    def get_augmenter(self, samples: int):
        strength = self.conf.text_shift_stength #rand() #! change me    
        # augmenter1 = naw.AntonymAug(aug_p=strength/2)
        augmenter2 = naw.SynonymAug(aug_p=strength)    
            
        def mask_augment(inp: str, strength: float):
            tok = inp.split()
            if len(tok) == 0:
                return ''
            for i in random.sample(range(len(tok)), int(strength*len(tok))+1):
                tok[i] = TOKENIZER_MASK
            return ' '.join(tok)

        def wrap(inp: str): #! implement n_samples
            x = mask_augment(inp, strength)
            # print(x)
            return x
            return augmenter2.augment(inp, n = samples) # augmenter1.augment(inp)

        return wrap


    def get_noise_samples(self, batch: List[str], conf: Config) -> List[List[str]]:
        """get_noise_samples

            Returns samples from the augmenter (get_augmenter) fir the current batch of data

        Parameters
        ----------
        batch : List[str]
            The batch provided
        conf : Config
            The config
        
        Returns
        -------
        List[str]
            The augmented data from the batch
        """
        if conf.loader.batch_size < 2:
            raise ValueError("Batch size has to be at least 2")
        aug = self.get_augmenter(conf.noise_samples)
        return [aug(a) for a in batch]

    def augment_and_prepare_batch(self, batch, augment=True):
        augmenter = self.get_augmenter(1) if augment else lambda x: x
        new_batch = []
        for elem in batch:
            new_batch.append([ augmenter(self.get_element(elem, self.conf.dataset.input_position)), 
                               self.get_element(elem, self.conf.dataset.label_position)
                            ])

        x, y = [], []
        for elem in new_batch:
            x.append(self.get_element(elem, 0))#self.conf.dataset.input_position))
            y.append(self.get_element(elem, 1))#self.conf.dataset.label_position))

        # print(x) 
        # print("- @ - @ - @ - @ - @ - @ - @ - @ - @ - @ - @ - @ - @ - @ - @ - @ - @  ")
        # print(y)
        # exit()
        return self.tokenize_and_make(x), self.conf.env.make(utils.to_tensor(y)) 

    def forward(self, model: nn.Module, data: Any):
        return model(**data)

    def collate_fn(self, data):
        # self.batch_num += 1
        # x, y = [], []
        # for e in data:
        #     #! dataset specific
        #     x.append(e[1])
        #     y.append(e[0])

        # clear_x = self.tokenize_and_make(self.conf, self.tokenizer, x, padding='max_length', max_length=self.conf.tokenizer_max_length)
        
        # # strength = 0. * (self.batch_num * .5/(self.dataset_len/self.conf.loader.batch_size)) * .3 #! put in config TODO: increase during train
        # noisy_samples: List[List[str]] = self.get_noise_samples(x, self.conf) # batch x samples
        # # noisy_samples = list(filter(lambda e: type(e) == list, noisy_samples))
        # noisy_samples = [self.tokenize_and_make(self.conf, self.tokenizer, [a[i] for a in noisy_samples], padding='max_length', max_length=self.conf.tokenizer_max_length) for i in range(len(noisy_samples[0]))] # List(l=samples)[Tensor[BSxSEQ]]
        # return clear_x, noisy_samples, y
        return data
