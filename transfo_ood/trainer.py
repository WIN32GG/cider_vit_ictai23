import logging
import torch
import torchbooster.utils as utils
import torchbooster.distributed as dist
from transfo_ood.cider import Cider

from transfo_ood.config import Config
from transfo_ood.evaluator import ModelEvaluator
from transfo_ood.preparator import DataPreparator
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def fit(conf: Config, model, preparator: DataPreparator, dataset, test_dataset, optim, scheduler, writer: SummaryWriter, evaluator: ModelEvaluator):
    logging.info("Start training")
    model = conf.env.make(model)

    cider_loss = Cider(conf)
    evaluator(steps = 0, epoch_fraction = conf.eval_train_epoch_fraction)
    
    for epoch in range(conf.epochs):
        pbar = tqdm(dataset, disable=not dist.is_primary())
        model = ModelEvaluator.unfreeze(model.train())
        logging.info(f'Epoch {epoch}')
        
        for batch_num, batch in enumerate(pbar):

            step = len(dataset)*epoch + batch_num
            x, y = preparator.augment_and_prepare_batch(batch)
            out  = preparator.forward(model, x) # bs x ProjectorSize

            l_dispersion, l_compactness = cider_loss(out, y)
            loss = conf.lambda_d * l_dispersion + conf.lambda_c * l_compactness

            utils.step(loss, optim, scheduler, clip=conf.clip)
            pbar.set_postfix(l_d=l_dispersion.detach().item(), l_c=l_compactness.detach().item())

            if dist.is_primary():
                writer.add_scalar("loss/L_disper",  l_dispersion.detach().item(), step)
                writer.add_scalar("loss/L_compact", l_compactness.detach().item(), step)
                writer.add_scalar("loss/L_total",   loss.detach().item(), step)
                writer.add_scalar("proto/std", torch.stack(cider_loss.prototypes).std(0).mean().detach().item(), step)

    # print_projector(conf, model, test_dataset, preparator, writer, steps=step)
    evaluator(steps = step, epoch_fraction = conf.eval_train_epoch_fraction)


