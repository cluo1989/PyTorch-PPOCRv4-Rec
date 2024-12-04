import time
import os, sys
# enable ppocr module can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import random
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

from argsparser import ArgsParser, load_config, merge_config
from utils import AverageMeter, ProgressMeter, Summary
from ppocr.modeling.architectures.base_model import BaseModel
from ppocr.datasets.rec_dataset import RecDataset
from ppocr.metrics.rec_metric import RecMetric

accuracy = RecMetric()
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def train(train_loader, model, criterion, optimizer, scheduler, epoch, device, config):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    accs = AverageMeter('Acc', ':6.2f')
    
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, accs],
        prefix="Epoch: [{}]".format(epoch)
        )
    
    # train mode
    model.train()
    num_batches = len(train_loader)

    end = time.time()
    for i, (images, labels, label_lengths) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # move data to the same device as model
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        label_lengths = label_lengths.to(device, non_blocking=True)

        # compute output
        outputs = model(images)
        loss  = criterion(outputs, labels, label_lengths)

        # measure acc and record loss
        acc = accuracy(outputs, labels)['acc']
        losses.update(loss.item(), images.size(0))
        accs.update(acc, images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            progress.display(i + 1)

        # save checkpoint
        step = epoch * num_batches + i + 1
        if step % config.SAVE_FREQ == 0:
            save_file = os.path.join(
                config.OUTPUT_DIR, 
                ''.join([
                    f'checkpoint_{epoch}_{i}',
                    f'_{round(loss.item(), 4)}',
                    f'_{round(acc, 4)}.pth'
                ])
                )

            torch.save({
                'epoch': epoch,
                'step': step,
                'best_acc':acc,
                'loss': loss,
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, save_file)

def validate(val_loader, model, criterion, device, config):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    accs = AverageMeter('Acc', ':6.2f', Summary.AVERAGE)

    progress = ProgressMeter(
        len(val_loader), 
        [batch_time, losses, accs],
        prefix='Test: ')
    
    model.eval()

    def run_validate(loader, base_progress=0):
        with torch.no_grad():
            end = time.time()
            for i, (images, labels, label_lengths) in enumerate(loader):
                i = base_progress + i
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                label_lengths = label_lengths.to(device, non_blocking=True)

                # compute output
                output = model(images)
                loss = criterion(output, labels, label_lengths)

                # measure acc and record loss
                acc = accuracy(output, labels)['acc']
                losses.update(loss.item(), images.size(0))
                accs.update(acc, images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % config.PRINT_FREQ == 0:
                    progress.display(i + 1)

    run_validate(val_loader)
    progress.display_summary()
    return accs.avg

def main(config):
    # set seeds
    SEED = config['Global'].get('seed', 1024)

    if SEED is not None:
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)

        cudnn.benchmark = config.CUDNN
    # build dataloader
    # build model
    # training loop
    print(config['Global'])


if __name__ == "__main__":
    FLAGS = ArgsParser().parse_args()
    config = load_config(FLAGS.config)
    config = merge_config(config, FLAGS.opt)
    profiler_dic = {"profiler_options": FLAGS.profiler_options}
    config = merge_config(config, profiler_dic)
    main(config)