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
from ppocr.losses.rec_ctc_loss import CTCLoss
from ppocr.optimizers.optimizer import Adam
from ppocr.datasets.rec_dataset import RecDataset
from ppocr.metrics.rec_metric import RecMetric

accuracy = RecMetric()
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29671'


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

def main(config, local_rank=0):
    # set seeds & cudnn
    SEED = config['Global']['SEED']
    BENCHMARK = config['CUDNN']['BENCHMARK']
    DETERMINISTIC = config['CUDNN']['DETERMINISTIC']
    ENABLED = config['CUDNN']['ENABLED']

    if SEED is not None:
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)

    cudnn.benchmark = BENCHMARK
    cudnn.deterministic = DETERMINISTIC
    cudnn.enabled = ENABLED

    # device
    if torch.cuda.is_available():
        ngpus = 1
        #dist.init_process_group(backend=dist.Backend.NCCL, world_size=ngpus, rank=local_rank) 
        #torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        #dist.init_process_group(backend=dist.Backend.GLOO, world_size=1, rank=local_rank)
        device = torch.device("cpu")

    # build model
    from ppocr.datasets.charset import alphabet
    char_num = len(alphabet)
    if config['Architecture']['Head']['name'] == 'MultiHead':
        if config['PostProcess']['name'] == 'SARLabelDecode':
            char_num = char_num - 2
        if config['PostProcess']['name'] == 'NRTRLabelDecode':
            char_num = char_num - 3
        
        out_channels_list = {}
        out_channels_list['CTCLabelDecode'] = char_num

        if list(config['Loss']['loss_config_list'][1].keys())[0] == 'SARLoss':
            if config['Loss']['loss_config_list'][1]['SARLoss'] is None:
                config['Loss']['loss_config_list'][1]['SARLoss'] = {'ignore_index': char_num + 1}
            else:
                config['Loss']['loss_config_list'][1]['SARLoss']['ignore_index'] = char_num + 1

            out_channels_list['SARLabelDecode'] = char_num + 2
        elif list(config['Loss']['loss_config_list'][1].keys())[0] == 'NRTRLoss':
            out_channels_list['NRTRLabelDecode'] = char_num + 3

        config['Architecture']['Head']['out_channels_list'] = out_channels_list

    else:
        config['Architecture']['Head']['out_channels'] = char_num

    if config['PostProcess']['name'] == 'SARLabelDecode':
        config['Loss']['ignore_index'] = char_num - 1

    model = BaseModel(config['Architecture'])
    model.to(device)
    criterion = CTCLoss().to(device)
    optimizer = Adam(config['Optimizer'])(model)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config['Global']['lr_step'],
        gamma=config['Global']['lr_factor'],
        last_epoch=config['Global']['start_eopch'] - 1
    )

    # resume
    best_acc = 0.0
    resume_file = config['RESUME']
    if resume_file:
        if os.path.isfile(resume_file):
            print(f"Loading checkpoint '{resume_file}'")
            if torch.cuda.is_available():
                checkpoint = torch.load(resume_file, map_location=device)
            else:
                checkpoint = torch.load(resume_file)

            # parse & load from checkpoint
            config['Global']['start_eopch'] = checkpoint['epoch']
            best_acc = checkpoint['best_acc']

            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler']) 
            print("=> loaded checkpoint '{}' (epoch {})".format(resume_file, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume_file))

    # model = DistributedDataParallel(model)

    # build dataset
    trainset = RecDataset(config['Train']['dataset']['label_file'], config['Train']['dataset']['image_dir'])
    # trainsampler = DistributedSampler(trainset)
    train_ld_conf = config['Train']['loader']
    trainloader = DataLoader(
        dataset=trainset,
        batch_size=train_ld_conf['batch_size_per_card'],
        num_workers=train_ld_conf['num_workers'],
        pin_memory=train_ld_conf['pin_memory'],
        drop_last=train_ld_conf['drop_last'],
        shuffle=train_ld_conf['shuffle'],
        # sampler=trainsampler
    )

    valset = RecDataset(config['Eval']['dataset']['label_file'], config['Eval']['dataset']['image_dir'])
    # valsampler = DistributedSampler(valset)
    val_ld_conf = config['Eval']['loader']
    valloader = DataLoader(
        dataset=valset,
        batch_size=val_ld_conf['batch_size_per_card'],
        num_workers=val_ld_conf['num_workers'],
        pin_memory=val_ld_conf['pin_memory'],
        drop_last=val_ld_conf['drop_last'],
        shuffle=val_ld_conf['shuffle'],
        # sampler=valsampler
    )

    # training loop
    for epoch in range(config['Global']['start_eopch'], config['Global']['epoch_num']):
        # trainsampler.set_epoch(epoch)
        train(trainloader, model, criterion, optimizer, scheduler, epoch, device, config)
        acc = validate(valloader, model, criterion, device, config)
        scheduler.step()

        is_best = acc > best_acc        
        if is_best:
            best_acc = max(acc, best_acc)
            save_file = os.path.join(config['Global']['save_model_dir'], f'model_best_{epoch}_{round(best_acc, 4)}.pth')
            torch.save({
                'epoch': epoch + 1,
                'best_acc': best_acc,
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, save_file)

    print(config['Global'])


if __name__ == "__main__":
    # parse configuration file 
    # merge optional args
    FLAGS = ArgsParser().parse_args()
    config = load_config(FLAGS.config)
    config = merge_config(config, FLAGS.opt)
    profiler_dic = {"profiler_options": FLAGS.profiler_options}
    config = merge_config(config, profiler_dic)
    main(config)