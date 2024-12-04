import time
import os, sys
# enable ppocr module can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from argsparser import ArgsParser, load_config, merge_config
from utils import AverageMeter, ProgressMeter, Summary
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

def validate():
    pass
def main(config):
    # parse configs
    # build dataloader
    # build model
    # training loop
    print(config)


if __name__ == "__main__":
    FLAGS = ArgsParser().parse_args()
    config = load_config(FLAGS.config)
    config = merge_config(config, FLAGS.opt)
    profiler_dic = {"profiler_options": FLAGS.profiler_options}
    config = merge_config(config, profiler_dic)
    main(config)