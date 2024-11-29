from torch import nn
from rec_ctc_loss import CTCLoss
from rec_sar_loss import SARLoss


class MultiLoss(nn.Module):
    def __init__(self, **kwargs):
        super(MultiLoss, self).__init__()
        self.loss_funcs = {}
        self.loss_list = kwargs.pop('loss_config_list')
        self.weight1 = kwargs.pop('weight1', 1.0)
        self.weight2 = kwargs.pop('weight2', 1.0)
        self.gtc_loss = kwargs.get('gtc_loss', 'sar')
        for loss_info in self.loss_list:
            for name, param in loss_info.items():
                if param is not None:
                    kwargs.update(param)
                loss = eval(name)(**kwargs)
                self.loss_funcs[name] = loss

    def forward(self, predicts, batch):
        self.total_loss = {}
        total_loss = 0.0

        for name, loss_func in self.loss_funcs.items():
            if name == 'CTCLoss':
                loss = loss_func(predicts['ctc'], batch[:2] + batch[3:]) * self.weight1
            elif name == 'SARLoss':
                loss = loss_func(predicts['sar'], batch[:1] + batch[2:]) * self.weight2
            else:
                raise NotImplementedError(f'{name} is not supported in MultiLoss yet.')
            
            self.total_loss[name] = loss
            total_loss += loss

        self.total_loss['loss'] = total_loss
        return self.total_loss
    