import torch
from torch import nn
import torch.nn.functional as F


class NRTRLoss(nn.Module):
    def __init__(self, smoothing=True, ignore_index=0, **kwargs):
        super(NRTRLoss, self).__init__()
        if ignore_index >= 0 and not smoothing:
            self.loss_func = nn.CrossEntropyLoss(
                reduction='mean', ignore_index=ignore_index
                )
        self.smoothing = smoothing

    def forward(self, logits, labels):
        max_len = labels[2].max()
        tgt = labels[1][:, 1 : max_len + 2]
        tgt = tgt.reshape(-1)
        logits = logits.reshape(-1, logits.shape[2])        

        if self.smoothing:
            eps = 0.1
            n_class = logits.shape[1]
            one_hot = F.one_hot(tgt, n_class)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(logits, dim=1)
            non_pad_mask = torch.not_equal(tgt, torch.zeros(tgt.shape, dtype=tgt.dtype))
            loss = -(one_hot * log_prb).sum(dim=1)
            loss = loss.masked_select(non_pad_mask).mean()
        else:
            loss = self.loss_func(logits, tgt)
        return loss
