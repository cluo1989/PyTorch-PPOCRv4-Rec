from torch import nn


class SARLoss(nn.Module):
    def __init__(self):
        super(SARLoss, self).__init__()
        self.loss_func = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, logits, labels):
        logits = logits[:-1, :, :]  # ignore last index of outputs to be in same seq_len with targets
        labels = labels[:, 1:]      # ignore first index of targets in loss calculation
        
        num_classes = logits.shape[2]  # N, T, C
        logits = logits.reshape([-1, num_classes])
        labels = labels.reshape([-1])
        loss = self.loss_func(logits, labels)

        return loss
    
if __name__ == '__main__':
    import torch

    loss_func = SARLoss()
    logits = torch.randn(31, 16, 20)  # (T, N, C)
    labels = torch.randint(1, 20, (16, 31), dtype=torch.long)  # (N, S)
    print(loss_func(logits, labels))
