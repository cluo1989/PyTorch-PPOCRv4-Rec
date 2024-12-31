import torch
from torch import nn
# from datasets.charset import alphabet
BLANK_INDEX = 0  # len(alphabet)


class CTCLoss(nn.Module):
    def __init__(self, use_focal_loss=False, **kwargs):
        super(CTCLoss, self).__init__()
        self.loss_func = nn.CTCLoss(blank=BLANK_INDEX, reduction='mean')
        self.use_focal_loss = use_focal_loss

    def forward(self, logits, batch):
        logits = logits.permute(1, 0, 2)  # N T C -> T N C
        pred_probs = torch.log_softmax(logits, dim=2)

        batch_size = logits.shape[1]
        time_steps = logits.shape[0]
        pred_lengths = torch.full((batch_size, ), time_steps)

        labels = batch[1]
        label_lengths = batch[2]

        loss = self.loss_func(pred_probs, labels, pred_lengths, label_lengths)

        if self.use_focal_loss:
            weight = torch.square(1 - torch.exp(-loss))
            loss = torch.multiply(loss, weight)

        return loss
    

if __name__ == '__main__':
    loss_func = CTCLoss()

    N = 16                                        # N = batch size
    T = 30                                        # T >= max_len
    C = BLANK_INDEX + 1                           # C = num_classes + 1 (blank)
    min_len = 10
    max_len = 30
    logits = torch.randn(N, T, C)                          # [N, T, C]
    labels = torch.randint(0, C, (N, max_len))             # [0, C, (N, max_len)]
    label_lengths = torch.randint(min_len, max_len, (N, )) # [min_len, max_len, (N, )]
    print(label_lengths.max())

    inputs = [logits, labels, label_lengths]
    loss = loss_func(*inputs)
    print(loss)

    # # 100 = BLANK_INDEX + 1
    # logits = torch.randn(32, 50, 100)
    # labels = torch.randint(0, 100, (32, 10))
    # label_lengths = torch.randint(1, 10, (32, ))
    # inputs = [logits, labels, label_lengths]
    # loss = loss_func(*inputs)
    # print(loss)