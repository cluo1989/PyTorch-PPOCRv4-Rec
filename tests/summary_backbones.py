import os, sys
# enable ppocr module can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


if __name__ == '__main__':
    import torch
    from torchsummary import summary

    from ppocr.modeling.backbones.rec_lcnetv3 import PPLCNetV3
    from ppocr.modeling.backbones.rec_svtrnet import SVTRNet

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    backbone1 = PPLCNetV3().to(device)
    # summary(backbone1, input_size=(3, 32, 320), batch_size=1, device='cpu')
    summary(backbone1, input_size=(3, 32, 320), batch_size=1, device='cuda')

    backbone2 = SVTRNet(img_size=[32, 320]).to(device)
    # summary(backbone2, input_size=(3, 32, 320), batch_size=1, device='cpu')
    summary(backbone2, input_size=(3, 32, 320), batch_size=1, device='cuda')
