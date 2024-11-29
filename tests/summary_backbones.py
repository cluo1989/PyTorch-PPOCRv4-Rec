import os, sys
# enable ppocr module can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


if __name__ == '__main__':
    from ppocr.modeling.backbones.rec_lcnetv3 import PPLCNetV3
    from ppocr.modeling.backbones.rec_svtrnet import SVTRNet
    from torchsummary import summary
    
    backbone1 = PPLCNetV3()
    summary(backbone1, input_size=(3, 32, 320), batch_size=1)

    backbone2 = SVTRNet(img_size=[32, 320])
    summary(backbone2, input_size=(3, 32, 320), batch_size=1)
