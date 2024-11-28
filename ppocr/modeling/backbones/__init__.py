__all__ = ['build_backbone']


def build_backbone(config, model_type):
    if model_type == 'rec' or model_type == 'cls':
        from .rec_lcnetv3 import PPLCNetV3
        from .rec_svtrnet import SVTRNet
        support_dict = ['PPLCNetV3', 'SVTRNet']
    else:
        raise NotImplementedError
    
    module_name = config.pop('name')
    assert module_name in support_dict, Exception(
        'when model type is {}, backbone only support {}'.format(model_type, support_dict)
    )
    module_class = eval(module_name)(**config)
    return module_class