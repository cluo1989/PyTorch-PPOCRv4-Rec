__all__ = ['build_head']


def build_head(config):
    from .rec_ctc_head import CTCHead
    from .rec_nrtr_head import Transformer
    from .rec_multi_head import MultiHead    
    
    support_dict = ['CTCHead', 'Transformer', 'MultiHead']
    module_name = config.pop('name')
    assert module_name in support_dict, Exception(
        'Head only support {}'.format(support_dict)
    )
    module_class = eval(module_name)(**config)
    return module_class
