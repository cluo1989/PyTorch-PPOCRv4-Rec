__all__ = ['build_transform']

def build_transform(config):
    from .tps import TPS

    support_dict = ['TPS']
    module_name = config.pop('name')
    assert module_name in support_dict, Exception(
        'Transform only support {}'.format(support_dict)
    )
    # build class
    module_class = eval(module_name)(**config)
    return module_class