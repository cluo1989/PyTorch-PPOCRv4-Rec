from torch import optim


'''class Momentum(object):
    """
    Simple Momentum Optimizer with velocity state.
    Args:
        learning_rate (float) - The learning rate used to update parameters.
            Can be a float value or a Variable with one float value as data element.
        momentum (float) - Momentum factor.
        regularization (WeightDecayRegularizer, optional) - The strategy of regularization.
    """
    def __init__(
            self, learning_rate, momentum, weight_decay=None, grad_clip=None, **args
    ):
        super(Momentum, self).__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip

    def __call__(self, model):
        train_parameters = filter(lambda p: p.requires_grad, model.parameters())
        opt = optim.Momentum(
            train_parameters,
        )
        return opt'''

class Adam(object):
    def __init__(
            self, 
            config
    ):
        self.learning_rate = config['lr']['learning_rate']
        self.beta1 = config['beta1']
        self.beta2 = config['beta2']
        self.epsilon = config['epsilon']
        self.weight_decay = config['regularizer']['factor']

    def __call__(self, model):
        train_parameters = filter(lambda p: p.requires_grad, model.parameters())
        return optim.Adam(
            train_parameters,
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2),
            eps=self.epsilon,
            weight_decay=self.weight_decay
        )