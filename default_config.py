# DEFAULT GENES TYPES
## DONOT CHANGE ANYTHING HERE, IT WILL MAKE THIS PROGRAM STOP WORKING

DEFAULT_GENES = {   'layers': 1,
                    'Conv2D': {
                        'filters': 8,
                        'kernel_size': 3,
                        'strides': 1,
                        'activation': 'relu',
                        'padding': 'same',
                        'dilation_rate': 1,
                    },
                    'Dense': {
                       'units': 8,
                       'activation': 'relu'
                    },
                    'res_block': {
                       'layers': 2,
                    },
                    'BatchNormalization': {
                        'momentum': 0.99
                    },
                    'Activation': {
                        'function': ['relu',
                                    'sigmoid',
                                    'softmax',
                                    'softplus',
                                    'softsign',
                                    'tanh',
                                    'selu',
                                    'elu',
                                    'exponential'],
                    },
                }