# DEFAULT GENES TYPES
## DONOT CHANGE ANYTHING HERE, IT WILL MAKE THIS PROGRAM STOP WORKING

DEFAULT_GENES = {   'layers': 1,
                    'conv2d': {
                        'filter': 8,
                        'kernel': 3,
                        'strides': 1,
                        'activation': 'relu',
                        'padding': 'same',
                        'dilation_rate': 1,
                    },
                    'dense': {
                       'unit': 8,
                       'activation': 'relu'
                    },
                    'res_block': {
                       'layers': 2,
                       'layer_type': ['conv2d'],
                       'conv2d': {
                           'filter': 8,
                           'kernel': 3,
                           'activation': 'relu',
                           'padding': 'same',
                           'dilation_rate': 1,
                       }
                    }
                }