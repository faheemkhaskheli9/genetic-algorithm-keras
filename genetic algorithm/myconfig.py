my_genes = {'layers': list(range(1, 4)),
              'layer_choice': ['conv2d1'],
              'conv2d1': {'filter': [2**n for n in range(3, 4)],
                          'activation': ['relu', 'tanh', 'sigmoid']},
              'conv2d2': {'filter': [2**n for n in range(3, 8)]},
              'keras_mapping': {'conv2d1': 'conv2d',
                                'conv2d2': 'conv2d',
                                }
            }
