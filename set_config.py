from myconfig import my_genes
from default_config import DEFAULT_GENES

GENES = {}

if 'layers' in my_genes:
    GENES['layers'] = my_genes['layers'].copy()

if 'layer_choice' in my_genes:
    GENES['layer_choice'] = my_genes['layer_choice'].copy()

if 'keras_mapping' in my_genes:
    GENES['keras_mapping'] = my_genes['keras_mapping'].copy()

    for layer in my_genes['keras_mapping']:
        keras_layer = my_genes['keras_mapping'][layer]
        if keras_layer == 'res_block':
            GENES[layer] = {}
            GENES[layer]['layers'] = my_genes[layer]['layers']
            GENES[layer]['layer_choice'] = my_genes[layer]['layer_choice'].copy()
            for res_layer in my_genes[layer]['layer_choice']:
                res_keras_layer = my_genes['keras_mapping'][res_layer]
                GENES[res_layer] = DEFAULT_GENES[res_keras_layer].copy()
                for parameter in my_genes[res_layer]:
                    GENES[res_layer][parameter] = my_genes[res_layer][parameter].copy()
        else:
            GENES[layer] = DEFAULT_GENES[keras_layer]
            if layer in my_genes:
                for parameter in my_genes[layer]:
                    if type(my_genes[layer][parameter]) == list:
                        GENES[layer][parameter] = my_genes[layer][parameter].copy()
                    else:
                        GENES[layer][parameter] = my_genes[layer][parameter]
