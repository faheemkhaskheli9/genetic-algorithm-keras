from myconfig import my_genes
from default_config import DEFAULT_GENES

GENES = {}

if 'layers' in my_genes:
    GENES['layers'] = my_genes['layers']
if 'layer_choice' in my_genes:
    GENES['layer_choice'] = my_genes['layer_choice']
if 'keras_mapping' in my_genes:
    GENES['keras_mapping'] = my_genes['keras_mapping']

    for layer in my_genes['keras_mapping']:
        keras_layer = my_genes['keras_mapping'][layer]
        GENES[layer] = DEFAULT_GENES[keras_layer]
        for parameter in my_genes[layer]:
            GENES[layer][parameter] = my_genes[layer][parameter]
print(GENES)