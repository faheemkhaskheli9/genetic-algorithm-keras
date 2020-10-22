from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense
import numpy as np

from set_config import GENES

class GeneticModel:

    def __init__(self, genes, layer_names, training_epochs, first_layer, last_layer):
        self.Genes = genes
        self.last_layer = last_layer
        self.layer_names = layer_names
        self.Model = Sequential()
        self.Model.add(first_layer)
        self.create_model()
#        self.Model.summary()
        self.training_epochs = training_epochs


    def create_model(self):
        for layer in range(self.Genes['layers']):
            layer_selected = np.random.choice(GENES['layer_choice'])
            keras_layer = GENES['keras_mapping'][layer_selected]
            if keras_layer == 'conv2d':
                self.Model = self.gene_to_conv(self.Model, self.Genes[layer_selected])
            if keras_layer == 'res_block':
                self.Model = self.gene_to_res_block(self.Model, self.Genes[layer_selected])
        self.Model = self.last_layer(self.Model)


    def gene_to_conv(self, model, gene):
        '''
        convert gene to conv layer
        '''
        filters = gene['filter']
        kernel = (gene['kernel'],gene['kernel'])
        strides = (gene['strides'],gene['strides'])
        activation = gene['activation']
        padding = gene['padding']
        dilation_rate = (gene['dilation_rate'],gene['dilation_rate'])
        model.add(Conv2D(filters=filters,
                         kernel_size=kernel,
                         strides=strides,
                         activation=activation,
                         padding=padding,
                         dilation_rate=dilation_rate,
                      ))
        return model


    def gene_to_res_block(self, model, gene):
        '''
        convert gene to residual block
        '''
        for layer in range(gene['layers']):
            layer_selected = np.random.choice(GENES['layer_choice'])
            layer_selected = GENES['keras_mapping'][layer_selected]
            model = self.gene_to_conv(model, gene[layer_selected])
        return model


    def evaluate_model(self,X_train, y_train, X_test, y_test):
        # train model for epochs
        self.Model.fit(X_train, y_train, epochs=self.training_epochs)
        return self.Model.evaluate(X_test, y_test)[1]
