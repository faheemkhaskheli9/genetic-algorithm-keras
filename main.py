from genetic_algorithm import GeneticAlgorithm, GeneticModel
from keras.datasets import mnist
import numpy as np
from keras.layers import Conv2D, Flatten, Dense

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = np.reshape(X_train, X_train.shape + (1,))
X_test = np.reshape(X_test, X_test.shape + (1,))
y_train = np.reshape(y_train, y_train.shape + (1,))
y_test = np.reshape(y_test, y_test.shape + (1,))

print(X_train.shape)
print(y_train.shape)


def last_layer(model):
    '''
    This will add last layer to models for Genetic Algorithm.
    Also Compile the models
    '''
    model.add(Flatten())
    model.add(Dense(1, activation='relu'))
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    return model

first_layer = Conv2D(8, (3,3), padding='same', activation='relu', input_shape=X_test.shape[1:])

g1 = GeneticAlgorithm(X_train, y_train, X_test, y_test, 10, first_layer, last_layer, mutation_rate=0.1, training_epochs=1, verbose=2)
g1.evolve()
