import numpy as np
from individual_model import GeneticModel, GENES
from keras.utils import plot_model
import matplotlib.pyplot as plt
from logging_genetic_algorithm import logger
import time


class GeneticAlgorithm:
    def __init__(self, X_train,
                 y_train,
                 X_test,
                 y_test,
                 first_layer,
                 last_layer,
                 generations=10,
                 population=10,
                 mutation_rate=0.1,
                 training_epochs=10,
                 evaluation_metrics='val_accuracy',
                 verbose=2
                 ):


        self.models = []
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.population = population
        self.generations = generations
        self.evaluation_metrics = evaluation_metrics
        self.training_epochs = training_epochs
        self.verbose = verbose
        self.mutation_rate = mutation_rate
        ##
        self.first_layer = first_layer
        self.last_layer = last_layer
        ##
        self.create_population()
        self.best_models = [[0,0] for _ in range(10)]
        #
        self.statistics = []


    def mutate_model(self, model):
        '''
        mutate the model
        :param model: model which will be mutated
        '''
        for gene in model.Genes:
            if (np.random.random()*100) < self.mutation_rate:
                if type(model.Genes[gene]) == int:
                    if type(GENES[gene]) == list:
                        rn = np.random.random()
                        if rn > 0.5:
                            model.Genes[gene] += np.random.choice(GENES[gene])
                        else:
                            model.Genes[gene] -= np.random.choice(GENES[gene])
                if type(model[gene]) == str:
                    if type(GENES[gene]) == list:
                        model.Genes[gene] = np.random.choice(GENES[gene])
        return model


    def create_model_from_initial_genes(self):
        '''
        Create Genes for new model randomly
        :return: single object of GeneticModel Randomly Initialized
        '''
        new_genes = {}
        new_genes['layers'] = np.random.choice(GENES['layers'])
        new_genes['layers_config'] = list()

        for layer in range(new_genes['layers']):
            layer_selected = np.random.choice(GENES['layer_choice'])
            keras_layer = GENES['keras_mapping'][layer_selected]
            if keras_layer == 'res_block':
                new_genes['layers_config'].append(self.resblock_to_gene(GENES[layer_selected]))
            else:
                new_genes['layers_config'].append(self.layer_to_gene(GENES[layer_selected], keras_layer))
        return GeneticModel(new_genes, self.training_epochs, self.first_layer, self.last_layer, self.verbose)


    def layer_to_gene(self, genes, name):
        '''
        add any layer as gene
        :param genes: parameter for batch norma as dictionary
        :param name: name of layer in keras
        :return: model
        '''
        new_gene = {}
        new_gene['name'] = name
        for gene in genes:
            if type(genes[gene]) == list:
                new_gene[gene] = np.random.choice(genes[gene])
            if (type(genes[gene]) == int) or (type(genes[gene]) == float) or (type(genes[gene]) == str):
                new_gene[gene] = genes[gene]
        return new_gene


    def resblock_to_gene(self, genes):
        '''
        add convolution 2d layer as gene
        '''
        new_gene = {}
        new_gene['name'] = 'res_block'
        new_gene['layers_config'] = list()
        for layer_num in range(genes['layers']):
            layer_name = np.random.choice(genes['layer_choice'])
            keras_layer = GENES['keras_mapping'][layer_name]
            if keras_layer == 'Conv2D':
                new_gene['layers_config'].append(self.layer_to_gene(GENES[layer_name], keras_layer))
        return new_gene


    def create_population(self):
        '''
        create population of models
        :return:
        '''
        for i in range(self.population):
            if self.verbose == 2:
                print('model id', i)
            self.models.append(self.create_model_from_initial_genes())


    def breeding(self, model1, model2):
        new_genes = {}

        if model1.Genes['layers'] >= model2.Genes['layers']:
            special_genes = model1.Genes
            other_genes = model2.Genes
            special_layers = special_genes['layers'] + 1
        elif model2.Genes['layers'] > model1.Genes['layers']:
            special_genes = model2.Genes
            other_genes = model1.Genes
            special_layers = special_genes['layers'] + 1
        new_genes['layers'] = np.random.randint(other_genes['layers'], special_layers)
        new_genes['layers_config'] = []
        for l in range(new_genes['layers']):
            if l < len(other_genes['layers_config']):
                rn = np.random.random()
                if rn > 0.5:
                    new_genes['layers_config'].append(special_genes['layers_config'][l])
                else:
                    new_genes['layers_config'].append(other_genes['layers_config'][l])
            else:
                new_genes['layers_config'].append(special_genes['layers_config'][l])
        model = self.mutate_model(GeneticModel(new_genes, self.training_epochs, self.first_layer, self.last_layer))
        return model


    def create_new_best_generation(self):
        indx = 0
        for m1 in self.best_models:
            if indx == (self.population - 1):
                break
            if (m1[1] == 0) or (m1[0] == 0):
                self.models[indx] = self.create_model_from_initial_genes()
                indx += 1
            else:
                for m2 in self.best_models:
                    if indx == (self.population - 1):
                        break
                    if m2[0] == 0:
                        continue
                    if m1 != m2:
                        self.models[indx] = self.breeding(m1[0], m2[0])
                        indx += 1


    def survival_of_fittest(self):
        '''
        evaluate models and select best
        :return: None
        '''
        model_results = []
        model_num = 0

        if self.verbose == 2:
            print(GENES)

        all_accuracies = []
        all_train_times = []

        for model in self.models:
            start_time = time.time()
            logger.debug('Model Number :' + str(model_num))
            if self.verbose >= 1:
                print('Training Model ', model_num)
            if self.verbose >= 2:
                print(model.Genes)
            if self.verbose >= 3:
                model.Model.summary()
            if self.verbose >= 5:
                plot_model(model.Model, to_file="./images/model_"+str(model_num)+".png")
                plt.show()

            if True:
                results = model.evaluate_model(self.X_train,
                                               self.y_train,
                                               self.X_test,
                                               self.y_test,)
                model_results.append([model_num,
                                      model,
                                      results])
            else:
                model_results.append([model_num,
                                      model,
                                      model_num])
            model_num += 1
            all_accuracies.append(results)
            all_train_times.append(time.time() - start_time)
            del model.Model
        self.statistics[-1]['accuracy'] = all_accuracies
        self.statistics[-1]['train_time'] = all_train_times

        if self.verbose >= 1:
            print('Model Fitness', model_results)

        mindx = lambda x: x[0]
        model_results = sorted(model_results, key=mindx, reverse=True)[:3]
        self.best_models = sorted(self.best_models, key=lambda x: x[1], reverse=True)
        for m in model_results[:3]:
            print(self.best_models[-1],self.best_models[-1][1], m[2])
            if self.best_models[-1][1] < m[2]:
                self.best_models[-1] = [m[1], m[2]]
                self.best_models = sorted(self.best_models, key=lambda x: x[1], reverse=True)
        self.write_to_file()
        if self.verbose >= 1:
            for m in self.best_models:
                print('Best Accuracy', m[1])


    def write_to_file(self):
        with open('./results/best_models.txt', 'w+') as file:
            for model in self.best_models:
                if type(model[0]) == int:
                    conf = ''
                else:
                    conf = model[0].Genes
                file.writelines('Configuration :' + str(conf))
                file.writelines('\n')
                file.writelines('Accuracy :' + str(model[1]))
                file.writelines('\n')


    def evolve(self):
        for g in range(self.generations):
            self.statistics.append({'accuracy': [],
                                    'train_time': []})
            logger.debug('Generation Number :' + str(g))
            if self.verbose >= 1:
                print('\nGeneration ', g)
            self.survival_of_fittest()
            self.create_new_best_generation()

            plt.bar(list(range(10)), self.statistics[-1]['accuracy'])
            plt.savefig('./images/Generation '+str(g)+".png")
            #plt.show()
