import numpy as np
from individual_model import GeneticModel, GENES


class GeneticAlgorithm:

    def __init__(self, X_train,
                 y_train,
                 X_test,
                 y_test,
                 population,
                 first_layer,
                 last_layer,
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


    def mutate_model(self, model):
        '''
        mutate the model
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
        new_genes = {}
        new_genes['layers'] = np.random.choice(GENES['layers'])
        layer_names = []
        for layer in range(new_genes['layers']):
            layer_selected = np.random.choice(GENES['layer_choice'])
            keras_layer = GENES['keras_mapping'][layer_selected]
            if keras_layer == 'conv2d':
                new_genes[layer_selected] = self.conv_to_gene(GENES[layer_selected])
                layer_names.append(layer_selected)
        return GeneticModel(new_genes, layer_names, self.training_epochs, self.first_layer, self.last_layer)


    def create_model_from_genes(self, genes, layer_names):
        new_genes = {}
        new_genes['layers'] = np.random.choice(GENES['layers'])
        new_layer_names = []
        for layer in range(new_genes['layers']):
            layer_selected = np.random.choice(GENES['layer_choice'])
            keras_layer = GENES['keras_mapping'][layer_selected]
            if keras_layer == 'conv2d':
                new_genes[layer_names[layer]] = self.conv_to_gene(genes[layer_selected])
                new_layer_names.append(layer_names[layer])
        return GeneticModel(new_genes, layer_names, self.training_epochs, self.first_layer, self.last_layer)


    def conv_to_gene(self, genes):
        '''
        add convolution 2d layer as gene
        '''
        new_gene = {}
        for gene in genes:
            if type(genes[gene]) == list:
                new_gene[gene] = np.random.choice(genes[gene])
            elif (type(genes[gene])) == int or (type(genes[gene]) == str):
                new_gene[gene] = genes[gene]
        return new_gene


    def create_population(self):
        '''
        create population of models
        :return:
        '''
        for i in range(self.population):
            self.models.append(self.create_model_from_initial_genes())


    def breeding(self, model1, model2):
        new_genes = {}
        new_layer_names = []

        if model1.Genes['layers'] >= model2.Genes['layers']:
            special_genes = model1.Genes
            other_genes = model2.Genes
            special_layers = special_genes['layers'] + 1
            special_model_layers = model1.layer_names
            other_model_layers = model2.layer_names
        elif model2.Genes['layers'] > model1.Genes['layers']:
            special_genes = model2.Genes
            other_genes = model1.Genes
            special_layers = special_genes['layers'] + 1
            special_model_layers = model2.layer_names
            other_model_layers = model1.layer_names

        new_genes['layers'] = np.random.randint(other_genes['layers'], special_layers)
        for l in range(new_genes['layers']):
            if l < len(other_model_layers):
                rn = np.random.random()
                if rn > 0.5:
                    new_genes[special_model_layers[l]] = special_genes[special_model_layers[l]]
                    new_layer_names.append(special_model_layers[l])
                else:
                    new_genes[other_model_layers[l]] = other_genes[other_model_layers[l]]
                    new_layer_names.append(special_model_layers[l])
            else:
                new_genes[special_model_layers[l]] = special_genes[special_model_layers[l]]
                new_layer_names.append(special_model_layers[l])

        model = self.mutate_model(GeneticModel(new_genes, new_layer_names, self.training_epochs, self.first_layer, self.last_layer))

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
        :return:
        '''

        model_results = []
        model_num = 0
        for model in self.models:
            print('Training Model ', model_num)
            if True:
                model_results.append([model_num,
                                      model,
                                      model.evaluate_model(self.X_train,
                                   self.y_train,
                                   self.X_test,
                                   self.y_test,)])
            else:
                model_results.append([model_num,
                                      model,
                                      model_num])
            model_num += 1
        print('Model Fittness', model_results)

        mindx = lambda x: x[0]
        model_results = sorted(model_results, key=mindx, reverse=True)[:3]
        for m in model_results[:3]:
            if self.best_models[-1][1] < m[2]:
                self.best_models[-1] = [m[1], m[2]]
                self.best_models = sorted(self.best_models, key=lambda x: x[1], reverse=True)
        print('Best ', self.best_models)


    def evolve(self):
        for g in range(10):
            print('')
            print('Generation ', g)
            self.survival_of_fittest()
            self.create_new_best_generation()
