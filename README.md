# Genetic Algorithm For Keras Model

This is in initial stage, only made using Conv2D layer of kerass.
it can random select keras Conv2D layers to create initial genration 
and train it for any number of epochs that you specified.

# how to use?
Download these files, in ````main.py```` you can load your own dataset,
you can specify your own first and last layer since they are different 
for each proejct. this model can do classification or regression based 
on final layer and its output activation function. which you can specify 
in ````main.py````.

You can also use ````myconfig.py```` file to specify the behaviour of 
random selector for each individual in generation. look at the 
````myconf.py```` file or ````default_config.py```` file for samples
but donot modify the ````default_config.py```` file. it is required for
initalizing default values of Conv2D layer.

### myconfig.py
in ````myconf.py```` you can create your own random configuration by 
adding a key value pair.

**Example**: 
````python
# 1st key layers will specify random layers
# model will have 1 to 4 layers randomly
my_genes = {'layers':list(range(1, 4))}
# default value of layers is 1 as specified in default_config.py
````

````python
my_genes = {'layers':list(range(1, 4)),
        'layer_choice':['mychoiceofrandomlayer'],
        'mychoiceofrandomlayer':{'filter':[2**n for n in range(3, 4)]},
        'keras_mapping':{'mychoiceofrandomlayer': 'conv2d'}
}
# 2nd key layer_choice will create your new configuration.
# 3rd key mychoiceofrandomlayer will specify which parameter of keras layer 
# will be randomly selected
# 4th key keras_mapping will be used to select the layer for which
# your configuration is for. example conv2d
````

#### possible values for config
-  **layers**: number of layer

-  **layer_choice**: which config to randomly select, if you create a new config but donot
 add it in layer_choice then it will not be randomly selected

- **mychoiceofrandomlayer**: parameter for this configuration to randomly select

- **conv2d**: following are con2d keras layer parameter that can be randomly selected
and specified in myconfig.py
   - **filter**: (int) or (list of int for random selection)
   - **kernel**: (int) or (list of int for random selection)
   - **strides**: (int) or (list of int for random selection)
   - **activation**: (str) or (list of str for random selection)
   - **padding**: (str) or (list of str for random selection)
   - **dilation_rate**: (int) or (list of int for random selection)

- **keras_mapping**: key value pair for config to map with keras layer.