# Genetic Algorithm For Keras Model

This is in initial stage, made using Conv2D layer, Batch 
Normalization, Activation Layers of keras and residual block.
it can random select keras Conv2D layers to create initial genration 
and train it for any number of epochs that you specified.

# Architecture
This Genetic Algorithm will generate random Architecture using following
layers of keras. you can also specify specify value of each parameters.
There are some default value pre-defined which will be used if you 
skip any parameter.

If you want to **generate random** values then specify those values in **list**. 

## Layers Supported for Architecture

- Conv2D
- Dense
- Activation
- Batch Normalization

## Other Architecture

- Residual Block
  - Combination of any number of Conv2D layers 

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

## Keras Mapping
In config the possible values for keras_mapping key are of 2 different
types, 1st one are directly from keras, and 2nd are block of multiple
layers such as residual block.

Following table show the name for keras layers.

|Name|Keras Layer|
|---|---|
|Conv2D|Conv2D|
|BatchNormalization|BatchNormalization|
|Activation|Activation|

Following table show the name for parameters for keras layers.

|Layer|Parameter Name|My Name|
|---|---|---|
|Conv2D|filters|filters|
| |kernel_size|kernel_size|
| |strides|strides|
| |padding|padding|
| |activation|activation|
| |dilation_rate|dilation_rate|
|BatchNormalization|momentum|momentum|
|Activation|name of activation|function|

|Layers Block|Description|
|---|---|
|res_block|Residual with n number of inner layer, you can use any 
| |layers as inner layer. Final layer is always concatenation
| |of initial and last layer before concatenation.|
## Verbose Level
|Verbose Level|Detail to Show|
|---|---|
|0|No output|
|1|Generation Number|
| |Best Selected Model Per Generation|
|2|Model Number|
| |Model Genes/Parameters|
| |Original Gene|
|3|Model Summary|
|4| |
|5|Plot Keras Model|