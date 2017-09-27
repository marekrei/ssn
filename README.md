Supervised Similarity Network
=============================

SSN is a neural network project for classifying word pairs. It is described in the following paper on metaphor detection:

[**Grasping the Finer Point: A Supervised Similarity Network for Metaphor Detection**](http://aclweb.org/anthology/D/D17/D17-1162.pdf)  
Marek Rei, Luana Bulat, Douwe Kiela and Katia Shutova  
*In Proceedings of EMNLP-2017*

Requirements
------------

* python (tested with 2.7.12)
* numpy (tested with 1.12.0)
* theano (tested with 0.8.2)

Data format
-----------

The data should be in a 3-column format. The first column contains the binary label (0 or 1), and the other columns contain the words. For example, here is a slice of the metaphor detection data:

    0   fierce   pirate
    0   bright   sun
    1   raw      idea
    1   healthy  economy


Running
-------

You can train a model with

    python experiment.py config.conf

There is also an example script for using saved models. Run it with:

    python example.py model_path/model_name.model word1:word2 word3:word4 ...

For example:

    python example.py model_ssn_traintsvgut_vecskipatt.model fierce:pirate bright:sun raw:idea healthy:economy

    fierce	pirate	0	0.287809903933
    bright	sun	0	0.137548316842
    raw	idea	1	0.733322964664
    healthy	economy	1	0.691645595258

The last two columns show predictions from the metaphor detection model -- label and score. Label 0 indicates literal phrases, label 1 shows metaphorical phrases.


Metaphor models
---------------

We provide 3 pre-trained models, based on the paper [**Grasping the Finer Point: A Supervised Similarity Network for Metaphor Detection**](http://aclweb.org/anthology/D/D17/D17-1162.pdf). In the paper, we reported the average scores over 25 runs with different random seeds. In order to approximate this, we provide here individual models that had the performance closest to that reported average.

1. [**model_ssn_traintsv_vecskipatt.model**](https://s3-eu-west-1.amazonaws.com/ssnmodels/model_ssn_traintsv_vecskipatt.model)  
The SSN fusion model trained on the Tsvetkov dataset. Receives F1=81.11 on the Tsvetkov test set.

2. [**model_ssn_traintsvgut_vecskipatt.model**](https://s3-eu-west-1.amazonaws.com/ssnmodels/model_ssn_traintsvgut_vecskip.model)  
The SSN fusion model trained on the Tsvetkov+Gutierrez dataset. Receives F1=88.30 on the Tsvetkov test set.

3. [**model_ssn_traintsvgut_vecskip.model**](https://s3-eu-west-1.amazonaws.com/ssnmodels/model_ssn_traintsvgut_vecskip.model)  
The SSN skip-gram model trained on the Tsvetkov+Gutierrez dataset. While the previous models only contain vocabulary present in the metaphor datasets, this one stores all the skip-gram embeddings in the model (vocaulary size 184,816). Therefore, while this doesn't receive the highest F-score, it should be the best model to use for downstream applications. Receives F1=86.73 on the Tsvetkov test set.



Configuration
-------------

The file *config.conf* contains an example configuration. *config_load.conf* contains an example of testing a pre-trained model.

There are a number of values that can be changed:

* **path_train** - Path to the training data.
* **path_dev** - Path to the development data, used for choosing the best epoch.
* **path_test** - Path to the test data.
* **word_embedding_path_a** - Path to the pretrained word embeddings, in word2vec plain text format. If your embeddings are in binary, you can use [convertvec](https://github.com/marekrei/convertvec) to convert them to plain text.
* **word_embedding_size_a** - Size of the word embeddings used in the model.
* **word_embedding_path_b** - Path to the second set of embeddings, if using model fusion.
* **word_embedding_size_b** - Size of the second set of embeddings.
* **stop_if_no_improvement_for_epochs** - Training will be stopped if there has been no improvement for n epochs.
* **examples_per_batch** - Batch size.
* **shuffle_training_data** - Shuffle the training data after each epoch.
* **gating** - Gating the second word based on the first word.
* **embedding_mapping_size** - Size on the supervised word mappings.
* **embedding_combination** - Method for combining the word representations.
* **late_fusion** - Using late fusion.
* **cost** - Cost objective type.
* **update_embeddings** - Whether word embeddings should be updated. 
* **extend_vocabulary** - Extend the model vocabulary when preloading the embeddings.
* **cost_l2** - L2 penalty.
* **optimisation_strategy** - Optimisation strategy. Default is adadelta.
* **learningrate** - Learningrate.
* **hidden_layer_size** - Size of the combined hidden layer.
* **save** - Path for saving the model.
* **load** - Path for loading the model.
* **epochs** - Number of training epochs
* **random_seed** - Random seed.



License
---------------------------

MIT License

Copyright (c) 2017 Marek Rei

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
