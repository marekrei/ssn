import sys
import theano
import numpy
import collections
import cPickle
import lasagne

sys.setrecursionlimit(50000)
floatX=theano.config.floatX

class WordPairClassifier(object):
    def __init__(self, config):
        self.config = config
        self.params = collections.OrderedDict()
        self.rng = numpy.random.RandomState(config["random_seed"])

        word1_ids = theano.tensor.ivector('word1_ids')
        word2_ids = theano.tensor.ivector('word2_ids')
        label_ids = theano.tensor.ivector('label_ids')
        learningrate = theano.tensor.fscalar('learningrate')
        is_training = theano.tensor.iscalar('is_training')

        self.word_embedding_matrix_A = self.create_parameter_matrix('word_embedding_matrix_a', (config["n_words"], config["word_embedding_size_a"]))
        scores = self.construct_network(word1_ids, word2_ids, self.word_embedding_matrix_A, config["word_embedding_size_a"], self.config, "A")

        if config["late_fusion"] == True:
            self.word_embedding_matrix_B = self.create_parameter_matrix('word_embedding_matrix_b', (config["n_words"], config["word_embedding_size_b"]))
            scoresB = self.construct_network(word1_ids, word2_ids, self.word_embedding_matrix_B, config["word_embedding_size_b"], self.config, "B")
            gamma = theano.tensor.nnet.sigmoid(self.create_parameter_matrix('late_fusion_gamma', (1,)))[0]
            scores = gamma * scores + (1.0 - gamma) * scoresB

        cost = 0.0
        label_ids_float = theano.tensor.cast(label_ids, 'float32')
        if config["cost"] == "mse":
            cost += ((scores - label_ids_float)*(scores - label_ids_float)).sum()
        elif config["cost"] == "hinge":
            difference = theano.tensor.abs_(scores - label_ids_float)
            se = (scores - label_ids_float)*(scores - label_ids_float)
            cost += theano.tensor.switch(theano.tensor.gt(difference, 0.4), se, 0.0).sum()

        predicted_labels = theano.tensor.switch(theano.tensor.ge(scores, 0.5), 1, 0)

        if config["update_embeddings"] == True:
            params_ = self.params
        else:
            params_ = self.params.copy()
            del params_['word_embedding_matrix_a']
            if 'word_embedding_matrix_b' in params_:
                del params_['word_embedding_matrix_b']

        if config["cost_l2"] > 0.0:
            for param in params_.values():
                cost += config["cost_l2"] * theano.tensor.sum(param ** 2)

        gradients = theano.tensor.grad(cost, params_.values(), disconnected_inputs='ignore')
        if hasattr(lasagne.updates, config["optimisation_strategy"]):
            update_method = getattr(lasagne.updates, config["optimisation_strategy"])
        else:
            raise ValueError("Optimisation strategy not implemented: " + str(config["optimisation_strategy"]))
        updates = update_method(gradients, params_.values(), learningrate)

        input_vars_test = [word1_ids, word2_ids, label_ids]
        input_vars_train = input_vars_test + [learningrate]
        output_vars = [cost, predicted_labels, scores]
        self.train = theano.function(input_vars_train, output_vars, updates=updates, on_unused_input='ignore', allow_input_downcast = True, givens=({is_training: numpy.cast['int32'](1)}))
        self.test = theano.function(input_vars_test, output_vars, on_unused_input='ignore', allow_input_downcast = True, givens=({is_training: numpy.cast['int32'](0)}))


    def construct_network(self, word1_ids, word2_ids, word_embedding_matrix, word_embedding_size, config, name):
        word1_embeddings = word_embedding_matrix[word1_ids]
        word2_embeddings = word_embedding_matrix[word2_ids]

        if config["gating"] == True:
            gating = theano.tensor.nnet.sigmoid(self.create_layer(word1_embeddings, word_embedding_size, word_embedding_size, name + "_gating"))
            word2_embeddings = word2_embeddings * gating

        if config["embedding_mapping_size"] > 0:
            word1_embeddings = theano.tensor.tanh(self.create_layer(word1_embeddings, word_embedding_size, config["embedding_mapping_size"], "word1_mapping_"+name))
            word2_embeddings = theano.tensor.tanh(self.create_layer(word2_embeddings, word_embedding_size, config["embedding_mapping_size"], "word2_mapping_"+name))
            word_embedding_size = config["embedding_mapping_size"]

        if config["embedding_combination"] == "concat":
            combination = theano.tensor.concatenate([word1_embeddings, word2_embeddings], axis=1)
            combination_size = 2*word_embedding_size
        elif config["embedding_combination"] == "multiply":
            combination = word1_embeddings * word2_embeddings
            combination_size = word_embedding_size
        elif config["embedding_combination"] == "add":
            combination = word1_embeddings + word2_embeddings
            combination_size = word_embedding_size
        else:
            raise ValueError("Unknown combination: " + config["embedding_combination"])

        if config["hidden_layer_size"] > 0:
            combination = theano.tensor.tanh(self.create_layer(combination, combination_size, config["hidden_layer_size"], "hidden_"+name))
            combination_size = config["hidden_layer_size"]

        scores = theano.tensor.nnet.sigmoid(self.create_layer(combination, combination_size, 1, "output_"+name)).reshape((word1_ids.shape[0],))
        return scores


    def save(self, filename):
        dump = {}
        dump["config"] = self.config
        dump["params"] = {}
        for param_name in self.params:
            dump["params"][param_name] = self.params[param_name].get_value()
        f = file(filename, 'wb')
        cPickle.dump(dump, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()


    @staticmethod
    def load(filename, new_config=None, new_output_layer_size=None):
        f = file(filename, 'rb')
        dump = cPickle.load(f)
        f.close()
        model = WordPairClassifier(dump["config"])
        for param_name in model.params:
            assert(param_name in dump["params"])
            model.params[param_name].set_value(dump["params"][param_name])
        return model


    def create_parameter_matrix(self, name, size):
        param_vals = numpy.asarray(self.rng.normal(loc=0.0, scale=0.1, size=size), dtype=floatX)
        self.params[name] = theano.shared(param_vals, name)
        return self.params[name]


    def create_layer(self, input_matrix, input_size, output_size, name):
        W = self.create_parameter_matrix(name + 'W', (input_size, output_size))
        bias = self.create_parameter_matrix(name + 'bias', (output_size,))
        result = theano.tensor.dot(input_matrix, W) + bias
        return result

