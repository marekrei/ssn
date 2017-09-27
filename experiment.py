import sys
import collections
import random
import numpy
import os
import cPickle
import math
import operator
import scipy
import gc

import config_parser
from model import WordPairClassifier

def read_dataset(dataset_path):
    dataset = []
    with open(dataset_path, 'r') as f:
        for line in f:
            line_parts = line.strip().split()
            assert(len(line_parts) == 3)
            dataset.append((int(line_parts[0]), line_parts[1], line_parts[2]))
    return dataset


def construct_vocabulary(datasets):
    vocab = []
    vocab_set = set()
    for dataset in datasets:
        for entry in dataset:
            for word in [entry[1], entry[2]]:
                if word not in vocab_set:
                    vocab_set.add(word)
                    vocab.append(word)
    return vocab


def load_embeddings_into_matrix(embedding_path, main_separator, remove_multiword, shared_matrix, word2id):
    embedding_matrix = shared_matrix.get_value()
    with open(embedding_path, 'r') as f:
        line_length = None
        for line in f:
            line_parts = line.strip().split(main_separator, 1)
            if len(line_parts) < 2:
                continue
            if remove_multiword == True and len(line_parts[0].split()) > 1:
                continue

            vector = line_parts[1].strip().split()
            if line_length == None:
                line_length = len(vector)
            assert(line_length == len(vector)), "Mismatched vector length: " + str(line_length) + " " + str(len(vector))
            if line_parts[0] in word2id:
                embedding_matrix[word2id[line_parts[0]]] = numpy.array([float(x) for x in vector])
    shared_matrix.set_value(embedding_matrix)


def extend_vocabulary(vocabulary, path1, path2, main_separator, remove_multiword):
    vocab_set = set(vocabulary)
    for path in [path1, path2]:
        if path != None and len(path) > 0:
            with open(path, 'r') as f:
                for line in f:
                    line_parts = line.strip().split(main_separator, 1)
                    if len(line_parts) < 2:
                        continue
                    if remove_multiword == True and len(line_parts[0].split()) > 1:
                        continue
                    if line_parts[0] not in vocab_set:
                        vocab_set.add(line_parts[0])
                        vocabulary.append(line_parts[0])


def evaluate(all_datapoints, all_predicted_scores, all_predicted_labels, all_gold_labels):
    assert(len(all_datapoints) == len(all_predicted_scores))
    assert(len(all_datapoints) == len(all_predicted_labels))
    assert(len(all_datapoints) == len(all_gold_labels))

    count_correct, count_total = 0, 0
    tp, fp, tn, fn = 0.0, 0.0, 0.0, 0.0

    for i in range(len(all_datapoints)):
        if all_gold_labels[i] == 1:
            if all_predicted_labels[i] == 1:
                tp += 1.0
            else:
                fn += 1.0
        elif all_gold_labels[i] == 0:
            if all_predicted_labels[i] == 1:
                fp += 1.0
            else:
                tn += 1.0
        if all_gold_labels[i] == all_predicted_labels[i]:
            count_correct += 1
        count_total += 1

    assert(int(tp + fn + fp + tn) == count_total)
    pearsonr = scipy.stats.pearsonr([x[0] for x in all_datapoints], all_predicted_scores)[0]

    results = collections.OrderedDict()
    results["count_correct"] = count_correct
    results["count_total"] = count_total
    results["tp"] = tp
    results["tn"] = tn
    results["fp"] = fp
    results["fn"] = fn
    results["accuracy"] = float(count_correct) / float(count_total)
    p = (tp / (tp + fp)) if (tp + fp) > 0.0 else 0.0
    r = (tp / (tp + fn)) if (tp + fn) > 0.0 else 0.0
    results["p"] = p
    results["r"] = r
    results["fmeasure"] = (2.0 * p * r / (p+r)) if (p+r) > 0.0 else 0.0
    results["pearsonr"] = pearsonr
    return results


def process_dataset(dataset, model, word2id, is_testing, config, name):
    if is_testing == False and config["shuffle_training_data"] == True:
        random.shuffle(dataset)

    cost_sum = 0.0
    all_datapoints, all_predicted_scores, all_predicted_labels, all_gold_labels = [], [], [], []
    for i in range(0, len(dataset), config["examples_per_batch"]):
        batch = dataset[i:i+config["examples_per_batch"]]
        if is_testing == False and config["shuffle_training_data"] == True:
            random.shuffle(batch)

        word1_ids = [word2id[word1] for label, word1, word2 in batch]
        word2_ids = [word2id[word2] for label, word1, word2 in batch]
        label_ids = [(1 if label > 0 else 0) for label, word1, word2 in batch]


        if is_testing == True:
            cost, predicted_labels, scores = model.test(word1_ids, word2_ids, label_ids)
        else:
            cost, predicted_labels, scores = model.train(word1_ids, word2_ids, label_ids, config["learningrate"])

        assert(math.isnan(cost) == False and math.isinf(cost) == False), "Cost is "+str(cost) + ", exiting."

        cost_sum += cost

        for x in batch:
            all_datapoints.append(x)
        for x in scores:
            all_predicted_scores.append(x)
        for x in predicted_labels:
            all_predicted_labels.append(x)
        for x in label_ids:
            all_gold_labels.append(x)

        gc.collect()

    results = evaluate(all_datapoints, all_predicted_scores, all_predicted_labels, all_gold_labels)
    results["cost"] = cost_sum
    for key in results:
        print(name + "_" + key + ": " + str(results[key]))

    return results



def run_experiment(config_path):
    config = config_parser.parse_config("config", config_path)
    random.seed(config["random_seed"] if "random_seed" in config else 123)
    temp_model_path = config_path + ".model"

    if "load" in config and config["load"] is not None and len(config["load"]) > 0:
        model = WordPairClassifier.load(config["load"])
        data_test = read_dataset(config["path_test"])
        word2id = model.config["word2id"]
        config = model.config
        process_dataset(data_test, model, word2id, True, config, "test")
        sys.exit()

    data_train = read_dataset(config["path_train"])
    data_dev = read_dataset(config["path_dev"])
    data_test = read_dataset(config["path_test"])

    vocabulary = construct_vocabulary([data_train, data_dev, data_test])
    if "extend_vocabulary" in config and config["extend_vocabulary"] == True:
        extend_vocabulary(vocabulary, config["word_embedding_path_a"], config["word_embedding_path_b"], "\t", True)
    word2id = collections.OrderedDict()
    for i in range(len(vocabulary)):
        word2id[vocabulary[i]] = i
    assert(len(word2id) == len(set(vocabulary)))
    config["n_words"] = len(vocabulary)
    config["word2id"] = word2id

    model = WordPairClassifier(config)
    load_embeddings_into_matrix(config["word_embedding_path_a"], "\t", True, model.word_embedding_matrix_A, word2id)
    if config["word_embedding_size_b"] > 0:
        load_embeddings_into_matrix(config["word_embedding_path_b"], "\t", True, model.word_embedding_matrix_B, word2id)

    for key, val in config.items():
        if key not in ["word2id"]:
            print(str(key) + ": " + str(val))

    best_score = 0.0
    for epoch in range(config["epochs"]):
        print("epoch: " + str(epoch))
        results_train = process_dataset(data_train, model, word2id, False, config, "train")
        results_dev = process_dataset(data_dev, model, word2id, True, config, "dev")
        score_dev = results_dev["fmeasure"]

        if epoch == 0 or score_dev > best_score:
            best_epoch = epoch
            best_score = score_dev
            model.save(temp_model_path)
        print("best_epoch: " + str(best_epoch))
        print("best_dev_fscore: " + str(best_score))

        if config["stop_if_no_improvement_for_epochs"] > 0 and (epoch - best_epoch) >= config["stop_if_no_improvement_for_epochs"]:
            break

    if os.path.isfile(temp_model_path):
        model = WordPairClassifier.load(temp_model_path)
        os.remove(temp_model_path)

    if "save" in config and config["save"] is not None and len(config["save"]) > 0:
        model.save(config["save"])

    score_dev = process_dataset(data_dev, model, word2id, True, config, "dev_final")
    score_test = process_dataset(data_test, model, word2id, True, config, "test")


if __name__ == "__main__":
    run_experiment(sys.argv[1])
