import sys

from model import WordPairClassifier

if __name__ == "__main__":
    modelfile = sys.argv[1]
    word_pairs = [(p.split(":")[0], p.split(":")[1]) for p in sys.argv[2:]]

    print("Loading model...")
    model = WordPairClassifier.load(modelfile)
    word2id = model.config["word2id"]

    for pair in word_pairs:
        if pair[0] in word2id and pair[1] in word2id:
            cost, predicted_labels, scores = model.test([word2id[pair[0]]], [word2id[pair[1]]], [0])
            print(pair[0] + "\t" + pair[1] + "\t" + str(predicted_labels[0]) + "\t" + str(scores[0]))
        else:
            print(pair[0] + "\t" + pair[1] + "\t" + "words not in vocabulary")

