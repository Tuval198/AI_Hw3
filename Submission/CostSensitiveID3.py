
from ID3 import get_data, calculate_loss, majority_label, split, BEST_M, DEFAULT_M
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

DEFAULT_MAX_DEPTH = np.inf
INIT_DEPTH = 0
DEFAULT_B_WEIGHT = 1
BEST_B = 8
BEST_DEPTH = 3

# node of decision tree
class Node:

    def __init__(self, default, depth=INIT_DEPTH):
        self.threshold = None
        self.feat = None
        self.classify = default
        self.left_son = None
        self.right_son = None
        self.depth = depth

    # id3 algorithm
    def fit(self, features, examples, diagnosis, M=DEFAULT_M, max_depth=DEFAULT_MAX_DEPTH, B_weight=DEFAULT_B_WEIGHT):

        best_ig = 0
        best_feat = None
        best_threshold = 0

        if self.depth <= max_depth:

            # k features
            for feat in range(examples.shape[1]):
                sorted_values = np.unique(examples[:, feat])

                # go through all values 1<=j<=k-1 in the feature to select threshold with highest ig
                for j in range(0, len(sorted_values) - 1):
                    threshold = (sorted_values[j] + sorted_values[j + 1]) / 2
                    bigger_indexes, smaller_indexes = split(examples, threshold, feat)
                    bigger_diagnosis = diagnosis[bigger_indexes]
                    smaller_diagnosis = diagnosis[smaller_indexes]
                    information_gain = ig(bigger_diagnosis, smaller_diagnosis, diagnosis)

                    # get the optimal ig with highest index
                    if information_gain >= best_ig:
                        best_ig = information_gain
                        best_feat = feat
                        best_threshold = threshold

            self.threshold = best_threshold
            self.feat = best_feat

            # get childs of best split
            bigger_indexes, smaller_indexes = split(examples, best_threshold, best_feat)
            smaller_diagnosis, bigger_diagnosis = diagnosis[smaller_indexes], diagnosis[bigger_indexes]

            # find majority
            smaller_classify = majority_label(smaller_diagnosis)
            bigger_classify = majority_label(bigger_diagnosis)

            smaller_consistent_len = len(smaller_diagnosis[(smaller_diagnosis[:] == smaller_classify)])
            bigger_consistent_len = len(bigger_diagnosis[(bigger_diagnosis[:] == bigger_classify)])
            smaller_len, bigger_len = len(smaller_diagnosis), len(bigger_diagnosis)
            new_depth = self.depth + 1

            # more likely to prune for 'B' classify
            if bigger_classify == 'B':
                bigger_len /= B_weight

            if smaller_classify == 'B':
                smaller_len /= B_weight

            if smaller_len == 0:
                self.left_son = None

            # prune
            elif smaller_len < M:
                # leaf with father classify
                self.left_son = Node(self.classify, new_depth)

            # all the examples consistent or prune
            elif smaller_consistent_len == len(smaller_diagnosis):
                # leaf
                self.left_son = Node(smaller_classify, new_depth)

            else:
                self.left_son = Node(smaller_classify, new_depth)
                smaller_examples = examples[smaller_indexes, :]
                self.left_son.fit(features, smaller_examples, smaller_diagnosis, M, max_depth, B_weight)

            if bigger_len == 0:
                self.right_son = None

            # prune
            elif bigger_len < M:
                # leaf with father classify
                self.right_son = Node(self.classify, new_depth)

            # all the examples consistent or prune
            elif bigger_consistent_len == len(bigger_diagnosis):
                # leaf
                self.right_son = Node(bigger_classify, new_depth)

            else:
                self.right_son = Node(bigger_classify, new_depth)
                bigger_examples = examples[bigger_indexes, :]
                self.right_son.fit(features, bigger_examples, bigger_diagnosis, M, max_depth, B_weight)

    # return numpy array of classifications where 'B' is 1, 'M' is 0
    def predict(self, examples):

        # no children
        if self.left_son is None and self.right_son is None:
            if self.classify == 'B':
                return 1
            else:
                return 0

        preds = np.zeros(len(examples))

        bigger_indexes, smaller_indexes = split(examples, self.threshold, self.feat)
        bigger_examples = examples[bigger_indexes, :]
        smaller_examples = examples[smaller_indexes, :]
        preds[bigger_indexes] = self.right_son.predict(bigger_examples)
        preds[smaller_indexes] = self.left_son.predict(smaller_examples)
        # print(f"depth: {self.depth}")

        return preds


# calculate entropy
def loss_entropy(diagnosis_labels):
    # if we have just one label entropy is 0
    if len(diagnosis_labels) == 0:
        return 0

    # number of elements of each group
    labels, counts = np.unique(diagnosis_labels, return_counts=True)
    label_count = dict(zip(labels, counts))
    ent = 0

    for label in labels:
        if label == 'B':
            prob = label_count['B']*0.1 / len(diagnosis_labels)
        else:
            prob = label_count['M'] / len(diagnosis_labels)
        ent -= prob * np.log2(prob)

    return ent


# calculate information gain
def ig(bigger__labels, smaller__labels, diagnosis):
    bigger_ent = loss_entropy(bigger__labels)
    bigger_fraction = len(bigger__labels) / len(diagnosis)
    smaller_ent = loss_entropy(smaller__labels)
    smaller_fraction = len(smaller__labels) / len(diagnosis)
    ent = loss_entropy(diagnosis)

    information_gain = ent - (bigger_ent * bigger_fraction + smaller_ent * smaller_fraction)

    return information_gain


def loss_experiment(train_examples, train_diagnosis, features):

    M_values = [1, 2, 3]
    max_depths = [1, 2, 3, 4]
    B_weight_values = [1, 3, 5, 8, 10, 12, 15, 25, 50]
    id = 312419971
    parameters = []

    for M in M_values:
        for depth in max_depths:
            mean_losses = []
            for B in B_weight_values:
                print(f"depth is: {depth}, M is: {M}")
                mean_loss = Kfold_loss(train_examples, train_diagnosis, features, M, id, depth, B)
                mean_losses.append(mean_loss)
                parameters.append((B, depth, M))

            plt.plot(B_weight_values, mean_losses, label=f'Depth is: {depth}, M is: {M}', marker='o',
                     markerfacecolor='black', markersize=12)

    plt.xlabel('B weight')
    plt.ylabel('loss')
    plt.title(f'Prune with K-Fold Cross Validation')
    plt.legend()
    plt.show()
    print(mean_losses)
    print(parameters)


def Kfold_loss(examples, diagnosis, features, M, id, depth, B):

    kf = KFold(n_splits=5, shuffle=True, random_state=id)
    losses = []

    for train_index, test_index in kf.split(examples):
        train_examples, train_diagnosis = examples[train_index], diagnosis[train_index]
        test_examples, test_diagnosis = examples[test_index], diagnosis[test_index]

        default = majority_label(train_diagnosis)
        root = Node(default, 0)
        root.fit(features, train_examples, train_diagnosis, M, depth, B)
        predictions = root.predict(test_examples)
        test_loss = calculate_loss(predictions, test_diagnosis)
        losses.append(test_loss)

    mean_loss = np.sum(losses) / len(losses)

    return mean_loss


def main():
    # get data
    test_file_path = 'test.csv'
    train_file_path = 'train.csv'
    train_examples, train_diagnosis, features = get_data(train_file_path)
    test_examples, test_diagnosis, _ = get_data(test_file_path)

    """Question 4.3 - best loss"""

    # experiment
    # loss_experiment(train_examples, train_diagnosis, features)

    # fit
    default = majority_label(train_diagnosis)
    tree_root = Node(default)
    tree_root.fit(features, train_examples, train_diagnosis, BEST_M, BEST_DEPTH, BEST_B)

    # predict
    predictions = tree_root.predict(test_examples)

    # calculate loss after improvement
    loss = calculate_loss(predictions, test_diagnosis)
    print(loss)


if __name__ == "__main__":
    main()