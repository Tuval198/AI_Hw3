import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

DEFAULT_M = 0
BEST_M = 1

# get csv file and open it
# return numpy arrays of diagnosis and examples and a list of features
def get_data(file_path):
    # DataFrame object
    examples = pd.read_csv(file_path)
    features = examples.columns.tolist()
    features.remove("diagnosis")

    examples_data = examples[features].to_numpy()
    examples_diagnosis = examples["diagnosis"].to_numpy()

    return examples_data, examples_diagnosis, features


# calculate entropy
def entropy(diagnosis_labels):
    # if we have just one label entropy is 0
    if len(diagnosis_labels) == 0:
        return 0

    # number of elements of each group
    labels, counts = np.unique(diagnosis_labels, return_counts=True)
    label_count = dict(zip(labels, counts))
    ent = 0

    for label in labels:
        prob = label_count[label] / len(diagnosis_labels)
        ent -= prob * np.log2(prob)

    return ent


# calculate information gain
def ig(bigger__labels, smaller__labels, diagnosis):
    bigger_ent = entropy(bigger__labels)
    bigger_fraction = len(bigger__labels) / len(diagnosis)
    smaller_ent = entropy(smaller__labels)
    smaller_fraction = len(smaller__labels) / len(diagnosis)
    ent = entropy(diagnosis)

    information_gain = ent - (bigger_ent * bigger_fraction + smaller_ent * smaller_fraction)

    return information_gain


# split to a bigger then threshold group and smaller one
def split(examples, threshold, feat):
    bigger_indexes = (examples[:, feat] >= threshold)
    smaller_indexes = (examples[:, feat] < threshold)

    return bigger_indexes, smaller_indexes


# find the majority of the group
def majority_label(diagnosis):
    diagnose, count = np.unique(diagnosis, return_counts=True)
    max_count = np.max(count)
    diagnose_dict = dict(zip(count, diagnose))
    majority_diagnose = diagnose_dict[max_count]

    return majority_diagnose


# node of decision tree
class Node:

    def __init__(self, default):
        self.threshold = None
        self.feat = None
        self.classify = default
        self.left_son = None
        self.right_son = None

    # id3 algorithm
    def fit(self, features, examples, diagnosis, M=DEFAULT_M):

        best_ig = 0
        best_feat = None
        best_threshold = 0

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

        if smaller_len == 0:
            self.left_son = None

        # prune
        elif smaller_len < M:
            # leaf with father classify
            self.left_son = Node(self.classify)

        # all the examples consistent or prune
        elif smaller_consistent_len == len(smaller_diagnosis):
            # leaf
            self.left_son = Node(smaller_classify)

        else:
            self.left_son = Node(smaller_classify)
            smaller_examples = examples[smaller_indexes, :]
            self.left_son.fit(features, smaller_examples, smaller_diagnosis, M)

        if bigger_len == 0:
            self.right_son = None

        # prune
        elif bigger_len < M:
            # leaf with father classify
            self.right_son = Node(self.classify)

        # all the examples consistent or prune
        elif bigger_consistent_len == len(bigger_diagnosis):
            # leaf
            self.right_son = Node(bigger_classify)

        else:
            self.right_son = Node(bigger_classify)
            bigger_examples = examples[bigger_indexes, :]
            self.right_son.fit(features, bigger_examples, bigger_diagnosis, M)

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

    # for 6.1 KNN forest
    def predict_example(self, example):
        if self.left_son is None and self.right_son is None:
            return self.classify

        if example[self.feat] >= self.threshold:
            classify = self.right_son.predict_example(example)
        else:
            classify = self.left_son.predict_example(example)

        return classify


# id3 predictions vs real predictions
def calculate_error(predictions, diagnosis):
    # predictions is 1 for 'B' or 0 for 'M'
    diagnosis = np.where(diagnosis == 'B', 1, diagnosis)
    diagnosis = np.where(diagnosis == 'M', 0, diagnosis)

    true_predictions = np.sum((predictions == diagnosis))
    error = true_predictions / len(diagnosis)

    return error


# loss of FP vs FN
def calculate_loss(predictions, diagnosis):
    # predictions is 1 for 'B' or 0 for 'M'
    diagnosis = np.where(diagnosis == 'B', 1, diagnosis)
    diagnosis = np.where(diagnosis == 'M', 0, diagnosis)

    false_predictions = (predictions != diagnosis)
    false_positive = np.sum((diagnosis[false_predictions] == 1))
    false_negative = np.sum((diagnosis[false_predictions] == 0))
    loss = (0.1 * false_positive + false_negative) / len(diagnosis)

    return loss


""" 3.3 - To plot the graph : comment out the call to this function on the main func"""

def experiment(train_examples, train_diagnosis, features):
    M_values = [1, 2, 4, 8, 16, 32, 64, 128]
    id = 312419971
    mean_errors = []

    for M in M_values:
        mean_error = Kfold_error(train_examples, train_diagnosis, features, M, id)
        mean_errors.append(mean_error)

    plt.plot(M_values, mean_errors, color='blue', marker='o', markerfacecolor='black', markersize=8)
    plt.xlabel('M')
    plt.ylabel('Error')
    plt.title('Prune with K-Fold Cross Validation')
    plt.show()


def Kfold_error(examples, diagnosis, features, M, id):
    kf = KFold(n_splits=5, shuffle=True, random_state=id)
    errors = []

    for train_index, test_index in kf.split(examples):
        train_examples, train_diagnosis = examples[train_index], diagnosis[train_index]
        test_examples, test_diagnosis = examples[test_index], diagnosis[test_index]

        default = majority_label(train_diagnosis)
        root = Node(default)
        root.fit(features, train_examples, train_diagnosis, M)
        predictions = root.predict(test_examples)
        test_error = calculate_error(predictions, test_diagnosis)
        errors.append(test_error)

    mean_error = np.sum(errors) / len(errors)

    return mean_error


def main():
    # get data
    test_file_path = 'test.csv'
    train_file_path = 'train.csv'
    train_examples, train_diagnosis, features = get_data(train_file_path)
    test_examples, test_diagnosis, _ = get_data(test_file_path)

    """Question 1.1 - id3"""

    # make root and use fit to find the tree, M is 0 (no prune)
    default = majority_label(train_diagnosis)
    tree_root = Node(default)
    tree_root.fit(features, train_examples, train_diagnosis)

    # use pred to get predictions where 'B' is 1 and 'M' is 0
    predictions = tree_root.predict(test_examples)

    error = calculate_error(predictions, test_diagnosis)
    print(error)

    """Question 3.2 + 3.3 - early prune"""
    """ comment out to plot the graph  """

    # experiment(train_examples, train_diagnosis, features)

    """Question 3.4 - best M"""

    # best M
    # tree_root.fit(features, train_examples, train_diagnosis, BEST_M)
    # predictions = tree_root.predict(test_examples)
    # error = calculate_error(predictions, test_diagnosis)
    # print(error)

    """Question 4.1 - best M loss"""

    # loss = calculate_loss(predictions, test_diagnosis)
    # print(f"loss is: {loss}")


if __name__ == "__main__":
    main()
