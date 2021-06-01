
from ID3 import Node, majority_label, BEST_M
from KNNForest import calculate_error, get_tree_examples, calculate_centroid, \
    calculate_dist_to_centroid, calculate_mean_error

import random
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

DEFAULT_BONUS = 0
BEST_BONUS = 2
BEST_N = 20
BEST_K = 13
BEST_P = 0.4
NUM_OF_AVERAGE_ERRORS = 5

class Forest:
    def __init__(self, N, k):
        self.num_of_trees = N
        self.k = k
        self.trees = []
        self.centroids = []

    def fit(self, examples, diagnosis, features, p):

        n = len(diagnosis)

        for tree_index in range(self.num_of_trees):
            # get random examples for each tree
            tree_examples, tree_diagnosis = get_tree_examples(p, n, examples, diagnosis)
            # fit tree
            default = majority_label(tree_diagnosis)
            tree_root = Node(default)
            tree_root.fit(features, tree_examples, tree_diagnosis, BEST_M)
            self.trees.append(tree_root)
            self.centroids.append(calculate_centroid(tree_examples))

    def find_k_trees(self, example, k):
        distances = []
        trees = []
        best_k_dist = []

        for i in range(self.num_of_trees):
            distances.append(calculate_dist_to_centroid(self.centroids[i], example))
        best_k_idx = (np.argpartition(distances, -k)[-k:])
        for j in best_k_idx:
            trees.append(self.trees[j])
            best_k_dist.append(distances[j])

        return trees, best_k_dist

    def predict(self, examples, dist_bonus):
        predictions = []
        for e in examples:
            k_predictions = []
            k_trees, k_distances = self.find_k_trees(e, self.k)
            for t in k_trees:
                k_predictions.append(t.predict_example(e))
            pred = predict_by_distance(k_predictions, k_distances, dist_bonus)
            predictions.append(pred)

        return predictions


# get csv file and open it
# return numpy arrays of diagnosis and examples and a list of features
def get_normalized_data():
    test_file_path = 'test.csv'
    train_file_path = 'train.csv'

    # train data
    train_examples = pd.read_csv(train_file_path)
    features = train_examples.columns.tolist()
    features.remove("diagnosis")

    train_examples_data = train_examples[features].to_numpy()
    train_examples_diagnosis = train_examples["diagnosis"].to_numpy()

    max_args = np.zeros(len(features))
    min_args = np.zeros(len(features))

    for feat in range(len(features)):
        max_args[feat] = np.max(train_examples_data[:, feat])
        min_args[feat] = np.min(train_examples_data[:, feat])

    # test data
    test_examples = pd.read_csv(test_file_path)
    test_examples_data = test_examples[features].to_numpy()
    test_examples_diagnosis = test_examples["diagnosis"].to_numpy()

    # normalize
    for feat in range(len(features)):
        train_examples_data[:, feat] = (train_examples_data[:, feat] - min_args[feat]) / \
                                       (max_args[feat] - min_args[feat])
        test_examples_data[:, feat] = (test_examples_data[:, feat] - min_args[feat]) / \
                                       (max_args[feat] - min_args[feat])

    return train_examples_data, train_examples_diagnosis, features, test_examples_data, test_examples_diagnosis


def predict_by_distance(predictions, distances, dist_bonus):
    # closer have more weight
    dist_weights = np.true_divide(1, distances)
    sorted_weights = dist_weights[np.argsort(dist_weights)]
    B_weight = 0
    M_weight = 0
    B_count = 0
    M_count = 0

    # print(f"dist_weights before: {dist_weights}")
    # # closer get more bonus (i is bigger)
    for j in range(len(sorted_weights)):
        dist_weights[j] *= j*dist_bonus

    # print(f"dist_weights after: {dist_weights}")

    for i in range(len(dist_weights)):
        if predictions[i] == 'B':
            B_count += 1
            B_weight += dist_weights[i]+1
        else:
            M_count += 1
            M_weight += dist_weights[i]+1

    # print(f"dist_bonus: {dist_bonus}, B weight: {B_weight}, M_weight: {M_weight}, B_count: {B_count},
    #       M_count: {M_count}")
    if B_weight >= M_weight:
        return 'B'
    else:
        return 'M'


def bonus_experiment(train_examples, train_diagnosis, features):
    dist_bonus = [0, 1, 2, 4, 8, 10]
    id = 312419971
    mean_errors = []

    for bonus in dist_bonus:
        mean_error = Kfold_error(train_examples, train_diagnosis, features, BEST_N, BEST_K, BEST_P, id, bonus)
        print(f"bonus: {bonus}, mean error: {mean_error}")
        mean_errors.append(mean_error)

    plt.plot(dist_bonus, mean_errors, color='blue', marker='o', markerfacecolor='black', markersize=12)
    plt.xlabel('bonus')
    plt.ylabel('Mean Error')
    plt.title('K-Fold Cross Validation')
    plt.show()

    print(mean_errors)


def Kfold_error(examples, diagnosis, features, n, k, p, id, bonus=DEFAULT_BONUS):
    kf = KFold(n_splits=5, shuffle=True, random_state=id)
    errors = []

    for train_index, test_index in kf.split(examples):
        train_examples, train_diagnosis = examples[train_index], diagnosis[train_index]
        test_examples, test_diagnosis = examples[test_index], diagnosis[test_index]
        sum_error = 0
        # get mean error
        error = calculate_mean_error(train_examples, train_diagnosis, features, test_examples, test_diagnosis, bonus)
        errors.append(error)

    mean_error = np.sum(errors) / len(errors)

    return mean_error


def calculate_mean_error(train_examples, train_diagnosis, features, test_examples, test_diagnosis,
                         bonus=DEFAULT_BONUS):
    errors = []
    for i in range(NUM_OF_AVERAGE_ERRORS):
        f = Forest(BEST_N, BEST_K)
        f.fit(train_examples, train_diagnosis, features, BEST_P)
        predictions = f.predict(test_examples, bonus)
        error = calculate_error(predictions, test_diagnosis)
        print(f"accuracy is: {error}")
        errors.append(error)

    mean_error = np.sum(errors) / len(errors)
    return mean_error

def improve_features(train_examples, train_diagnosis, features):
    new_features = PolynomialFeatures(2)


def main():
    # get normalized data
    train_examples, train_diagnosis, features, test_examples, test_diagnosis = get_normalized_data()

    """Question 7.2 - KNN forest improved"""

    # bonus_experiment(train_examples, train_diagnosis, features)

    # calculate mean error because we are using random data
    # mean_error = calculate_mean_error(train_examples, train_diagnosis, features, test_examples, test_diagnosis,
    #                                   BEST_BONUS)
    # print(f"mean accuracy is: {mean_error}")

    # calculate error
    f = Forest(BEST_N, BEST_K)
    f.fit(train_examples, train_diagnosis, features, BEST_P)
    predictions = f.predict(test_examples, BEST_BONUS)
    error = calculate_error(predictions, test_diagnosis)
    print(f"accuracy is: {error}")


if __name__ == "__main__":
    main()
