
from ID3 import Node, majority_label, BEST_M, get_data
import random
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import pandas as pd

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

        for i in range(self.num_of_trees):
            distances.append(calculate_dist_to_centroid(self.centroids[i], example))
        best_k_idx = (np.argpartition(distances, -k)[-k:])
        for j in best_k_idx:
            trees.append(self.trees[j])

        return trees

    def predict(self, examples):
        predictions = []
        for e in examples:
            k_predictions = []
            trees = self.find_k_trees(e, self.k)
            for t in trees:
                k_predictions.append(t.predict_example(e))
            pred = majority_label(k_predictions)
            predictions.append(pred)

        return predictions


def get_tree_examples(p, n, examples, diagnosis):

    k = int(n*p)
    all_indexes = range(n)
    rand_indexes = random.sample(all_indexes, k=k)
    rand_examples = examples[rand_indexes, :]
    rand_diagnosis = diagnosis[rand_indexes]

    return rand_examples, rand_diagnosis

def calculate_centroid(examples):
    centroid = np.zeros(examples.shape[1])

    for feat in range(examples.shape[1]):
        centroid[feat] = np.average(examples[:, feat])

    return centroid

def calculate_dist_to_centroid(centroid, example):
    sum_sq = np.sum(np.square(centroid - example))
    return np.sqrt(sum_sq)

def calculate_error(predictions, diagnosis):

    true_predictions = np.sum((predictions == diagnosis))
    error = true_predictions / len(diagnosis)

    return error

def knn_forest_experiment(train_examples, train_diagnosis, features):
    N_values = [6, 10, 16, 20]
    K_values = [3, 5, 7, 13, 17]
    p_values = [0.3, 0.4, 0.5, 0.6, 0.7]
    id = 312419971
    parameters = []

    for n in N_values:
        for k in K_values:
            if k > n:
                break
            mean_errors = []
            for p in p_values:
                mean_error = Kfold_error(train_examples, train_diagnosis, features, n, k, p, id)
                print(f"n is: {n}, k is: {k}, p is: {p}, mean error is: {mean_error}")
                mean_errors.append(mean_error)
                parameters.append((n, k, p))

            plt.plot(p_values, mean_errors, label=f'n is: {n}, k is: {k}', marker='o',
                     markerfacecolor='black', markersize=5)

    plt.xlabel('p values')
    plt.ylabel('precise')
    plt.title(f'K-Fold Cross Validation')
    plt.legend()
    plt.show()
    print(mean_errors)
    print(parameters)

def Kfold_error(examples, diagnosis, features, n, k, p, id):
    kf = KFold(n_splits=5, shuffle=True, random_state=id)
    errors = []

    for train_index, test_index in kf.split(examples):
        train_examples, train_diagnosis = examples[train_index], diagnosis[train_index]
        test_examples, test_diagnosis = examples[test_index], diagnosis[test_index]
        sum_error = 0
        # get mean error
        error = calculate_mean_error(train_examples, train_diagnosis, features, test_examples, test_diagnosis)
        errors.append(error)

    mean_error = np.sum(errors) / len(errors)

    return mean_error

def calculate_mean_error(train_examples, train_diagnosis, features, test_examples, test_diagnosis):
    errors = []
    for i in range(NUM_OF_AVERAGE_ERRORS):
        f = Forest(BEST_N, BEST_K)
        f.fit(train_examples, train_diagnosis, features, BEST_P)
        predictions = f.predict(test_examples)
        error = calculate_error(predictions, test_diagnosis)
        print(f"accuracy is: {error}")
        errors.append(error)

    mean_error = np.sum(errors) / len(errors)
    return mean_error

def main():
    # get data
    test_file_path = 'test.csv'
    train_file_path = 'train.csv'
    train_examples, train_diagnosis, features = get_data(train_file_path)
    test_examples, test_diagnosis, _ = get_data(test_file_path)

    """Question 6.1 - KNN forest"""
    # big_data_examples = np.concatenate([train_examples, test_examples])
    # big_data_diagnosis = np.concatenate([train_diagnosis, test_diagnosis])
    # knn_forest_experiment(big_data_examples, big_data_diagnosis, features)

    # calculate mean error because we are using random data
    # mean_error = calculate_mean_error(train_examples, train_diagnosis, features, test_examples, test_diagnosis)
    # print(f"mean accuracy is: {mean_error}")

    # calculate error
    f = Forest(BEST_N, BEST_K)
    f.fit(train_examples, train_diagnosis, features, BEST_P)
    predictions = f.predict(test_examples)
    error = calculate_error(predictions, test_diagnosis)
    print(f"accuracy is: {error}")


if __name__ == "__main__":
    main()

