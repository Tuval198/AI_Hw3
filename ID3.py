
import numpy as np
import pandas as pd

def entropy (diagnosis_labels):

    #if we have just one label entropy is 0
    if len(diagnosis_labels) == 0:
        return 0

    #number of elements of each group
    labels , counts = np.unique(diagnosis_labels, return_counts=True)
    label_count = dict(zip(labels,counts))
    ent = 0
    print(f"label_count: {label_count}")

    for label in labels:
        prob = label_count[label] / len(diagnosis_labels)
        print(f"prob: {prob}, tag: {label}, log2: {np.log2(prob)}")
        ent -= prob*np.log2(prob)

    return ent


def IG (feature_values, examples):

    # find entropy for each specific feature
    pass

# set of diagnosis, set of examples, set of features, default is majority, select feature is function
def TDIDT (examples, features, default, select_feature):

    if len(examples) == 0:
        return None, {}, default

    # c is classification
    c = majority_label(examples['diagnosis'])
    consistent_examples = [e for e in examples['data'] if examples['diagnosis'][e] == c]

    if len(features) == 0 or len(consistent_examples) == len(examples['diagnosis']):
        # all the examples consistent or we have a noise
        return None, {}, c

    feature = select_feature(features, examples)
    np.delete(features, feature)

    for value in feature.values:
        feature_examples = (e for e in examples['data'] if feature(e) == value)
        sub_tree = {value, TDIDT(feature_examples, features, c, select_feature)}

    return feature, sub_tree, c



def majority_label (diagnosis):
    diagnose, count = np.unique(diagnosis, return_counts=True)
    max_count = np.max(count)
    diagnose_dict = dict(zip(count, diagnose))
    majority_diagnose = diagnose_dict[max_count]
    print(diagnose_dict)
    print(majority_diagnose)

    return majority_diagnose

def ID3 (examples, features):

    majority = majority_label(examples['diagnosis'])

if __name__ == "__main__":
    labels1 = np.array(['B', 'B', 'B', 'B', 'B', 'B', 'B', 'M', 'B', 'B', 'B', 'B'])
    labels2 = np.array(['B', 'B', 'B', 'B', 'B', 'B', 'M', 'M', 'M', 'B', 'M', 'B'])
    labels3 = np.array(['B', 'B', 'B', 'B', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M'])
    ent1 = entropy(labels1)
    ent2 = entropy(labels2)
    ent3 = entropy(labels3)
    print(ent1, ent2, ent3)

    test_file = pd.read_csv('AI_Hw3/test.csv',header=None)
    train_file = pd.read_csv('AI_Hw3/train.csv',header=None)

    test_diagnosis = np.array(test_file[0][1:])
    test_examples = np.array(test_file[1:])
    train_diagnosis = np.array(train_file[0][1:])
    train_data = np.array(train_file[1:])

    train_examples = {
        'diagnosis': train_diagnosis,
        'data': train_data
    }

    features = np.array

    ID3(train_examples, features)
