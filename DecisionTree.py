import pandas as pd
import numpy as np
from sklearn.datasets import load_iris



def test_split(index, value, dataset: pd.DataFrame):
    left = dataset.loc[dataset[index] < value]
    right = dataset.loc[dataset[index] >= value]
    return left, right


def get_desc_values(dataset: pd.DataFrame, class_col, index):
    # groups = split_classes(dataset, class_col)
    maximuns = dataset.groupby(class_col).max()[index].to_list()
    minimuns = dataset.groupby(class_col).min()[index].to_list()
    medians = dataset.groupby(class_col).median()[index].to_list()
    means = dataset.groupby(class_col).mean()[index].to_list()

    return maximuns+minimuns+medians+means
    # means  = dataset[index].groupby(class_col).qu()[index].to_list()


def gini_index(groups, classes, class_col):
    # count all samples at split point
    n_instances = float(sum([len(group.index) for group in groups]))
    # sum weighted Gini index for each group
    gini = 0.0
    for group in groups:
        size = float(len(group))
        # avoid divide by zero
        if size == 0:
            continue
        score = 0.0
        # score the group based on the score for each class
        for class_val in classes:
            p = group[group[class_col] == class_val].shape[0] / size
            score += p * p
        # weight the group score by its relative size
        gini += (1.0 - score) * (size / n_instances)
    return gini


def to_terminal(group, class_col):
    return group[class_col].value_counts().index[0]


def get_split(dataset: pd.DataFrame, class_col):
    """
    Descobrir valor otimo que divide os grupos
    """
    class_values = list(dataset[class_col].unique())
    b_index, b_value, b_score, b_groups = None, 999, 999, None
    for index in dataset.columns:
        if index == class_col:
            continue

        for f in get_desc_values(dataset, class_col, index):
            groups = test_split(index, f, dataset)
            gini = gini_index(groups, class_values, class_col)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, f, gini, groups
    return {'index': b_index, 'value': b_value, 'groups': b_groups}


def split(node: dict, class_col, max_depth, min_size, depth):
    left, right = node['groups']
    del (node['groups'])

    if left.size == 0 or right.size == 0:
        node['left'] = node['right'] = to_terminal(
            pd.concat([left, right]), class_col)
        return

    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(
            left, class_col), to_terminal(right, class_col)
        return

    if len(left) <= min_size:
        node['left'] = to_terminal(left, class_col)
    else:
        node['left'] = get_split(left, class_col)
        split(node['left'], class_col, max_depth, min_size, depth+1)
    # process right child
    if len(right) <= min_size:
        node['right'] = to_terminal(right, class_col)
    else:
        node['right'] = get_split(right, class_col)
        split(node['right'], class_col, max_depth, min_size, depth+1)


def build_tree(dataset, class_col, max_depth, min_size):
    root = get_split(dataset, class_col)
    split(root, class_col, max_depth, min_size, 1)
    return root


def print_tree(node, depth=0):
    if isinstance(node, dict):
        print('%s[%s < %.3f]' %
                ((depth*' ', (node['index']), node['value'])))
        print_tree(node['left'], depth+1)
        print_tree(node['right'], depth+1)
    else:
        print('%s[%s]' % ((depth*' ', node)))



def predict(node, data):
    if isinstance(node, dict):
        if data[node["index"]] < node['value']:
            return predict(node['left'], data)
        return predict(node['right'], data)
    
    return node



if __name__ == "__main__":
    iris = load_iris()

    # np.c_ is the numpy concatenate function
    # which is used to concat iris['data'] and iris['target'] arrays 
    # for pandas column argument: concat iris['feature_names'] list
    # and string list (in this case one string); you can make this anything you'd like..  
    # the original dataset would probably call this ['Species']
    df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                        columns= iris['feature_names'] + ['target'])
    print(df.head())
    tree = build_tree(df, "target", 5, 1)
    print_tree(tree)