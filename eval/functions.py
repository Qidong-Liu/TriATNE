from collections import defaultdict

import numpy
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score
from sklearn.utils import shuffle as sk_shuffle
import gensim

def read_cora_labels(labels_file):
    labels = {}
    all_labels = set()
    with open(labels_file, encoding='utf-8') as hfile:
        for i, line in enumerate(hfile):
            # omit the 1st and last item in the list as they are always empty
            node_labels = set(line.strip().split('/')[1:-1])
            labels[str(i)] = node_labels # i start from 0
            all_labels.update(node_labels)
    return labels, all_labels

def read_blogcat_labels(labels_file, delimiter=','):
    labels = defaultdict(set)
    all_labels = set()
    with open(labels_file, encoding='utf-8') as hfile:
        for line in hfile:
            node, label = line.strip().split(delimiter)
            labels[node].add(label)
            all_labels.add(label)
    return labels, all_labels

def read_ml_class_labels(labels_file):
    labels = {}
    with open(labels_file, encoding='utf-8') as hfile:
        for line in hfile:
            node, node_labels = line.split(' ')
            labels[node] = set(node_labels.split(','))
    return labels

def get_label_repr(node_labels, label_list):
    # return a binary vector where each index corresponds to a label
    result = []
    for label in label_list:
        if label in node_labels:
            result.append(1)
        else:
            result.append(0)
    return result


def __get_repr(labels, label_list, emb, shuffle=True):
    # sort all labels so the encodings are consistent
    label_list_sorted = sorted(label_list)

    X = []
    y = []
    for node, node_labels in labels.items():
        X.append(emb[node])
        y.append(get_label_repr(node_labels, label_list_sorted))
    if shuffle:
        return sk_shuffle(numpy.asarray(X), numpy.asarray(y))
    return numpy.asarray(X), numpy.asarray(y)


def __get_f1(predictions, y, number_of_labels):
    # find the indices (labels) with the highest probabilities (ascending order)
    pred_sorted = numpy.argsort(predictions, axis=1)

    # the true number of labels for each node
    num_labels = numpy.sum(y, axis=1)

    # we take the best k label predictions for all nodes, where k is the true number of labels
    pred_reshaped = []
    pred_set = set()
    for pr, num in zip(pred_sorted, num_labels):
        pred_reshaped.append(pr[-num:].tolist())
        pred_set.update(pr[-num:])

    # convert back to binary vectors
    pred_transformed = MultiLabelBinarizer(range(number_of_labels)).fit_transform(pred_reshaped)
    print('pred_label:', len(pred_set))

    f1_micro = f1_score(y, pred_transformed, average='micro')
    f1_macro = f1_score(y, pred_transformed, average='macro')
    return f1_micro, f1_macro


def get_f1_cross_val(labels, label_list, cv, emb, verbose=True):
    # workaround to predict probabilities during cross-validation
    # "method='predict_proba'" does not seem to work
    class ovrc_prob(OneVsRestClassifier):
        def predict(self, X):
            return self.predict_proba(X)

    if verbose:
        print('transforming inputs...')
    X, y = __get_repr(labels, label_list, emb)

    if verbose:
        print('shape of X: {}'.format(X.shape))
        print('shape of y: {}'.format(y.shape))
        print('running {}-fold cross-validation...'.format(cv))
    ovrc = ovrc_prob(LogisticRegression(solver='liblinear'))
    pred = cross_val_predict(ovrc, X, y, cv=cv)

    return __get_f1(pred, y, len(label_list))


def get_f1(train_labels, labels, label_list, emb, verbose=True):
    if verbose:
        print('transforming inputs...')
    # these labels are already shuffled
    X_train, y_train = __get_repr(train_labels, label_list, emb, False)
    X, y = __get_repr(labels, label_list, emb, False)

    if verbose:
        print('shape of X_train: {}'.format(X_train.shape))
        print('shape of y_train: {}'.format(y_train.shape))
        print('shape of X: {}'.format(X.shape))
        print('shape of y: {}'.format(y.shape))
        print('fitting classifier...')
    ovrc = OneVsRestClassifier(LogisticRegression())
    ovrc.fit(X_train, y_train)

    if verbose:
        print('evaluating...')
    pred = ovrc.predict_proba(X)
    return __get_f1(pred, y, len(label_list))


def read_w2v_emb(file_path, binary):                                                                 #mark
    return gensim.models.KeyedVectors.load_word2vec_format(file_path, binary=binary).wv


