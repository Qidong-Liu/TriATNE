import numpy as np
from sklearn.metrics import roc_auc_score

def AUC(sess, model, positive, negative):
    score_pos = sess.run(model.P_pred_score, feed_dict={model.u: positive[:, 0], model.v: positive[:, 1]})
    score_neg = sess.run(model.P_pred_score, feed_dict={model.u: negative[:, 0], model.v: negative[:, 1]})

    max_pos = np.max(score_pos)
    min_pos = np.min(score_pos)
    max_neg = np.max(score_neg)
    min_neg = np.min(score_neg)

    max_all = np.maximum(max_pos, max_neg)

    score_pos_n = score_pos - max_all
    score_neg_n = score_neg - max_all

    #preds_pos = 1./(1. + np.exp(-score_pos_n))
    #preds_neg = 1./(1. + np.exp(-score_neg_n))
    preds_pos = np.exp(score_pos_n)/(1. + np.exp(score_pos_n))
    preds_neg = np.exp(score_neg_n)/(1. + np.exp(score_neg_n))

    print('max_pos: ', max_pos, 'min_pos: ', min_pos, 'max_neg: ', max_neg, 'min_neg: ', min_neg)

    preds_all = np.hstack([preds_pos, preds_neg])
    labels_all = np.hstack([np.ones(len(preds_pos)), np.zeros(len(preds_neg))])

    roc_score = roc_auc_score(labels_all, preds_all)
    return roc_score

def AUC2(sess, model, positive, negative):

    score_pos = sess.run(model.S_pred_score, feed_dict={model.u: positive[:, 0], model.v: positive[:, 1]})
    score_neg = sess.run(model.S_pred_score, feed_dict={model.u: negative[:, 0], model.v: negative[:, 1]})

    #preds_pos = 1./(1. + np.exp(-score_pos))
    #preds_neg = 1./(1. + np.exp(-score_neg))

    #preds_all = np.hstack([preds_pos, preds_neg])
    #labels_all = np.hstack([np.ones(len(preds_pos)), np.zeros(len(preds_neg))])

    max_pos = np.max(score_pos)
    min_pos = np.min(score_pos)
    max_neg = np.max(score_neg)
    min_neg = np.min(score_neg)

    max_all = np.maximum(max_pos, max_neg)

    score_pos_n = score_pos - max_all
    score_neg_n = score_neg - max_all

    # preds_pos = 1./(1. + np.exp(-score_pos_n))
    # preds_neg = 1./(1. + np.exp(-score_neg_n))
    preds_pos = np.exp(score_pos_n) / (1. + np.exp(score_pos_n))
    preds_neg = np.exp(score_neg_n) / (1. + np.exp(score_neg_n))

    print('max_pos: ', max_pos, 'min_pos: ', min_pos, 'max_neg: ', max_neg, 'min_neg: ', min_neg)

    preds_all = np.hstack([preds_pos, preds_neg])
    labels_all = np.hstack([np.ones(len(preds_pos)), np.zeros(len(preds_neg))])

    roc_score = roc_auc_score(labels_all, preds_all)
    return roc_score
