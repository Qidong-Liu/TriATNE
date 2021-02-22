import numpy as np

def ndcg_at_K(sess, model, test_positive, train_positive, nodes_set, k=20):
    ndcg = 0.0
    cnt = 0
    for node in test_positive:
        pos_set = set(test_positive[node])
        pred_list = list(nodes_set - set(train_positive[node]))
        if len(pred_list) < k:
            continue

        pred_list_score = sess.run(model.S_pred_score, feed_dict={model.u: [node] * len(pred_list), model.v: pred_list})
        pred_score = zip(pred_list, pred_list_score)
        pred_score = sorted(pred_score, key=lambda x: x[1], reverse=True)

        dcg = 0.0
        for i in range(0, k):
            (candidate, score) = pred_score[i]
            if candidate in pos_set:
                dcg += (1 / np.log2(i + 2))
        n = len(pos_set) if len(pos_set) < k else k
        idcg = np.sum(np.ones(n) / np.log2(np.arange(2, n + 2)))

        ndcg += (dcg / idcg)
        cnt += 1

    return ndcg / float(cnt)


def ndcg_at_k(sess, model, query_pos_test, query_pos_train, query_url_feature, k=5):
    ndcg = 0.0
    cnt = 0
    for query in query_pos_test.keys():
        pos_set = set(query_pos_test[query])
        pred_list = list(set(query_url_feature[query].keys()) - set(query_pos_train.get(query, [])))
        if len(pred_list) < k:
            continue

        pred_list_feature = [query_url_feature[query][url] for url in pred_list]
        pred_list_feature = np.asarray(pred_list_feature)
        pred_list_score = sess.run(model.S_pred_score, feed_dict={model.pred_data: pred_list_feature})
        pred_url_score = zip(pred_list, pred_list_score)
        pred_url_score = sorted(pred_url_score, key=lambda x: x[1], reverse=True)

        dcg = 0.0
        for i in range(0, k):
            (url, score) = pred_url_score[i]
            if url in pos_set:
                dcg += (1 / np.log2(i + 2))
        n = len(pos_set) if len(pos_set) < k else k
        idcg = np.sum(np.ones(n) / np.log2(np.arange(2, n + 2)))

        ndcg += (dcg / idcg)
        cnt += 1

    return ndcg / float(cnt)


def ndcg_at_k_user(sess, model, query_pos_test, query_pos_train, query_url_feature, k=5):
    ndcg_list = []
    query_test_list = sorted(query_pos_test.keys())
    for query in query_test_list:
        pos_set = set(query_pos_test[query])
        pred_list = list(set(query_url_feature[query].keys()) - set(query_pos_train.get(query, [])))
        if len(pred_list) < k:
            continue

        pred_list_feature = [query_url_feature[query][url] for url in pred_list]
        pred_list_feature = np.asarray(pred_list_feature)
        pred_list_score = sess.run(model.pred_score, feed_dict={model.pred_data: pred_list_feature})
        pred_url_score = zip(pred_list, pred_list_score)
        pred_url_score = sorted(pred_url_score, key=lambda x: x[1], reverse=True)

        dcg = 0.0
        for i in range(0, k):
            (url, score) = pred_url_score[i]
            if url in pos_set:
                dcg += (1 / np.log2(i + 2))
        n = len(pos_set) if len(pos_set) < k else k
        idcg = np.sum(np.ones(n) / np.log2(np.arange(2, n + 2)))

        ndcg_list.append(dcg / idcg)

    return ndcg_list