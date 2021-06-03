import torch

from data.corpus.corpus import Corpus


def get_predicted(pred, limit):
    """Get what actually is predicted after applying sigmoid and the limit to nnet results"""
    pred = torch.sigmoid(pred)
    predicted = (pred > limit).long()
    return predicted


def get_classif_metrics(predicted, target):
    """Gets tp, fp, tn, fn classification metrics based on prediction and target passed as parameter"""
    if predicted is not None:
        tps = (predicted == target.long()).byte()
    else:
        if target.is_cuda:
            tps = torch.cuda.ByteTensor([0])
        else:
            tps = torch.ByteTensor([0])
    return tps


def get_results_ilp_naive_test(corpus: Corpus):
    # print('inside get_results_ilp_naive')
    counts = dict()
    for idx, curr_content in enumerate(corpus.corpus_test['content']):
        curr_target = corpus.corpus_test['targets'][idx]
        curr_question_id = curr_content['id']
        if curr_question_id not in counts:
            counts[curr_question_id] = list()
        assert curr_target is not None
        counts[curr_question_id].append(curr_target)
        # if value is not None:
        #     curr_dict[key].append(value)
    assert len(counts) > 0
    return counts


def get_results_info(res_matrix, top_n=1, sort_predicted=True):
    # breaks up matrix into ids, stores in dictionary grouped by id the tuples (score, gold)
    dict_res: dict = dict()
    dict_res_single: dict = dict()
    for row in res_matrix:
        q_id = int(row[0])
        gold_label = int(row[1])
        pred_prob = row[2]
        if q_id not in dict_res:
            dict_res[q_id] = [(gold_label, pred_prob)]
        else:
            dict_res[q_id].append((gold_label, pred_prob))
    if sort_predicted:
        # sorts according to the score assigned by the model
        dict_res = dict([(itm[0], sorted(itm[1], key=lambda x: x[1], reverse=True)[:top_n])
                         for itm in dict_res.items()])
    else:
        # doesn't sort by predicted score: used to evaluate the naive ILP (same order as read from ILP files)
        dict_res = dict([(itm[0], itm[1][:top_n]) for itm in dict_res.items()])

    list_res = list()
    for itm in dict_res.items():
        ress = [it[0] for it in itm[1]]
        if 1 in ress:
            list_res.append(1)
            dict_res_single[itm[0]] = 1
        else:
            list_res.append(0)
            dict_res_single[itm[0]] = 0

    acc = sum(list_res) / len(list_res)
    return acc, len(list_res), dict_res, dict_res_single


def get_accuracy(classif_metrics):
    """Gets the precision based on classification metrics passed as parameter"""
    try:
        sum_tp = classif_metrics.sum().item()
        to_ret = sum_tp / (classif_metrics.shape[0])
        return to_ret
    except ZeroDivisionError:
        return float('nan')
