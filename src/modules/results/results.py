# module with functions related to get the data results (accuracies) both from ALGES as from our classifiers
import csv
import os
import re
from collections import OrderedDict

import numpy as np
from torch import nn

from data.corpus.data import load_into_binary_tree, DataIterator
from data.corpus.parser import parse
from models.metrics import get_results_info, get_results_ilp_naive_test
from models.train import execute_validation
from tree import BinaryTree
from utils.utils import normalize, tokenize


def load_ilp_details(questions, ilp_pad, ilp_path, top_ilp=100):
    ilp_details = dict()
    for curr_question_id in questions:
        to_format = 'q{:0' + str(ilp_pad) + 'd}.txt.out'
        file_name = to_format.format(curr_question_id)
        ilp_file_path = os.path.join(ilp_path, file_name)

        cnt = 0
        ilp_details[curr_question_id] = {'ilp_contains_correct': False}
        with open(ilp_file_path) as infile:
            for curr_line in infile:

                if curr_line.lower()[:4] == 'expr':
                    cnt += 1
                    splitted = curr_line.lower().split('|')
                    splitted = [itm.strip() for itm in splitted]
                    is_correct = int(splitted[0][-1])
                    if is_correct == 1:
                        ilp_details[curr_question_id]['ilp_contains_correct'] = True

                if cnt >= top_ilp:
                    break

    return ilp_details


class MathResultsSet:
    def __init__(self):
        self.all_accuracy_runs = dict()
        self.all_accuracy_runs_naive = dict()
        self.all_accuracy_runs_ceiling = dict()

        self.multi_ops_accuracy_runs = dict()
        self.multi_ops_accuracy_runs_ceiling = dict()
        self.multi_ops_accuracy_runs_naive = dict()

        self.single_ops_accuracy_runs = dict()
        self.single_ops_accuracy_runs_ceiling = dict()
        self.single_ops_accuracy_runs_naive = dict()

        self.asymmetric_accuracy_eq1op_runs = dict()
        self.asymmetric_accuracy_eq1op_runs_ceiling = dict()
        self.asymmetric_accuracy_eq1op_runs_naive = dict()

        self.asymmetric_accuracy_big2op_runs = dict()
        self.asymmetric_accuracy_big2op_runs_ceiling = dict()
        self.asymmetric_accuracy_big2op_runs_naive = dict()

        self.symmetric_accuracy_eq1op_runs = dict()
        self.symmetric_accuracy_eq1op_runs_ceiling = dict()
        self.symmetric_accuracy_eq1op_runs_naive = dict()

        self.symmetric_accuracy_big2op_runs = dict()
        self.symmetric_accuracy_big2op_runs_ceiling = dict()
        self.symmetric_accuracy_big2op_runs_naive = dict()

        self.accuracy_single_symm_ngrams_folds = dict()


def replace_if_number(curr_tok):
    nr_regex = r"[-]{0,1}(\d*\.\d+|\d+,\d+|\d+)"
    p = re.compile(nr_regex)
    matches = p.match(curr_tok)
    if matches is not None:
        return 'n'
    else:
        return curr_tok


def add_alges_result(math_results: MathResultsSet, line: str, question_id: int, questions: dict, curr_run_nr,
                     asymmetric_ops: list):
    orig_equation = questions[question_id]
    try:
        norm = normalize(orig_equation)
        toks = tokenize(norm)
        tree = parse(toks)
        binary_tree: BinaryTree = load_into_binary_tree(tree)
        binary_tree.rearrange_tree()
        normalized_equation = str(binary_tree)
    except:
        normalized_equation = orig_equation

    splitted_by_op = re.compile("[*/+\-]").split(normalized_equation)
    nr_ops = len(splitted_by_op) - 1

    is_correct = (1 if 'CORRECT' == line.strip() else 0)

    math_results.all_accuracy_runs[curr_run_nr].append(is_correct)

    if nr_ops > 1:
        math_results.multi_ops_accuracy_runs[curr_run_nr].append(is_correct)

    if nr_ops == 1:
        math_results.single_ops_accuracy_runs[curr_run_nr].append(is_correct)

    if nr_ops == 1 and any([aop in normalized_equation for aop in asymmetric_ops]):
        math_results.asymmetric_accuracy_eq1op_runs[curr_run_nr].append(is_correct)

    if nr_ops > 1 and any([aop in normalized_equation for aop in asymmetric_ops]):
        math_results.asymmetric_accuracy_big2op_runs[curr_run_nr].append(is_correct)

    if nr_ops == 1 and not any([aop in normalized_equation for aop in asymmetric_ops]):
        math_results.symmetric_accuracy_eq1op_runs[curr_run_nr].append(is_correct)

    if nr_ops > 1 and not any([aop in normalized_equation for aop in asymmetric_ops]):
        math_results.symmetric_accuracy_big2op_runs[curr_run_nr].append(is_correct)


def complete_alges_math_results(alges_res_path: csv, questions: dict, math_results: MathResultsSet):
    questions_to_complete = set(questions.keys())

    # ALGES has a single run only
    run_nr = '0'
    math_results.all_accuracy_runs[run_nr] = list()

    math_results.multi_ops_accuracy_runs[run_nr] = list()

    math_results.single_ops_accuracy_runs[run_nr] = list()

    math_results.asymmetric_accuracy_eq1op_runs[run_nr] = list()

    math_results.asymmetric_accuracy_big2op_runs[run_nr] = list()

    math_results.symmetric_accuracy_eq1op_runs[run_nr] = list()

    math_results.symmetric_accuracy_big2op_runs[run_nr] = list()

    asymmetric_ops = ['-', '/']

    for file in os.listdir(alges_res_path):
        infile_path = os.path.join(alges_res_path, file)
        print('parsing ALGES predictions from', infile_path)
        nr_file_scanned = 0
        prev_line = ''
        with open(infile_path, 'r') as infile:
            for line in infile:
                if 'CORRECT' == line.strip() or 'INCORRECT' == line.strip():
                    question_id = re.findall(r'\d+', prev_line)[0]
                    question_id = int(question_id)
                    assert question_id in questions_to_complete
                    questions_to_complete.remove(question_id)
                    nr_file_scanned += 1

                    add_alges_result(math_results=math_results, line=line, question_id=question_id,
                                     questions=questions, curr_run_nr=run_nr, asymmetric_ops=asymmetric_ops)

                prev_line = line

    if len(questions_to_complete) > 0:
        for curr_q_to_complete in questions_to_complete:
            add_alges_result(math_results=math_results, line='INCORRECT', question_id=curr_q_to_complete,
                             questions=questions, curr_run_nr=run_nr, asymmetric_ops=asymmetric_ops)

    return math_results


def complete_ugent_math_results(path_csv: str, questions: dict, loaded_json_part, math_results: MathResultsSet):
    if loaded_json_part['run_nr'] not in math_results.all_accuracy_runs.keys():
        math_results.all_accuracy_runs[loaded_json_part['run_nr']] = list()
        math_results.all_accuracy_runs_ceiling[loaded_json_part['run_nr']] = list()
        math_results.all_accuracy_runs_naive[loaded_json_part['run_nr']] = list()

    if loaded_json_part['run_nr'] not in math_results.multi_ops_accuracy_runs.keys():
        math_results.multi_ops_accuracy_runs[loaded_json_part['run_nr']] = list()
        math_results.multi_ops_accuracy_runs_ceiling[loaded_json_part['run_nr']] = list()
        math_results.multi_ops_accuracy_runs_naive[loaded_json_part['run_nr']] = list()

    if loaded_json_part['run_nr'] not in math_results.single_ops_accuracy_runs.keys():
        math_results.single_ops_accuracy_runs[loaded_json_part['run_nr']] = list()
        math_results.single_ops_accuracy_runs_ceiling[loaded_json_part['run_nr']] = list()
        math_results.single_ops_accuracy_runs_naive[loaded_json_part['run_nr']] = list()

    if loaded_json_part['run_nr'] not in math_results.asymmetric_accuracy_eq1op_runs.keys():
        math_results.asymmetric_accuracy_eq1op_runs[loaded_json_part['run_nr']] = list()
        math_results.asymmetric_accuracy_eq1op_runs_ceiling[loaded_json_part['run_nr']] = list()
        math_results.asymmetric_accuracy_eq1op_runs_naive[loaded_json_part['run_nr']] = list()

    if loaded_json_part['run_nr'] not in math_results.asymmetric_accuracy_big2op_runs.keys():
        math_results.asymmetric_accuracy_big2op_runs[loaded_json_part['run_nr']] = list()
        math_results.asymmetric_accuracy_big2op_runs_ceiling[loaded_json_part['run_nr']] = list()
        math_results.asymmetric_accuracy_big2op_runs_naive[loaded_json_part['run_nr']] = list()

    if loaded_json_part['run_nr'] not in math_results.symmetric_accuracy_eq1op_runs.keys():
        math_results.symmetric_accuracy_eq1op_runs[loaded_json_part['run_nr']] = list()
        math_results.symmetric_accuracy_eq1op_runs_ceiling[loaded_json_part['run_nr']] = list()
        math_results.symmetric_accuracy_eq1op_runs_naive[loaded_json_part['run_nr']] = list()

    if loaded_json_part['run_nr'] not in math_results.symmetric_accuracy_big2op_runs.keys():
        math_results.symmetric_accuracy_big2op_runs[loaded_json_part['run_nr']] = list()
        math_results.symmetric_accuracy_big2op_runs_ceiling[loaded_json_part['run_nr']] = list()
        math_results.symmetric_accuracy_big2op_runs_naive[loaded_json_part['run_nr']] = list()

    asymmetric_ops = ['-', '/']

    id_to_correct_rnn = dict()
    if os.path.isfile(path_csv):
        with open(path_csv) as csv_file:
            csv_reader = csv.reader(csv_file)
            for idx, row in enumerate(csv_reader):
                if idx == 0:
                    continue
                question_id = int(row[0])
                orig_equation = questions[question_id]
                try:
                    norm = normalize(orig_equation)
                    toks = tokenize(norm)
                    tree = parse(toks)
                    binary_tree: BinaryTree = load_into_binary_tree(tree)
                    binary_tree.rearrange_tree()
                    normalized_equation = str(binary_tree)

                except Exception as e:
                    normalized_equation = orig_equation

                splitted_by_op = re.compile("[*/+\-]").split(normalized_equation)
                nr_ops = len(splitted_by_op) - 1

                is_naive_ilp_correct = int(row[6])

                is_correct = int(row[5])
                is_correct = 0 if is_correct <= 0 else is_correct

                # 1 if ILP has potential solution, 0 otherwise
                is_potentially_solvable = int(row[7])

                if question_id not in id_to_correct_rnn:
                    id_to_correct_rnn[question_id] = [is_correct]
                else:
                    id_to_correct_rnn[question_id].append(is_correct)

                math_results.all_accuracy_runs[loaded_json_part['run_nr']].append(is_correct)

                math_results.all_accuracy_runs_ceiling[loaded_json_part['run_nr']].append(is_potentially_solvable)
                math_results.all_accuracy_runs_naive[loaded_json_part['run_nr']].append(is_naive_ilp_correct)

                if nr_ops > 1:
                    math_results.multi_ops_accuracy_runs[loaded_json_part['run_nr']].append(is_correct)
                    math_results.multi_ops_accuracy_runs_ceiling[loaded_json_part['run_nr']].append(
                        is_potentially_solvable)
                    math_results.multi_ops_accuracy_runs_naive[loaded_json_part['run_nr']].append(
                        is_naive_ilp_correct)

                if nr_ops == 1:
                    math_results.single_ops_accuracy_runs[loaded_json_part['run_nr']].append(is_correct)

                    math_results.single_ops_accuracy_runs_ceiling[loaded_json_part['run_nr']].append(
                        is_potentially_solvable)
                    math_results.single_ops_accuracy_runs_naive[loaded_json_part['run_nr']].append(
                        is_naive_ilp_correct)

                if nr_ops == 1 and any(
                        [aop in normalized_equation for aop in asymmetric_ops]):
                    math_results.asymmetric_accuracy_eq1op_runs[loaded_json_part['run_nr']].append(is_correct)

                    math_results.asymmetric_accuracy_eq1op_runs_ceiling[loaded_json_part['run_nr']].append(
                        is_potentially_solvable)
                    math_results.asymmetric_accuracy_eq1op_runs_naive[loaded_json_part['run_nr']].append(
                        is_naive_ilp_correct)

                if nr_ops > 1 and any(
                        [aop in normalized_equation for aop in asymmetric_ops]):
                    math_results.asymmetric_accuracy_big2op_runs[loaded_json_part['run_nr']].append(
                        is_correct)

                    math_results.asymmetric_accuracy_big2op_runs_ceiling[loaded_json_part['run_nr']].append(
                        is_potentially_solvable)
                    math_results.asymmetric_accuracy_big2op_runs_naive[loaded_json_part['run_nr']].append(
                        is_naive_ilp_correct)

                if nr_ops == 1 and not any(
                        [aop in normalized_equation for aop in asymmetric_ops]):
                    if is_correct == 0:
                        pass
                    math_results.symmetric_accuracy_eq1op_runs[loaded_json_part['run_nr']].append(is_correct)

                    math_results.symmetric_accuracy_eq1op_runs_ceiling[loaded_json_part['run_nr']].append(
                        is_potentially_solvable)
                    math_results.symmetric_accuracy_eq1op_runs_naive[loaded_json_part['run_nr']].append(
                        is_naive_ilp_correct)

                if nr_ops > 1 and not any(
                        [aop in normalized_equation for aop in asymmetric_ops]):
                    math_results.symmetric_accuracy_big2op_runs[loaded_json_part['run_nr']].append(is_correct)

                    math_results.symmetric_accuracy_big2op_runs_ceiling[loaded_json_part['run_nr']].append(
                        is_potentially_solvable)
                    math_results.symmetric_accuracy_big2op_runs_naive[loaded_json_part['run_nr']].append(
                        is_naive_ilp_correct)


def execute_model_for_test_data(corpus, model, output_csv_results_path, questions):
    test_content = [item for sublist in [corp.corpus_test['content'] for corp in corpus] for item in sublist]
    test_targets = [item for sublist in [corp.corpus_test['targets'] for corp in corpus] for item in sublist]

    data_iterator_test = DataIterator(content=test_content, targets=test_targets, train=False)

    loss_fn = nn.BCEWithLogitsLoss(reduction='none', pos_weight=None)
    model.eval()
    dictionary = corpus[len(corpus) - 1].dictionary

    res = execute_validation(model, data_iterator_test, loss_fn, 0.5, get_details=True, dictionary=dictionary,
                             calculate_nr_order_statistic=True)

    test_gr_accuracy, len_list_res, dict_res_top1, dict_res_single_top1 = get_results_info(res['val_gr_metrics'])
    print('test accuracy: ', test_gr_accuracy)

    dict_res_naive_ilp = get_results_ilp_naive_test(corpus[0])

    details: dict = res['details']
    sort_keys = sorted(details.keys())

    print('writing csv into: ', output_csv_results_path)

    with open(output_csv_results_path, 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerow(
            ['id',  # 0
             'question',  # 1
             'gold',  # 2
             'nr operators',  # 3
             'predicted',  # 4
             'correct',  # 5
             'naive_ilp_correct',  # 6
             'solvable']  # 7
        )
        for key in sort_keys:
            is_naive_ilp_correct = dict_res_naive_ilp[key][0]
            details_key = details[key]
            splitted_by_op = re.compile("[*/+\-]").split(questions[key])
            row = [key,  # 0
                   details_key['decoded'],  # 1
                   questions[key],  # 2
                   len(splitted_by_op),  # 3
                   details_key['best_predicted'],  # 4
                   details_key['correct'],  # 5
                   is_naive_ilp_correct,  # 6
                   0 if details_key['gold_tree_nr_vars'] == -1 else 1]  # 7

            writer.writerow(row)


def deploy_ilp_coverage(results: MathResultsSet, is_ilp_asym):
    all_accuracy_accs_folds_ceiling = [(sum(curr_acc_fold) / len(curr_acc_fold)) * 100 for curr_acc_fold in
                                       results.all_accuracy_runs_ceiling.values()]

    res_all_accuracy_ceiling = sum(all_accuracy_accs_folds_ceiling) / len(all_accuracy_accs_folds_ceiling)

    all_accuracy_accs_folds_naive = [(sum(curr_acc_fold) / len(curr_acc_fold)) * 100 for curr_acc_fold in
                                     results.all_accuracy_runs_naive.values()]

    res_all_accuracy_naive = sum(all_accuracy_accs_folds_naive) / len(all_accuracy_accs_folds_naive)

    single_ops_accs_folds_ceiling = [(sum(curr_acc_fold) / len(curr_acc_fold)) * 100 for curr_acc_fold in
                                     results.single_ops_accuracy_runs_ceiling.values()]

    res_single_ops_ceiling = sum(single_ops_accs_folds_ceiling) / len(single_ops_accs_folds_ceiling)

    single_ops_accs_folds_naive = [(sum(curr_acc_fold) / len(curr_acc_fold)) * 100 for curr_acc_fold in
                                   results.single_ops_accuracy_runs_naive.values()]

    res_single_ops_naive = sum(single_ops_accs_folds_naive) / len(single_ops_accs_folds_naive)

    multi_ops_accs_folds_ceiling = [(sum(curr_acc_fold) / len(curr_acc_fold)) * 100 for curr_acc_fold in
                                    results.multi_ops_accuracy_runs_ceiling.values()]

    res_multi_ops_ceiling = sum(multi_ops_accs_folds_ceiling) / len(multi_ops_accs_folds_ceiling)

    multi_ops_accs_folds_naive = [(sum(curr_acc_fold) / len(curr_acc_fold)) * 100 for curr_acc_fold in
                                  results.multi_ops_accuracy_runs_naive.values()]

    res_multi_ops_naive = sum(multi_ops_accs_folds_naive) / len(multi_ops_accs_folds_naive)

    asym_eq1op_accs_folds_ceiling = [(sum(curr_acc_fold) / len(curr_acc_fold)) * 100 for curr_acc_fold in
                                     results.asymmetric_accuracy_eq1op_runs_ceiling.values()]

    res_asymmetric_eq1op_ceiling = sum(asym_eq1op_accs_folds_ceiling) / len(asym_eq1op_accs_folds_ceiling)

    asym_eq1op_accs_folds_naive = [(sum(curr_acc_fold) / len(curr_acc_fold)) * 100 for curr_acc_fold in
                                   results.asymmetric_accuracy_eq1op_runs_naive.values()]

    res_asymmetric_eq1op_naive = sum(asym_eq1op_accs_folds_naive) / len(asym_eq1op_accs_folds_naive)

    asym_b2op_accs_folds_ceiling = [(sum(curr_acc_fold) / len(curr_acc_fold)) * 100 for curr_acc_fold in
                                    results.asymmetric_accuracy_big2op_runs_ceiling.values()]

    res_asymmetric_big2op_ceiling = sum(asym_b2op_accs_folds_ceiling) / len(asym_b2op_accs_folds_ceiling)
    asym_b2op_accs_folds_naive = [(sum(curr_acc_fold) / len(curr_acc_fold)) * 100 for curr_acc_fold in
                                  results.asymmetric_accuracy_big2op_runs_naive.values()]

    res_asymmetric_big2op_naive = sum(asym_b2op_accs_folds_naive) / len(asym_b2op_accs_folds_naive)

    sym_eq1op_accs_folds_ceiling = [(sum(curr_acc_fold) / len(curr_acc_fold)) * 100 for curr_acc_fold in
                                    results.symmetric_accuracy_eq1op_runs_ceiling.values()]

    res_symmetric_eq1op_ceiling = sum(sym_eq1op_accs_folds_ceiling) / len(sym_eq1op_accs_folds_ceiling)
    sym_eq1op_accs_folds_naive = [(sum(curr_acc_fold) / len(curr_acc_fold)) * 100 for curr_acc_fold in
                                  results.symmetric_accuracy_eq1op_runs_naive.values()]

    res_symmetric_eq1op_naive = sum(sym_eq1op_accs_folds_naive) / len(sym_eq1op_accs_folds_naive)
    sym_b2op_accs_folds_ceiling = [(sum(curr_acc_fold) / len(curr_acc_fold)) * 100 for curr_acc_fold in
                                   results.symmetric_accuracy_big2op_runs_ceiling.values()]

    res_symmetric_big2op_ceiling = sum(sym_b2op_accs_folds_ceiling) / len(sym_b2op_accs_folds_ceiling)
    sym_b2op_accs_folds_naive = [(sum(curr_acc_fold) / len(curr_acc_fold)) * 100 for curr_acc_fold in
                                 results.symmetric_accuracy_big2op_runs_naive.values()]

    res_symmetric_big2op_naive = sum(sym_b2op_accs_folds_naive) / len(sym_b2op_accs_folds_naive)

    ilp_coverage_name = 'ILP Coverage'
    print('{:<15}{:.2f}        \t{:.2f}        \t{:.2f}        \t{:.2f}        \t{:.2f}        \t'
          '{:.2f}        \t{:.2f}'
          .format(ilp_coverage_name, res_all_accuracy_ceiling, res_single_ops_ceiling, res_multi_ops_ceiling,
                  res_symmetric_eq1op_ceiling, res_symmetric_big2op_ceiling, res_asymmetric_eq1op_ceiling,
                  res_asymmetric_big2op_ceiling))
    print('{:<15}{:.2f}        \t{:.2f}        \t{:.2f}        \t{:.2f}        \t{:.2f}        \t'
          '{:.2f}        \t{:.2f}'
          .format('ILP Naive', res_all_accuracy_naive, res_single_ops_naive,
                  res_multi_ops_naive, res_symmetric_eq1op_naive, res_symmetric_big2op_naive,
                  res_asymmetric_eq1op_naive, res_asymmetric_big2op_naive))


def deploy_results_table_header():
    print('{:<15}{:<13}\t{:<13}\t{:<13}\t{:<13}\t{:<13}\t'
          '{:<13}\t{:<13}'
          .format('Model', 'Full', 'Single', 'Multi', 'Single_sym', 'Multi_sym', 'Single_asym', 'Multi_asym'))
    print('-' * 129)


def deploy_models_results(results: MathResultsSet, model_name=None):
    results.all_accuracy_runs = OrderedDict(sorted(results.all_accuracy_runs.items()))
    all_accuracy_accs_folds = [(sum(curr_acc_fold) / len(curr_acc_fold)) * 100 for curr_acc_fold in
                               results.all_accuracy_runs.values()]

    stdev = np.std(all_accuracy_accs_folds)

    res_all_accuracy = sum(all_accuracy_accs_folds) / len(all_accuracy_accs_folds)
    stdev_all_accuracy = float(stdev)

    results.single_ops_accuracy_runs = OrderedDict(sorted(results.single_ops_accuracy_runs.items()))

    single_op_accs_folds = [(sum(curr_acc_fold) / len(curr_acc_fold)) * 100 for curr_acc_fold in
                            results.single_ops_accuracy_runs.values()]
    stdev = np.std(single_op_accs_folds)

    res_single_ops = sum(single_op_accs_folds) / len(single_op_accs_folds)
    stdev_single_ops = float(stdev)

    results.multi_ops_accuracy_runs = OrderedDict(sorted(results.multi_ops_accuracy_runs.items()))

    multi_ops_accs_folds = [(sum(curr_acc_fold) / len(curr_acc_fold)) * 100 for curr_acc_fold in
                            results.multi_ops_accuracy_runs.values()]

    stdev = np.std(multi_ops_accs_folds)

    res_multi_ops = sum(multi_ops_accs_folds) / len(multi_ops_accs_folds)
    stdev_multi_ops = float(stdev)

    results.asymmetric_accuracy_eq1op_runs = OrderedDict(sorted(results.asymmetric_accuracy_eq1op_runs.items()))

    asym_eq1op_accs_folds = [(sum(curr_acc_fold) / len(curr_acc_fold)) * 100 for curr_acc_fold in
                             results.asymmetric_accuracy_eq1op_runs.values()]
    stdev = np.std(asym_eq1op_accs_folds)

    res_asymmetric_eq1op = sum(asym_eq1op_accs_folds) / len(asym_eq1op_accs_folds)
    stdev_asymmetric_eq1op = float(stdev)

    results.asymmetric_accuracy_big2op_runs = OrderedDict(sorted(results.asymmetric_accuracy_big2op_runs.items()))

    asym_b2op_accs_folds = [(sum(curr_acc_fold) / len(curr_acc_fold)) * 100 for curr_acc_fold in
                            results.asymmetric_accuracy_big2op_runs.values()]
    stdev = np.std(asym_b2op_accs_folds)

    res_asymmetric_big2op = sum(asym_b2op_accs_folds) / len(asym_b2op_accs_folds)
    stdev_asymmetric_big2op = float(stdev)

    results.symmetric_accuracy_eq1op_runs = OrderedDict(sorted(results.symmetric_accuracy_eq1op_runs.items()))

    sym_eq1op_accs_folds = [(sum(curr_acc_fold) / len(curr_acc_fold)) * 100 for curr_acc_fold in
                            results.symmetric_accuracy_eq1op_runs.values()]
    stdev = np.std(sym_eq1op_accs_folds)

    res_symmetric_eq1op = sum(sym_eq1op_accs_folds) / len(sym_eq1op_accs_folds)
    stdev_symmetric_eq1op = float(stdev)

    results.symmetric_accuracy_big2op_runs = OrderedDict(sorted(results.symmetric_accuracy_big2op_runs.items()))

    sym_b2op_accs_folds = [(sum(curr_acc_fold) / len(curr_acc_fold)) * 100 for curr_acc_fold in
                           results.symmetric_accuracy_big2op_runs.values()]
    stdev = np.std(sym_b2op_accs_folds)

    res_symmetric_big2op = sum(sym_b2op_accs_folds) / len(sym_b2op_accs_folds)
    stdev_symmetric_big2op = float(stdev)

    # if multiple runs are present, then prints with standard deviation, if not without
    if len(all_accuracy_accs_folds) > 1:
        print('{:<15}{:.2f}(+-{:.2f})\t{:.2f}(+-{:.2f})\t{:.2f}(+-{:.2f})\t{:.2f}(+-{:.2f})\t{:.2f}(+-{:.2f})\t'
              '{:.2f}(+-{:.2f})\t{:.2f}(+-{:.2f})'
              .format(model_name, res_all_accuracy, stdev_all_accuracy, res_single_ops, stdev_single_ops,
                      res_multi_ops, stdev_multi_ops,
                      res_symmetric_eq1op, stdev_symmetric_eq1op, res_symmetric_big2op, stdev_symmetric_big2op,
                      res_asymmetric_eq1op, stdev_asymmetric_eq1op,
                      res_asymmetric_big2op, stdev_asymmetric_big2op))
    else:
        print('{:<15}{:.2f}        \t{:.2f}        \t{:.2f}        \t{:.2f}        \t{:.2f}        \t{:.2f}        '
              '\t{:.2f}'.format(model_name, res_all_accuracy, res_single_ops, res_multi_ops, res_symmetric_eq1op,
                                res_symmetric_big2op, res_asymmetric_eq1op, res_asymmetric_big2op))
