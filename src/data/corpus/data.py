""" Module with auxiliary functions for data/corpus loading and iterating """
import copy
import itertools
import re
from math import isclose
from typing import List

import numpy as np
import spacy

from data.corpus.parser import parse
from tree import Node, BinaryTree
from utils.utils import cast_to_number, delete_trailing_zeros, tokenize, normalize

nlp = spacy.load('en_core_web_sm')


class Question:
    orig_question = ''
    question_sentences = list()
    question_tokens = list()
    formula = ''
    formula_trees = None
    numbers = list()
    numbers_node = list()
    solution = 0.0
    id = 0


def load_ids(folds_paths: List):
    ids = []
    for fold in folds_paths:
        with open(fold) as infile:
            for line in infile:
                ids.append(int(line))
    return ids


def load_folds_ids(train_fold_paths: List = None, val_fold_paths: List = None, test_fold_paths: List = None):
    train_ids = None
    if train_fold_paths is not None:
        train_ids = load_ids(train_fold_paths)

    val_ids = None
    if val_fold_paths is not None:
        val_ids = load_ids(val_fold_paths)

    test_ids = load_ids(test_fold_paths)

    return {'train_ids': train_ids, 'val_ids': val_ids, 'test_ids': test_ids}


def load_into_binary_tree(parsed_tree: tuple):
    if parsed_tree[1] == '=':
        root_node: Node = load_into_binary_tree(parsed_tree[2])
        return BinaryTree(root_node)

    if isinstance(parsed_tree[0], tuple):
        left_child: Node = load_into_binary_tree(parsed_tree[0])
    else:
        left_child: Node = Node(value=cast_to_number(delete_trailing_zeros(parsed_tree[0])), ntype='value')

    if isinstance(parsed_tree[2], tuple):
        right_child: Node = load_into_binary_tree(parsed_tree[2])
    else:
        right_child: Node = Node(value=cast_to_number(delete_trailing_zeros(parsed_tree[2])), ntype='value')

    curr_node = Node(value=parsed_tree[1], ntype='operator')
    curr_node.left = left_child
    curr_node.right = right_child
    return curr_node


def get_variations(binary_tree: BinaryTree):
    comm_symbols = {'+', '*'}
    nr_comm_symbols = binary_tree.count_operators(comm_symbols)
    map_comm = list(itertools.product([0, 1], repeat=nr_comm_symbols))
    variations = []
    for idx, comm in enumerate(map_comm):
        binary_tree_c: BinaryTree = copy.deepcopy(binary_tree)
        binary_tree_c.commute_tree(comm, operators={'*', '+'})
        variations.append(binary_tree_c)
    return variations


def get_asymmetries(binary_tree: BinaryTree):
    comm_symbols = {'/', '-'}
    nr_comm_symbols = binary_tree.count_operators(comm_symbols)
    map_comm = list(itertools.product([0, 1], repeat=nr_comm_symbols))
    variations = []
    for idx, comm in enumerate(map_comm):
        binary_tree_c: BinaryTree = copy.deepcopy(binary_tree)
        binary_tree_c.commute_tree(comm, operators={'-', '/'})
        variations.append(binary_tree_c)
    return variations


def get_ilp_trees_variations_last(question: Question, ilp_out_path: str, max_falses_per_equation=-1,
                                  max_equations_per_problem=-1,
                                  include_gold_formula=False, include_asymmetric=True, length_asymmetric=4,
                                  manipulate_equation: bool = True):
    include_in_accuracy = dict()
    lloaded_trees = list()
    cnt_falses = 0

    if question.formula_trees is not None and include_gold_formula:
        for correct_formula in question.formula_trees:
            if correct_formula.is_valid_tree():
                if correct_formula not in include_in_accuracy.keys():
                    correct_formula.enrich_indices(question.numbers_node.copy())
                    enriched_indices = correct_formula.get_indices()
                    if -1 in enriched_indices:
                        continue
                    include_in_accuracy[correct_formula] = False
                    lloaded_trees.append({'label': 1, 'tree': correct_formula})

    potential_warning = False
    print_warning = False
    possible_variations = list()
    with open(ilp_out_path, 'r') as infile:
        for idx, line in enumerate(infile):
            if line.lower()[:4] == 'expr':
                splitted = line.lower().split('|')
                splitted = [itm.strip() for itm in splitted]

                if (manipulate_equation and 'x' in splitted[6]) or \
                        (splitted[6][:2] == 'x=' or splitted[6][-2:] == '=x'):

                    label = int(splitted[0][-1:])
                    formula = splitted[6]
                    norm = normalize(formula)

                    sp_norm_e = norm.split('=')
                    assert sp_norm_e[0] == 'x'
                    if sp_norm_e[1][0] == '-':
                        p = re.compile("[\+\-\*\/]")

                        is_there_second_adyacent_minus = False
                        for idx, m in enumerate(p.finditer(sp_norm_e[1][1:])):
                            if idx == 0:
                                if sp_norm_e[1][1:][m.start():m.end()] == '-':
                                    is_there_second_adyacent_minus = True

                        is_there_second_adyacent_times = False
                        for idx, m in enumerate(p.finditer(sp_norm_e[1][1:])):
                            if idx == 0:
                                if sp_norm_e[1][1:][m.start():m.end()] == '*':
                                    is_there_second_adyacent_times = True

                        if is_there_second_adyacent_minus:
                            if '+' in sp_norm_e[1]:
                                index_plus = sp_norm_e[1].index('+')

                                if index_plus > 0 and ('(' not in sp_norm_e[1][:index_plus]):
                                    new_eq = sp_norm_e[1][index_plus + 1:] + sp_norm_e[1][:index_plus]
                                    new_eq = 'x=' + new_eq.strip()
                                    norm = new_eq

                        if is_there_second_adyacent_times:
                            if '+' in sp_norm_e[1]:
                                index_plus = sp_norm_e[1].index('+')
                                if index_plus > 0 and ('(' not in sp_norm_e[1][:index_plus]):
                                    new_eq = sp_norm_e[1][index_plus + 1:] + sp_norm_e[1][:index_plus]
                                    new_eq = 'x=' + new_eq.strip()
                                    norm = new_eq

                    toks = tokenize(norm)

                    try:
                        tree = parse(toks)
                        binary_tree: BinaryTree = load_into_binary_tree(tree)
                    except:
                        continue

                    binary_tree.rearrange_tree()

                    binary_tree.enrich_indices(question.numbers_node.copy())

                    enriched_indices = binary_tree.get_indices()
                    if -1 in enriched_indices:
                        if label == 1:
                            potential_warning = True
                        continue
                    else:
                        if label == 1:
                            print_warning = False

                    binary_tree_variations = get_variations(binary_tree)
                    binary_tree_variations = [{'tree': curr_tr, 'label': label} for curr_tr in binary_tree_variations]
                    possible_variations.extend(binary_tree_variations)

                    for ilp_generated_binary_tree in [binary_tree]:
                        # for binary_tree_variation in [binary_tree]:
                        if not ilp_generated_binary_tree.is_valid_tree():
                            print('NOT A VALID TREE!!: ', ilp_generated_binary_tree)
                            continue

                        tree_eval = ilp_generated_binary_tree.evaluate()
                        gold_sol = question.solution
                        if label == 1:
                            assert isclose(tree_eval, gold_sol, rel_tol=1e-5, abs_tol=0.0)
                        if label == 0:
                            assert not isclose(tree_eval, gold_sol, rel_tol=1e-5, abs_tol=0.0)

                        if ilp_generated_binary_tree not in include_in_accuracy.keys():
                            if 0 < max_falses_per_equation <= cnt_falses and label == 0:
                                continue
                            if label == 0:
                                cnt_falses += 1
                            include_in_accuracy[ilp_generated_binary_tree] = True
                            lloaded_trees.append({'label': label, 'tree': ilp_generated_binary_tree})
                        else:
                            include_in_accuracy[ilp_generated_binary_tree] = True

                    if include_asymmetric:
                        curr_tree: BinaryTree = binary_tree
                        if len(curr_tree.get_values()) <= length_asymmetric:
                            asymms_trees = get_asymmetries(curr_tree)
                            for curr_asymm_tree in asymms_trees:
                                if curr_asymm_tree not in include_in_accuracy:
                                    include_in_accuracy[curr_asymm_tree] = True
                                    res = curr_asymm_tree.evaluate()
                                    if isclose(res, question.solution, rel_tol=1e-5, abs_tol=0.0):
                                        curr_label = 1
                                    else:
                                        curr_label = 0
                                    lloaded_trees.append({'label': curr_label, 'tree': curr_asymm_tree})

    for binary_tree_variation_i in possible_variations:
        # for binary_tree_variation in [binary_tree]:
        binary_tree_variation = binary_tree_variation_i['tree']
        label = binary_tree_variation_i['label']
        if not binary_tree_variation.is_valid_tree():
            print('NOT A VALID TREE (variation step)!!: ', binary_tree_variation)
            continue

        tree_eval = binary_tree_variation.evaluate()
        gold_sol = question.solution
        if label == 1:
            assert isclose(tree_eval, gold_sol, rel_tol=1e-5, abs_tol=0.0)
        if label == 0:
            assert not isclose(tree_eval, gold_sol, rel_tol=1e-5, abs_tol=0.0)

        if binary_tree_variation not in include_in_accuracy.keys():
            if cnt_falses >= max_falses_per_equation > 0 == label:
                continue
            if label == 0:
                cnt_falses += 1
            include_in_accuracy[binary_tree_variation] = True
            lloaded_trees.append({'label': label, 'tree': binary_tree_variation})
        else:
            include_in_accuracy[binary_tree_variation] = True

    if potential_warning and print_warning:
        print('WARNING, omitting the following equation because can not link the indices: ',
              '\n  question: ', question.orig_question, '\n  question.id: ', question.id,
              '\n gold formula: ', question.formula, '\n  numbers extracted: ',
              [nd.value for nd in question.numbers_node])
    # adds 'include_in_accuracy' to see if it has to be included in accuracy calculation , in case of coming
    # from the equation in the original question file, should not be included, only if it is also present in ILP
    lloaded_trees = [{'label': ltr['label'], 'tree': ltr['tree'],
                      'include_in_accuracy': include_in_accuracy[ltr['tree']]} for ltr in lloaded_trees]
    if len(lloaded_trees) == 0:
        return [{'label': 0, 'tree': None, 'include_in_accuracy': True}]
    else:
        if max_equations_per_problem > 0:
            return lloaded_trees[:max_equations_per_problem]
        else:
            return lloaded_trees


class DataIterator:
    def __init__(self, content: list, targets: list, train=False, random_seed=1234):
        batch_size = 1
        self.content = content
        self.targets = targets
        self.batch_size = batch_size
        self.train = train

        self.batching_seed = np.random.RandomState(random_seed)
        self.n_batches = int(np.ceil(len(content) / self.batch_size))

    def __iter__(self):
        inds = self.batching_seed.permutation(np.arange(len(self.content)))

        self.inds = inds

        for i in range(self.n_batches):
            batch_inds = inds[i * self.batch_size: (i + 1) * self.batch_size]
            batch_size = len(batch_inds)

            batch_content = [self.content[i] for i in batch_inds]
            batch_targets = [self.targets[i] for i in batch_inds]

            len_questions = []

            # loop to calculate the lengths
            for element in batch_content:
                len_questions.append(element['tokens'].shape[0])

            batch_questions_len = np.asarray(len_questions)
            max_len_questions = batch_questions_len.max()
            batch_questions = np.zeros((batch_size, max_len_questions), dtype=np.int32)

            batch_targets_ans = np.zeros((batch_size,), dtype=np.int32)
            batch_indexes = np.zeros((batch_size,), dtype=np.int32)

            batch_trees = list()
            include_in_accuracy = list()

            for idx_batch, element in enumerate(batch_content):
                batch_questions[idx_batch, :element['tokens'].shape[0]] = element['tokens']
                batch_targets_ans[idx_batch] = np.asarray(batch_targets[idx_batch], dtype=np.int32)
                batch_indexes[idx_batch] = element['id']
                batch_trees.append(element['tree'])
                if 'include_in_accuracy' in element.keys():
                    include_in_accuracy.append(element['include_in_accuracy'])
                else:
                    include_in_accuracy.append(True)

            # sort instances according to decreasing number of words for using PackedSequence with data batch
            perm_idx = batch_questions_len.argsort(axis=0)[::-1]  # sort backwards
            batch_questions = batch_questions[perm_idx]
            batch_questions_len = batch_questions_len[perm_idx]
            batch_targets_ans = batch_targets_ans[perm_idx]
            batch_indexes = batch_indexes[perm_idx]
            batch_trees = [batch_trees[i] for i in perm_idx]
            include_in_accuracy = [include_in_accuracy[i] for i in perm_idx]

            yield batch_questions[0], batch_questions_len[0], batch_targets_ans[0], batch_indexes[0], batch_trees[0], \
                  include_in_accuracy[0]
