import argparse
import json
import os
import pickle
from typing import List

import spacy
import torch

from data.corpus.corpus import Corpus
from data.corpus.data import load_folds_ids
from data.corpus.single_eq_corpus import SingleEQCorpus
from data.dictionary import Dictionary
from models.math_model import MathModel
from modules.results.results import execute_model_for_test_data, load_ilp_details, complete_ugent_math_results, \
    MathResultsSet, deploy_models_results, complete_alges_math_results, deploy_ilp_coverage, deploy_results_table_header


def load_questions(questions_path):
    """loads the questions id to equation"""
    qid_to_gold = dict()
    with open(questions_path, 'r') as infile:
        all_questions = json.load(infile)

        for question in all_questions:
            qid_to_gold[question['iIndex']] = question['lEquations'][0]

    return qid_to_gold


def load_question_content(questions_path):
    """loads the questions id to equation"""
    nlp = spacy.load('en_core_web_sm')
    qid_to_gold = dict()
    with open(questions_path, 'r') as infile:
        all_questions = json.load(infile)

        for question in all_questions:
            doc = nlp(question['sQuestion'])
            toks = [tok for tok in doc]
            qid_to_gold[question['iIndex']] = toks

    return qid_to_gold


def get_ugent_multirun_results(results: MathResultsSet, curr_fold, curr_model_name, curr_model_path,
                               data_text_id, use_gpu=False):
    assert 'fold' in curr_fold
    fold_nr = curr_fold[-1:]
    print('executing ', curr_model_name, ' test fold: ', fold_nr)
    model_hyperparams_path = os.path.join(param_input_params, curr_model_name,
                                          'fold{}'.format(fold_nr),
                                          '{}_fold{}_params.json'.format(curr_model_name.lower(), fold_nr))
    model_params = json.load(open(model_hyperparams_path, 'r'))
    model_params.update(shared_params)

    curr_model_fold_path = os.path.join(curr_model_path, curr_fold)
    curr_dictionary_path = os.path.join(param_input_models, curr_model_name, curr_fold,
                                        'dictionary_{}_{}.json'.format(data_text_id, curr_fold))
    curr_dataset_path = os.path.join(param_input_models, curr_model_name, curr_fold,
                                     'dataset_{}_{}.pkl'.format(data_text_id, curr_fold))

    if param_execute_models:
        fold_nr_asym_id = '{}_{}_{}'.format(curr_model_name, fold_nr, is_asym)
        if fold_nr_asym_id not in loaded_datasets:
            if not os.path.exists(curr_dataset_path) or param_load_datasets:

                print('loading dataset to', curr_dataset_path)
                curr_dictionary: Dictionary = Dictionary()
                curr_dictionary.load_from_json(curr_dictionary_path)
                loaded_ids = load_folds_ids(train_fold_paths=None,
                                            val_fold_paths=None,
                                            test_fold_paths=model_params['test_folds']['single_eq'])

                single_eq_corpus: SingleEQCorpus = SingleEQCorpus(loaded_ids={'train_ids': [], 'val_ids': [],
                                                                              'test_ids': loaded_ids[
                                                                                  'test_ids']},
                                                                  dictionary=curr_dictionary,
                                                                  params=model_params)
                loaded_datasets[fold_nr_asym_id] = single_eq_corpus

                pickle.dump(single_eq_corpus, open(curr_dataset_path, 'wb'))
                print('DONE loading dataset to ', curr_dataset_path)
            else:
                loaded_datasets[fold_nr_asym_id] = pickle.load(open(curr_dataset_path, 'rb'))

    for curr_model_run in os.listdir(curr_model_fold_path):
        if '.pt' not in curr_model_run:
            continue
        run_nr = curr_model_run[curr_model_run.index('_run') + 4:curr_model_run.index('.pt')]

        output_path_csv = os.path.join(param_output_results, curr_model_name, 'fold{}'.format(fold_nr),
                                       'results_{}_fold{}_run{}.csv'.format(curr_model_name.lower(), fold_nr,
                                                                            run_nr))
        output_dir = os.path.dirname(output_path_csv)
        os.makedirs(output_dir, exist_ok=True)

        if param_execute_models:
            curr_model_state_path = os.path.join(curr_model_fold_path, curr_model_run)

            dataset: Corpus = loaded_datasets[fold_nr_asym_id]
            model: MathModel = MathModel(dictionary=dataset.dictionary, word_embeddings=None,
                                         params=model_params)
            model.load_state_dict(torch.load(curr_model_state_path))
            if use_gpu:
                model.gpu = True
                model = model.to('cuda')
            else:
                model.gpu = False

            execute_model_for_test_data(corpus=[dataset], model=model,
                                        output_csv_results_path=output_path_csv,
                                        questions=questions)

        assert os.path.exists(output_path_csv)

        model_params['run_nr'] = run_nr
        complete_ugent_math_results(path_csv=output_path_csv, questions=questions, loaded_json_part=model_params,
                                    math_results=results)
    return results


def complete_alges_results(results: MathResultsSet, alges_res_path, questions):
    """

    :param results:
    :param curr_fold:
    :return:

    """
    complete_alges_math_results(alges_res_path, questions, results)


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='model experiments')
    parser.register('type', 'bool', str2bool)
    parser.add_argument('--use_gpu', type='bool', default=False, help='Whether to use CUDA or not')
    parser.add_argument('--execute_models', type='bool', default=False, help='Whether to execute the models'
                                                                            ' generating the results from scratch')
    parser.add_argument('--load_datasets', type='bool', default=False, help='Whether to load the datasets from '
                                                                           'the original SingleEQ files or to load the '
                                                                           'pre-loaded serialized files.')

    args = parser.parse_args()
    print('input parameters to the script: ', args)
    param_input_models = 'experiments/models'
    param_input_params = 'experiments/params'
    param_output_results = 'experiments/results'

    param_input_datasets = 'data/single_eq'
    param_questions_path = 'data/single_eq/questions.json'
    param_ilp_path = 'data/single_eq/ILP.out'
    param_top_ilp = 100
    param_accuracy_at = 1
    param_execute_models = args.execute_models
    param_load_datasets = args.load_datasets

    alges_paths = {'ALGES': 'experiments/results/ALGES',
                   'ALGES-asym': 'experiments/results/ALGES-asym'}

    dataset_models = [{'model_name': 'ALGES', 'is_ilp_asym': False},
                      {'model_name': 'B-LSTM', 'is_ilp_asym': False},
                      {'model_name': 'T-LSTM', 'is_ilp_asym': False},
                      {'model_name': 'NT-LSTM', 'is_ilp_asym': False},
                      {'model_name': 'ALGES', 'is_ilp_asym': True},
                      {'model_name': 'B-LSTM', 'is_ilp_asym': True},
                      {'model_name': 'T-LSTM', 'is_ilp_asym': True},
                      {'model_name': 'NT-LSTM', 'is_ilp_asym': True}]

    shared_hyperparams_path = os.path.join(param_input_params, 'shared_hyperparameters.json')
    shared_params = json.load(open(shared_hyperparams_path, 'r'))
    loaded_datasets = dict()

    questions = load_questions(param_questions_path)

    ilp_details = load_ilp_details(questions=questions, ilp_pad=3, ilp_path=param_ilp_path, top_ilp=param_top_ilp)

    all_results = list()
    coverage_ilp_asym = dict()
    coverage_ilp = dict()

    for idx_model, curr_model in enumerate(dataset_models):
        results = MathResultsSet()
        is_asym = curr_model['is_ilp_asym']
        data_text_id = 'orig' if not is_asym else 'asym'
        if is_asym:
            model_id = '{}-{}'.format(curr_model['model_name'], 'asym')
        else:
            model_id = curr_model['model_name']
        curr_model_path = os.path.join(param_input_models, model_id)
        if not 'alges' in model_id.lower():
            # obtain/execute result for our model
            for curr_fold in os.listdir(curr_model_path):
                results = get_ugent_multirun_results(results, curr_fold, model_id, curr_model_path, data_text_id,
                                                     use_gpu=args.use_gpu)
            if is_asym and len(coverage_ilp_asym) == 0:
                coverage_ilp_asym['results'] = results
            elif not is_asym and len(coverage_ilp) == 0:
                coverage_ilp['results'] = results
        else:
            # obtain results for our ALGES (Koncel-Kedziorski et al. 2015) model
            complete_alges_results(results, alges_res_path=alges_paths[model_id], questions=questions)

        all_results.append({'results': results, 'model_name': curr_model['model_name'],
                            'is_ilp_asym': curr_model['is_ilp_asym']})

    ilp_results: List = [res for res in all_results if not res['is_ilp_asym']]
    print()
    print('=' * 60, 'Results', '=' * 60)

    deploy_results_table_header()
    if 'results' in coverage_ilp:
        deploy_ilp_coverage(coverage_ilp['results'], False)

    for idx_model, curr_results in enumerate(ilp_results):
        deploy_models_results(results=curr_results['results'], model_name=curr_results['model_name'])
    print()
    print('=' * 54, 'ILP + Asym results', '=' * 55)
    deploy_results_table_header()
    ilp_asym_results = [res for res in all_results if res['is_ilp_asym']]
    if 'results' in coverage_ilp_asym:
        deploy_ilp_coverage(coverage_ilp_asym['results'], False)
    for idx_model, curr_results in enumerate(ilp_asym_results):
        deploy_ilp_coverage = idx_model == 1
        deploy_models_results(results=curr_results['results'], model_name=curr_results['model_name'])
