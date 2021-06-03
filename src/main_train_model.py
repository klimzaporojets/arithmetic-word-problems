import argparse
import os
import pickle
import time
from os.path import join

import git
import numpy as np
import torch
from tensorboard_logger import configure

from models.math_model import MathModel
from models.train import train_module
from data.corpus.data import DataIterator, load_folds_ids
from data.corpus.single_eq_corpus import SingleEQCorpus
from utils.arguments_parser import add_parser_default_values
from utils.utils import load_word_embeddings

pat_to_id = {'l1d': 'hdim_lstm1', 'lr': 'learning_rate', 'dpt': 'dropout', 'bs': 'batch_size', 'sg': 'scheduler_gamma',
             'f': 'fold', 'lr_v': 'variable_rate', 'dtr': 'hdim_tree', 'hatt': 'hdim_attention',
             'sss': 'scheduler_step_size'}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    params = add_parser_default_values(parser)

    # sets the seed
    torch.manual_seed(params['shuffle_seed'])

    os.makedirs(params['log_dir'], exist_ok=True)

    params['gpu'] = torch.cuda.is_available()

    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    params['git_commit'] = sha

    print('====================================================================================')
    print('Executing with the following parameters: ', str(params))

    # tensorboard logging configuration
    configure(params['log_dir'])

    corpora: list = list()

    if not params['load_corpora']:
        loaded_ids = load_folds_ids(params['train_folds']['single_eq'], params['val_folds']['single_eq'],
                                    params['test_folds']['single_eq'])
        single_eq_corpus: SingleEQCorpus = SingleEQCorpus(loaded_ids=loaded_ids, params=params)
        dictionary = single_eq_corpus.dictionary
        corpora.append(single_eq_corpus)
        with open(os.path.join(params['log_dir'], 'corpus.pkl'), 'wb') as fID:
            print('saving corpora to ', os.path.join(params['log_dir'], 'corpus.pkl'))
            pickle.dump(corpora, fID)
            print('end saving corpora')
    else:
        file_path_corpora = join(params['path_corpora'], params['file_corpora'])
        print('loading corpora from ', params['path_corpora'])
        with open(file_path_corpora, 'rb') as infile:
            corpora = pickle.load(infile)
            dictionary = corpora[-1].dictionary
        print('end loading corpora')

    with open(os.path.join(params['log_dir'], 'git_commit_hash.txt'), 'w') as fileout:
        fileout.write('git commit hash: ' + str(params['git_commit']))

    # iterator for train set
    train_content = [item for sublist in [corp.corpus_train['content'] for corp in corpora] for item in sublist]
    train_targets = [item for sublist in [corp.corpus_train['targets'] for corp in corpora] for item in sublist]
    data_iterator_train = DataIterator(content=train_content, targets=train_targets,
                                       train=True, random_seed=params['shuffle_seed'])

    # iterator for validation set
    val_content = [item for sublist in [corp.corpus_val['content'] for corp in corpora] for item in sublist]
    val_targets = [item for sublist in [corp.corpus_val['targets'] for corp in corpora] for item in sublist]
    data_iterator_val = DataIterator(content=val_content, targets=val_targets,
                                     train=False, random_seed=params['shuffle_seed'])

    word_embeddings: np.array = None
    # load glove embeddings if necessary
    if params['word_embeddings']:
        t0 = time.time()
        print('Loading word embeddings')
        word_embeddings = load_word_embeddings(dictionary, params)
        print('Time to load word embeddings: {} mins'.format((time.time() - t0) / 60))

        if params['gpu']:
            word_embeddings = torch.from_numpy(word_embeddings).cuda()
        else:
            word_embeddings = torch.from_numpy(word_embeddings)

    model = MathModel(dictionary=dictionary, word_embeddings=word_embeddings, params=params)

    params_total = sum(p.nelement() for p in model.parameters() if p.requires_grad)
    print('Total number of trainable parameters: ', params_total)

    if params['gpu']:
        model = model.cuda()

    train_module(model=model, train_iterator=data_iterator_train, val_iterator=data_iterator_val, params=params)
