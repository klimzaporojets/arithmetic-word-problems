""" The idea behind this module is to make the neural net experimentation process more scalable to multiple machines
and GPUs"""

import argparse
import itertools
import json
import os
import random
import re

CREATE_RUNFILES = True  # if False, will not remove folders or create run files, only test number remaining

DELETE_IF_NOT_COMPLETED = True

SHUFFLE_EXPERIMENTS = False

# the hyperparameters that will actually appear in the path of the experiment
# optimized for current experiments
in_path = ['bs', 'dpt', 'lr', 'l1d', 'dtr', 'itr', 'dia', 'hatt', 'iat', 'tgt', 'fold', 'aoo', 'ias', 'las', 'lr_v',
           'sss', 'sg', 'f', 'fc', 'meq', 'vl', 'een', 'ss']


def cartesian_product(dicts):
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))


def get_configs(hyperparam_space):
    configs = []

    for hs in hyperparam_space:
        seeds = []
        if 'ss' in hs:
            seeds = hs['ss']
            del hs['ss']

        if len(hs['train_folds']) >= len(hs['fc']):
            train_folds = hs['train_folds']
            val_folds = hs['val_folds']
            test_folds = hs['test_folds']
            del hs['train_folds']
            del hs['val_folds']
            del hs['test_folds']
            for idx, fold in enumerate(train_folds):
                cart_prod = list(cartesian_product(hs))
                for curr_cart_prod in cart_prod:
                    curr_cart_prod['train_folds'] = train_folds[idx]
                    curr_cart_prod['val_folds'] = val_folds[idx]
                    curr_cart_prod['test_folds'] = test_folds[idx]
                    curr_cart_prod['fold'] = idx
                    if 'f' in curr_cart_prod.keys():
                        # f always in front
                        curr_f = curr_cart_prod['f']
                        del curr_cart_prod['f']
                        curr_cart_prod['f'] = curr_f
                configs += cart_prod
        else:
            load_corpora = hs['fc']
            del hs['fc']
            for idx, fold in enumerate(load_corpora):
                cart_prod = list(cartesian_product(hs))
                for curr_cart_prod in cart_prod:
                    curr_cart_prod['fc'] = load_corpora[idx]
                    curr_cart_prod['fold'] = idx
                    if 'f' in curr_cart_prod.keys():
                        # f always in front
                        curr_f = curr_cart_prod['f']
                        del curr_cart_prod['f']
                        curr_cart_prod['f'] = curr_f
                configs += cart_prod
        if len(seeds) > 0:
            assert len(seeds) >= len(configs)
            for idx, config in enumerate(configs):
                config['ss'] = seeds[idx]
    return configs


def get_experiment_name(c):
    to_ret: str = '_'.join(['{}={}'.format(k, c[k]) for k in c.keys() if k in in_path])
    to_ret = to_ret.replace('(', '')
    to_ret = to_ret.replace(')', '')
    return to_ret


def config2cmd(c, gpu=None, logpath='./tmp', script='src/main_nnet.py'):
    main = os.path.join('.', script)
    command = 'python -u {}'.format(main)

    for k in c:
        if type(c[k]) == str:
            command += ' --{} \"{}\"'.format(k, c[k])
        else:
            command += ' --{} {}'.format(k, c[k])

    command += ' --log_dir \"{}\"'.format(logpath)

    if gpu is not None:
        command = 'CUDA_VISIBLE_DEVICES={} {}'.format(gpu, command)

    return command


def main():
    parser = argparse.ArgumentParser(description='model experiments')
    parser.add_argument('--experiments_conf_path', default="experiments/conf/expauto3.json",
                        help="path with the configuration of experiments")
    parser.add_argument('--gpu', nargs='*', default=[0, 1], type=int,
                        help='ids of gpus that get a separate run file (e.g. --gpu 0 1 0 1  '
                             'for launch scripts for 2 machines with each 2 gpus)')
    parser.add_argument('--results_path', default='logs/nnet')
    parser.add_argument('--logs_path', default='logs/output_logs/')
    parser.add_argument('--experiments_path', default='experiments/nnet')
    parser.add_argument('--index', type=int, default=0)
    args = parser.parse_args()
    gpus = list(args.gpu)
    start_index = args.index

    os.makedirs(args.experiments_path, exist_ok=True)

    with open(args.experiments_conf_path) as f:
        nocomments = re.sub("//.*", "", f.read(), flags=re.MULTILINE)
        experiments_conf = json.loads(nocomments)

    for experiment_conf in experiments_conf:
        # determine which are already done
        configs_todo = []
        tag = experiment_conf['tag']
        for cfg in get_configs([experiment_conf['grid_search']]):
            experiment_path = os.path.join(args.results_path, tag, get_experiment_name(cfg))
            print('experiment path', experiment_path)
            log_file = os.path.join(experiment_path, 'log.txt')
            completed = False
            if os.path.isfile(log_file):
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    completed = 'Finished' in content

            if not completed:
                configs_todo.append((cfg, completed))

        print('number of remaining configs:', len(configs_todo))

        if SHUFFLE_EXPERIMENTS:
            random.shuffle(configs_todo)

        if CREATE_RUNFILES:
            config_chunks = [[] for _ in range(len(gpus))]
            for i, c in enumerate(configs_todo):
                config_chunks[i % len(gpus)].append(c)

            all_experiment_files = []
            for i, (gpu, config_chunk) in enumerate(zip(gpus, config_chunks)):
                sh_file = '%s_%d_gpu%s.sh' % (tag, i + start_index, str(gpu)) if gpu is not None else '%s%d.sh' % (
                    tag, i)
                sh_file = os.path.join(args.experiments_path, sh_file)
                with open(sh_file, 'w') as fID:
                    for job_id, (cfg, completed) in enumerate(config_chunk):
                        experiment_path = os.path.join(args.results_path, tag, get_experiment_name(cfg))
                        if not completed:
                            fID.write(
                                'mkdir -p {} \n'.format(experiment_path))  # only make just before running experiment
                            line = config2cmd(cfg, gpu, experiment_path, script='src/main_train_model.py')
                            fID.write(line + '\n\n')
                    print('constructed experiment file {} with {} configs'.format(sh_file, len(config_chunk)))
                all_experiment_files.append(sh_file)

            run_all_path = os.path.join(args.experiments_path, 'run_all_{}.sh'.format(tag))
            with open(run_all_path, 'w') as fID:
                # handy when running all on 1 machine
                for idx, ef in enumerate(all_experiment_files):
                    fID.write(
                        'nohup {} > {} 2>&1& \n'.format(ef, '{}/run_all_{}_{}.log'.format(args.logs_path, tag, idx)))

                # at the end write wait, so the os has to wait to finish, useful for gpulab, so the machine actually
                # waits before finishing
                fID.write('\nwait\n')

        else:
            print('only run in test mode, to see number of simulations remaining')


if __name__ == '__main__':
    main()
