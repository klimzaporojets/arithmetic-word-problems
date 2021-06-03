import ast
import random


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def add_parser_default_values(parser):
    parser.register('type', 'bool', str2bool)

    parser.add_argument('--data_path', type=str,
                        help='path to the classification dataset')

    parser.add_argument('--train_folds', type=str,
                        help='train fold files')

    parser.add_argument('--val_folds', type=str,
                        help='val fold files')

    parser.add_argument('--test_folds', type=str,
                        help='test fold files')

    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--bs', type=int, help='batch size', dest='batch_size')

    parser.add_argument('--shuffle_seed', type=int,
                        help='seed to perform shuffle on classification file before splitting into train/val/test')
    parser.add_argument('--ss', type=int, default=1234, dest='shuffle_seed')

    parser.add_argument('--debug_length', type=int, default=-1, help='how many equations actually to process')

    parser.add_argument('--log_dir', type=str, default='tmp',
                        help='path to the directory where the output of experiment will be saved '
                             '(also tensorboard logs)')

    parser.add_argument('--dropout', type=float, default=0.0, help='dropout parameter to nnet')
    parser.add_argument('--dpt', type=float, default=0.0, dest='dropout')

    parser.add_argument('--word_embeddings_dim', type=int, default=4, help='dimensionality of word embeddings')
    parser.add_argument('--embd', type=int, default=4, dest='word_embeddings_dim')

    parser.add_argument('--word_embeddings', type='bool', default=False,
                        help='whether to use glove embeddings to initialize the lower layer of nnet')

    parser.add_argument('--emb', type='bool', default=False, dest='word_embeddings')

    parser.add_argument('--eg', type='bool', default=False, help='whether word embeddings have to get propagated '
                                                                 'gradients')

    parser.add_argument('--word_embeddings_path', type=str, default='data/glove/glove.6B.100d.txt',
                        help='path to word embeddings to be used if word_embeddings is in True')

    parser.add_argument('--learning_rate', type=float, default=1e-3, help='the learning rate')
    parser.add_argument('--lr', type=float, default=1e-3, dest='learning_rate')

    parser.add_argument('--variable_rate', type='bool', default=False,
                        help='whether the schedule is used or not, for now the schedule is just hardcoded in '
                             'train.py module')
    parser.add_argument('--lr_v', type='bool', default=False, dest='variable_rate')

    parser.add_argument('--scheduler_step_size', type=str, default='[20,40]', help='step size parameter for scheduler')
    parser.add_argument('--sss', type=str, default='[20,40]', dest='scheduler_step_size')

    parser.add_argument('--scheduler_gamma', type=float, default=0.316227766, help='the gamma parameter for scheduler')
    parser.add_argument('--sg', type=float, default=0.316227766, dest='scheduler_gamma')

    parser.add_argument('--epochs', type=int, default=100, help='the number of epochs')
    parser.add_argument('--eps', type=int, default=100, dest='epochs')

    parser.add_argument('--hdim_lstm1', type=int, default=256)
    parser.add_argument('--l1d', type=int, default=256, dest='hdim_lstm1')

    parser.add_argument('--hdim_lstm2', type=int, default=128)
    parser.add_argument('--l2d', type=int, default=128, dest='hdim_lstm2')

    parser.add_argument('--hdim_tree', type=int, default=128,
                        help='the dimensionality of the tree as well as baseline lstm!!!')
    parser.add_argument('--dtr', type=int, default=128, dest='hdim_tree')

    parser.add_argument('--layers_lstm', type=int, default=1)

    parser.add_argument('--bidirectional_lstm', type='bool', default=True)

    parser.add_argument('--hdim_ff1', type=int, default=256)
    parser.add_argument('--f1d', type=int, default=256, dest='hdim_ff1')

    parser.add_argument('--hdim_ff2', type=str, default='[1024,512]')
    parser.add_argument('--f2d', type=str, dest='hdim_ff2')

    parser.add_argument('--non_linearity_ff2', type=str, default='["tanh","relu"]', help='either tanh or relu')
    parser.add_argument('--f2nl', type=str, default='', dest='non_linearity_ff2')

    parser.add_argument('--outdim_ff1', type=int, default=128)
    parser.add_argument('--f1od', type=int, default=128, dest='outdim_ff1')
    parser.add_argument('--outdim_ff2', type=int, default=1)
    parser.add_argument('--f2od', type=int, default=1, dest='outdim_ff2')

    parser.add_argument('--do_ff1_nnet', type='bool', default=False)
    parser.add_argument('--df1', type='bool', default=False, dest='do_ff1_nnet')

    parser.add_argument('--do_ff2_nnet', type='bool', default=False,
                        help='whether ff2 network has to be executed, if False, a linear transformation '
                             'will be executed instead')
    parser.add_argument('--df2', type='bool', default=False, dest='do_ff2_nnet')

    parser.add_argument('--do_lstm1', type='bool', default=True)
    parser.add_argument('--dl1', type='bool', dest='do_lstm1')

    parser.add_argument('--lstm_equation_final_state', type='bool', default=True,
                        help='if in True, instead of doing max-pool, of all lstm hidden states, will take the '
                             'last hidden states and concatenate them')
    parser.add_argument('--l1fs', type='bool', dest='lstm_equation_final_state')

    parser.add_argument('--do_vertical_attention', type='bool', default=False)
    parser.add_argument('--dva', type='bool', default=False, dest='do_vertical_attention')

    parser.add_argument('--do_horizontal_attention', type='bool', default=False)
    parser.add_argument('--dha', type='bool', default=False, dest='do_horizontal_attention')

    parser.add_argument('--include_tree', type='bool', default=True, help='whether treeLSTM has to be executed, '
                                                                          'if not a simple lstm is executed instead')
    parser.add_argument('--itr', type='bool', dest='include_tree')

    parser.add_argument('--lstm_par_uniform', type='bool', default=False,
                        help='whether to initialize tree cell parameters uniformly instead of 0s')
    parser.add_argument('--lpu', type='bool', dest='lstm_par_uniform')

    parser.add_argument('--ordered_op', type='bool', default=False,
                        help='whether to use two different embeddings for - and / operations depending of the order'
                             'of the values')

    parser.add_argument('--oo', type='bool', default=False, dest='ordered_op')

    parser.add_argument('--do_intra_attention', type='bool', default=True)
    parser.add_argument('--dia', type='bool', dest='do_intra_attention')

    parser.add_argument('--att_only_operators', type='bool', default=False)
    parser.add_argument('--aoo', type='bool', dest='att_only_operators')

    parser.add_argument('--intra_attention_type', type=str, default='add',
                        help='dotp for dot product, add for additive, gen for general , '
                             'more details on https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html')

    parser.add_argument('--iat', type=str, dest='intra_attention_type')

    parser.add_argument('--hdim_attention', type=int, default=128,
                        help='parallel_tree_additive attention type requires '
                             'hidden dimension as hyper-parameter')

    parser.add_argument('--hatt', type=int, dest='hdim_attention')

    parser.add_argument('--two_gates_tree', type='bool', default=True,
                        help='whether to use the treelstm with two fully connected layers in gates: one for '
                             'left and another for right (the N-Ary version or also called NT-LSTM in our paper)')

    parser.add_argument('--tgt', type='bool', dest='two_gates_tree')

    parser.add_argument('--tree_gru', type='bool', default=False, help='whether to use LSTM or GRU tree')
    parser.add_argument('--tgru', type='bool', dest='tree_gru')

    parser.add_argument('--is1_gru', type='bool', default=False, help='whether to use LSTM or GRU as lstm1')
    parser.add_argument('--i1g', type='bool', dest='is1_gru')

    parser.add_argument('--is2_gru', type='bool', default=True, help='whether to use LSTM or GRU as lstm2')
    parser.add_argument('--i2g', type='bool', dest='is2_gru')

    parser.add_argument('--limit_sigmoid', type=float, default=0.5)
    parser.add_argument('--ls', type=float, default=0.5, dest='limit_sigmoid')

    parser.add_argument('--max_falses_per_equation', type=int, default=-1,
                        help='number of incorrect equations for each dataset point that will be added to training data')

    parser.add_argument('--mfpe', type=int, dest='max_falses_per_equation')

    parser.add_argument('--balanced', type='bool', default=False,
                        help='Whether to use pos_weight parameter to loss function')

    parser.add_argument('--bal', type='bool', default=False, dest='balanced')

    parser.add_argument('--test_output', type='bool', default=False,
                        help='Whether to perform the test of iterators by printing the encoded result to visually '
                             'check that it matches the original data')

    parser.add_argument('--use_ilp', type='bool', default=True,
                        help='whether instead of performing tree combinations (using tree_generator.py), uses the '
                             'output of ILP alges system.')

    parser.add_argument('--ilp_path', type=str,
                        help='The path to where the ILP alges equations are stored (used in case use_ilp is set to '
                             'True)')

    parser.add_argument('--max_equations_per_problem', type=int, default=100,
                        help='Maximum equations per problem to be considered during training/ validating and testing')
    parser.add_argument('--mepp', type=int, dest='max_equations_per_problem')

    parser.add_argument('--save_only_last', type='bool', default=False,
                        help='If has to save only the last model instead of the one performing best on dev set.')

    parser.add_argument('--fold', type=str, default='',
                        help='Just something assigned by main_experimenter.py to keep test folds apart.')

    parser.add_argument('--f', type=str, dest='fold')

    parser.add_argument('--use_brackets', type='bool', default=True,
                        help='Whether the information on the brackets has to be incorporated when doing tree '
                             'to lstm parse')

    parser.add_argument('--ub', type='bool', dest='use_brackets')

    parser.add_argument('--include_asymmetric', type='bool', default=True,
                        help='wheter to generate more asymmetric operations even if they don\'t exist in ILP')
    parser.add_argument('--ias', type='bool', dest='include_asymmetric')

    parser.add_argument('--length_asymmetric', type=int, default=10,
                        help='until what number of numbers in equation apply '
                             'the asymmetric inclusion (see --include_asymmetric)')
    parser.add_argument('--las', type=int, dest='length_asymmetric')

    parser.add_argument('--load_corpora', type='bool', default=False,
                        help='Whether instead of parsing/loading ilp equations/etc into corpus (takes long time), '
                             'just loads already pre-serialized corpus with train/dev/test sets')

    parser.add_argument('--lc', type='bool', dest='load_corpora', default=False)

    parser.add_argument('--path_corpora', type=str, default='',
                        help='if --load_corpora is in true, the path to the corpora to use')
    parser.add_argument('--pc', type=str, dest='path_corpora', default='')

    parser.add_argument('--file_corpora', type=str, default='',
                        help='if --load_corpora is in true, the file of the corpora to use')
    parser.add_argument('--fc', type=str, dest='file_corpora', default='')

    parser.add_argument('--save_all_models', type='bool', default=True,
                        help='whether has to save all the models produced by each epoch. If in False, only the '
                             'last and the best (on dev set) models are saved. If in True, all the models at end of '
                             'each epoch are saved. ')
    parser.add_argument('--sal', type='bool', dest='save_all_models', default=True)

    parser.add_argument('--manipulate_equation', type='bool', default=True,
                        help='whether to manipulate the equation if the one provided by '
                             'ILP does not begin with x= or '
                             'end with =x ')
    parser.add_argument('--meq', type='bool', dest='manipulate_equation', default=True)

    parser.add_argument('--continuous_lr_decrease', type='bool', default=False,
                        help='whether to decrease the learning rate continuously (linearly) making adjustments after'
                             ' each mini-batch')
    parser.add_argument('--cld', type='bool', dest='continuous_lr_decrease', default=False)

    parser.add_argument('--extract_enumerations', type='bool', default=True,
                        help='param that indicate whether enumerations have to be extracted (ex: Jill, Joe, and Adam '
                             'went to cinema, each of them spent ....)')
    parser.add_argument('--een', type='bool', dest='extract_enumerations', default=True)

    parser.add_argument('--variations_last', type='bool', default=True,
                        help='indicates whether the extra tree variations have to be performed last, prioritizing '
                             'first the trees loaded inside ILP files. I have strong evidence (ex: q274.txt.out) that '
                             'this way the coverage of long equations can be increased')
    parser.add_argument('--vl', type='bool', dest='variations_last', default=True)

    parser.add_argument('--end_lr_ratio', type=float, default=0.1,
                        help='in case --continous_lr_decrease is set to True, is the proportion of initial learning '
                             'rate that the end learning rate has to be (ex: 0.1, so if the model starts training with '
                             '1e-3, at the end the lr will be 1e-4)')
    parser.add_argument('--elr', type=float, dest='end_lr_ratio', default=0.1)

    params = vars(parser.parse_args())

    if params['shuffle_seed'] == 1234:
        params['shuffle_seed'] = random.randint(1, 9999)
    params['train_folds'] = ast.literal_eval(params['train_folds'])
    params['val_folds'] = ast.literal_eval(params['val_folds'])
    params['test_folds'] = ast.literal_eval(params['test_folds'])

    params['data_path'] = ast.literal_eval(params['data_path'])
    params['ilp_path'] = ast.literal_eval(params['ilp_path'])
    params['non_linearity_ff2'] = ast.literal_eval(params['non_linearity_ff2'])
    params['hdim_ff2'] = ast.literal_eval(params['hdim_ff2'])
    params['scheduler_step_size'] = ast.literal_eval(params['scheduler_step_size'])

    return params
