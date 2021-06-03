import torch
import torch as th
import torch.nn as nn

from models.tree_lstm_cell import ChildSumTreeLSTM
from tree import Node, BinaryTree


class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super(FeedforwardNeuralNetModel, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.fc1b = nn.Linear(hidden_dim, hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.dropout(self.relu(out))
        out = self.fc2(out)
        return out


class FeedforwardNeuralNetModelV2(nn.Module):
    """More parametrized version of nnet"""

    def __init__(self, input_dim, hidden_dims: list, hidden_non_linearities: list, output_dim, dropout):
        super(FeedforwardNeuralNetModelV2, self).__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.dim_model = torch.nn.Sequential()

        prev_dim = -1

        if len(hidden_dims) > len(hidden_non_linearities):
            max_params = len(hidden_non_linearities)
        else:
            max_params = len(hidden_dims)

        hidden_dims = hidden_dims[0:max_params]
        hidden_non_linearities = hidden_non_linearities[0:max_params]

        for idx, (dim, non_linearity) in enumerate(zip(hidden_dims, hidden_non_linearities)):
            if idx == 0:
                self.dim_model.add_module("dense_{}".format(idx), torch.nn.Linear(input_dim, dim))
            else:
                self.dim_model.add_module("dense_{}".format(idx), torch.nn.Linear(prev_dim, dim))

            prev_dim = dim
            activation = torch.nn.ReLU()
            if non_linearity == 'tanh':
                activation = torch.nn.Tanh()

            self.dim_model.add_module('{}_{}'.format(non_linearity, idx), activation)
            self.dim_model.add_module('dpt_{}'.format(idx), self.dropout)

        self.dim_model.add_module("dense_{}".format(len(hidden_dims)), torch.nn.Linear(prev_dim, output_dim))

    def forward(self, x):
        return self.dim_model(x)


class MathModel(nn.Module):

    def __init__(self, dictionary, word_embeddings, params):

        vocab_size = dictionary.len()
        super(MathModel, self).__init__()
        num_classes = 2

        self.gpu = params['gpu']
        self.word_emb_dim = params['word_embeddings_dim']

        self.word_embeddings = nn.Embedding(vocab_size, self.word_emb_dim)

        self.do_ff1_nnet = params['do_ff1_nnet']
        self.do_lstm1 = params['do_lstm1']
        self.hdim_lstm2 = params['hdim_lstm2']
        self.hdim_lstm1 = params['hdim_lstm1']
        self.hdim_tree = params['hdim_tree']
        self.hdim_ff1 = params['hdim_ff1']
        self.outdim_ff1 = params['outdim_ff1']
        self.hdim_ff2 = params['hdim_ff2']
        self.outdim_ff2 = params['outdim_ff2']
        self.layers_lstm = params['layers_lstm']
        self.bidirectional_lstm = params['bidirectional_lstm']
        self.include_tree = params['include_tree']
        self.ordered_word = params['ordered_op']
        self.do_intra_attention = params['do_intra_attention']
        self.two_gates_tree = params['two_gates_tree']
        self.ff2_layers = 2
        self.ff2_non_linearities = params['non_linearity_ff2']
        self.ff2_hid_dimensions = params['hdim_ff2']
        self.do_ff2_nnet = params['do_ff2_nnet']
        self.use_brackets = params['use_brackets']
        self.lstm_par_uniform = params['lstm_par_uniform']
        self.lstm_equation_final_state = params['lstm_equation_final_state']
        self.att_only_operators = params['att_only_operators']

        if word_embeddings is not None:
            self.word_embeddings.weight.data.copy_(word_embeddings)
            self.word_embeddings.weight.requires_grad = params['eg']

        self.dropout = nn.Dropout(params['dropout'])
        self.linear = nn.Linear(self.hdim_lstm1, num_classes)

        hidden_size1 = int(self.hdim_lstm1 / 2 if self.bidirectional_lstm else self.hdim_lstm1)

        hdim_prem = self.word_emb_dim
        if self.do_lstm1:
            # LSTM for hypothesis representation
            self.lstm_hypothesis = nn.LSTM(input_size=self.word_emb_dim, hidden_size=hidden_size1,
                                           num_layers=self.layers_lstm,
                                           batch_first=True,
                                           bidirectional=self.bidirectional_lstm)

            hdim_prem = self.hdim_lstm1

        # TreeLSTM
        if self.include_tree:
            cell = ChildSumTreeLSTM
            self.cell = cell(hdim_prem, self.hdim_tree, ordered_op=self.ordered_word,
                             two_gates=self.two_gates_tree, lstm_par_uniform=self.lstm_par_uniform)

        else:
            # tries simple embeddings with LSTM
            self.x_product = nn.Parameter(th.zeros(hdim_prem), requires_grad=True)
            self.x_product.data.uniform_(-1.0, 1.0)

            self.x_division = nn.Parameter(th.zeros(hdim_prem), requires_grad=True)
            self.x_division.data.uniform_(-1.0, 1.0)

            self.x_sum = nn.Parameter(th.zeros(hdim_prem), requires_grad=True)
            self.x_sum.data.uniform_(-1.0, 1.0)

            self.x_min = nn.Parameter(th.zeros(hdim_prem), requires_grad=True)
            self.x_min.data.uniform_(-1.0, 1.0)

            if self.use_brackets:
                self.bracket_left = nn.Parameter(th.zeros(hdim_prem), requires_grad=True)
                self.bracket_left.data.uniform_(-1.0, 1.0)

                self.bracket_right = nn.Parameter(th.zeros(hdim_prem), requires_grad=True)
                self.bracket_right.data.uniform_(-1.0, 1.0)

            if self.bidirectional_lstm:
                hd_dim = int(self.hdim_tree / 2)
            else:
                hd_dim = self.hdim_tree

            self.cell = nn.LSTM(input_size=hdim_prem, hidden_size=hd_dim, num_layers=self.layers_lstm,
                                batch_first=True, bidirectional=self.bidirectional_lstm)

        if self.do_ff1_nnet:
            self.f_hyp = FeedforwardNeuralNetModel(input_dim=self.hdim_tree * 1, hidden_dim=self.hdim_ff1,
                                                   output_dim=self.outdim_ff1, dropout=self.dropout)

            self.hdim_tree = self.outdim_ff1

        in_dim = self.hdim_tree

        if self.do_ff2_nnet:
            self.f2 = FeedforwardNeuralNetModelV2(input_dim=in_dim,
                                                  hidden_dims=self.ff2_hid_dimensions,
                                                  hidden_non_linearities=self.ff2_non_linearities,
                                                  output_dim=self.outdim_ff2, dropout=self.dropout)
        else:
            self.f2 = torch.nn.Linear(in_dim, self.outdim_ff2)

    def get_seq_hidden_tree(self, node_run: Node, intra_attention=False):
        if node_run.ntype == 'value':
            if self.gpu:
                return torch.Tensor().cuda()
            else:
                return torch.Tensor()
        else:
            left_hidden: torch.Tensor = self.get_seq_hidden_tree(node_run.left)
            right_hidden: torch.Tensor = self.get_seq_hidden_tree(node_run.right)
            if not intra_attention:
                self_hidden: torch.Tensor = node_run.h_state
            else:
                self_hidden: torch.Tensor = node_run.h_att_state
            result = torch.cat((left_hidden, right_hidden, self_hidden), dim=0)
            return result

    def get_seq_hidden_lstm(self, node_run: Node, input_embeddings):
        if node_run is None:
            if self.gpu:
                return torch.Tensor().cuda()
            else:
                return torch.Tensor()
        left_emb = self.get_seq_hidden_lstm(node_run.left, input_embeddings)

        if node_run.ntype == 'operator':
            if node_run.value == '+':
                curr_oper_emb = self.x_sum.unsqueeze(0)
            elif node_run.value == '/':
                curr_oper_emb = self.x_division.unsqueeze(0)
            elif node_run.value == '*':
                curr_oper_emb = self.x_product.unsqueeze(0)
            elif node_run.value == '-':
                curr_oper_emb = self.x_min.unsqueeze(0)
            else:
                raise NotImplementedError
        else:
            curr_oper_emb = input_embeddings[node_run.idx_token].unsqueeze(0)

        right_emb = self.get_seq_hidden_lstm(node_run.right, input_embeddings)
        if not self.use_brackets or node_run.ntype == 'value':
            return torch.cat((left_emb, curr_oper_emb, right_emb))
        else:
            return torch.cat(
                (self.bracket_left.unsqueeze(0), left_emb, curr_oper_emb, right_emb, self.bracket_right.unsqueeze(0)))

    def forward(self, questions, batch_trees):
        """Compute tree-lstm prediction given a batch."""
        questions.unsqueeze_(0)
        word_embs = self.word_embeddings(questions)
        word_embs = self.dropout(word_embs)

        if self.do_lstm1:

            out_hyp, (_, _) = self.lstm_hypothesis(word_embs)
            out_hyp = self.dropout(out_hyp)
        else:
            out_hyp = word_embs

        if self.include_tree:
            # node_run = self.cell(batch_trees.root, out_hyp[0], attentive=False)
            node_run = self.cell(batch_trees.root, out_hyp[0])

            out_hyp_tree = node_run[1]

            out_hyp_tree = self.dropout(out_hyp_tree)
        else:
            hid_seq_lstm = self.get_seq_hidden_lstm(node_run=batch_trees.root, input_embeddings=out_hyp[0])
            hid_seq_lstm = hid_seq_lstm.unsqueeze(0)
            out_hyp_tree, (hidden_st, c_st) = self.cell(hid_seq_lstm)

            if not self.lstm_equation_final_state:
                out_hyp_tree = self.dropout(out_hyp_tree)[0]
            else:
                hidden_state_h_reshaped = hidden_st.view(1, 1, -1)[0]
                out_hyp_tree = self.dropout(hidden_state_h_reshaped)

        BinaryTree.clean_tree_tensors(batch_trees.root)

        mix_hyp = out_hyp_tree.transpose(0, 1)

        if self.do_ff1_nnet:
            mix_hyp_ff = self.f_hyp(mix_hyp.transpose(0, 1)).unsqueeze(0)
        else:
            mix_hyp_ff = mix_hyp.transpose(0, 1).unsqueeze(0)

        if mix_hyp_ff[0].shape[0] > 1:
            mix_hyp_max_pool = torch.max(mix_hyp_ff[0], dim=0)[0]
        else:
            mix_hyp_max_pool = mix_hyp_ff[0][0]

        final_res = self.f2(mix_hyp_max_pool)

        return final_res
