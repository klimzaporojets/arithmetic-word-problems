import torch
import torch as th
import torch.nn as nn

from tree import Node, BinaryTree


class ChildSumTreeLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, ordered_op: bool,
                 two_gates=False,
                 lstm_par_uniform=False):
        """if do_attention is true, then will do the attention on the input and pass the result to
        target_node"""
        super(ChildSumTreeLSTM, self).__init__()

        self.two_gates = two_gates
        self.ordered_op = ordered_op

        self.W_i = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_o = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_u = nn.Linear(input_dim, hidden_dim, bias=False)

        self.U_i = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.U_o = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.U_u = nn.Linear(hidden_dim, hidden_dim, bias=False)

        if two_gates:
            self.U_i2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.U_o2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.U_u2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.U_f2 = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.b_i = nn.Parameter(th.zeros(1, hidden_dim), requires_grad=True)
        self.b_o = nn.Parameter(th.zeros(1, hidden_dim), requires_grad=True)
        self.b_u = nn.Parameter(th.zeros(1, hidden_dim), requires_grad=True)

        self.W_f = nn.Linear(input_dim, hidden_dim, bias=False)
        self.U_f = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.b_f = nn.Parameter(th.zeros(1, hidden_dim), requires_grad=True)

        if lstm_par_uniform:
            self.b_i.data.uniform_(-1.0, 1.0)
            self.b_o.data.uniform_(-1.0, 1.0)
            self.b_u.data.uniform_(-1.0, 1.0)
            self.b_f.data.uniform_(-1.0, 1.0)

        self.input_dim: int = input_dim
        self.hidden_dim: int = hidden_dim

        self.x_product = nn.Parameter(th.zeros(input_dim), requires_grad=True)
        self.x_product.data.uniform_(-1.0, 1.0)

        # division is not invertible, so two different parameters
        if ordered_op:
            self.x_division_left = nn.Parameter(th.zeros(input_dim), requires_grad=True)
            self.x_division_left.data.uniform_(-1.0, 1.0)
            self.x_division_right = nn.Parameter(th.zeros(input_dim), requires_grad=True)
            self.x_division_right.data.uniform_(-1.0, 1.0)
        else:
            self.x_division = nn.Parameter(th.zeros(input_dim), requires_grad=True)
            self.x_division.data.uniform_(-1.0, 1.0)

        self.x_sum = nn.Parameter(th.zeros(input_dim), requires_grad=True)
        self.x_sum.data.uniform_(-1.0, 1.0)

        # subtraction is not invertible, so two different parameters
        if ordered_op:
            self.x_min_left = nn.Parameter(th.zeros(input_dim), requires_grad=True)
            self.x_min_left.data.uniform_(-1.0, 1.0)
            self.x_min_right = nn.Parameter(th.zeros(input_dim), requires_grad=True)
            self.x_min_right.data.uniform_(-1.0, 1.0)
        else:
            self.x_min = nn.Parameter(th.zeros(input_dim), requires_grad=True)
            self.x_min.data.uniform_(-1.0, 1.0)

        self.c_init_state = nn.Parameter(th.zeros((2, self.hidden_dim)), requires_grad=True)

        self.h_init_state = nn.Parameter(th.zeros((2, self.hidden_dim)), requires_grad=True)

    def node_forward(self, x_input, c_state, h_state):
        if not self.two_gates:
            h_sum = torch.sum(h_state, dim=0, keepdim=True)
            i = torch.sigmoid(self.W_i(x_input) + self.U_i(h_sum) + self.b_i)
            o = torch.sigmoid(self.W_o(x_input) + self.U_o(h_sum) + self.b_o)
            u = torch.tanh(self.W_u(x_input) + self.U_u(h_sum) + self.b_u)
            f = torch.sigmoid(self.W_f(x_input).repeat(len(h_state), 1) + self.U_f(h_state) + self.b_f)
        else:
            i = torch.sigmoid(self.W_i(x_input) + self.U_i(h_state[0]) + self.U_i2(h_state[1]) + self.b_i)
            o = torch.sigmoid(self.W_o(x_input) + self.U_o(h_state[0]) + self.U_o2(h_state[1]) + self.b_o)
            u = torch.tanh(self.W_u(x_input) + self.U_u(h_state[0]) + self.U_u2(h_state[1]) + self.b_u)
            f = torch.sigmoid(self.W_f(x_input).repeat(len(h_state), 1) +
                              torch.cat(
                                  (self.U_f(h_state[0]).unsqueeze(0), self.U_f2(h_state[1]).unsqueeze(0))) + self.b_f)

        fc = torch.mul(f, c_state)
        c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)
        h = torch.mul(o, torch.tanh(c))
        # context = None

        return c, h, None

    def get_op_input(self, node: Node):
        if node.value == '+':
            return self.x_sum
        elif node.value == '/':
            if self.ordered_op:
                if BinaryTree.get_min_idx_left(node) < BinaryTree.get_min_idx_right(node):
                    return self.x_division_left
                else:
                    return self.x_division_right
            else:
                return self.x_division
        elif node.value == '*':
            return self.x_product
        elif node.value == '-':
            if self.ordered_op:
                if BinaryTree.get_min_idx_left(node) < BinaryTree.get_min_idx_right(node):
                    return self.x_min_left
                else:
                    return self.x_min_right
            else:
                return self.x_min
        else:
            raise NotImplementedError

    def forward_normal_v2(self, node: Node, input):

        if node.ntype == 'value':
            c_state = self.c_init_state
            h_state = self.h_init_state
            x = input[node.idx_token]
        else:
            res_left = self.forward_normal_v2(node.left, input)
            res_right = self.forward_normal_v2(node.right, input)

            c_state = torch.cat((res_left[0], res_right[0]))
            h_state = torch.cat((res_left[1], res_right[1]))

            x = self.get_op_input(node)

        res = self.node_forward(x_input=x, c_state=c_state, h_state=h_state)
        node.h_state = res[1]

        return res

    def forward(self, node: Node, input
                # , gpu=False
                ):
        return self.forward_normal_v2(node=node, input=input)
