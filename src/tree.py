from math import isclose

import torch

from utils.utils import cast_to_number


class Node(object):
    # numerical value (ex: 12.34) or one of possible operators ('+', '-', '*', '/')
    value = None
    # the alias is used in case we want to later on print it instead of the actual value, for example, the value con be
    # the number 1.2, but the alias can be '1.20': the exact string as it occurs in the text
    alias = None
    idx_token = -1
    # ntype is either 'operator' or 'value'
    ntype: str = None

    # the left and right is of type Node
    left = None
    right = None
    level: int = None
    h_state = None
    c_state = None
    h_att_state = None
    c_att_state = None
    attention_weights = None

    def __init__(self, value, alias=None, idx_token=-1, ntype='value', source='digit'):
        self.value = value
        self.alias = alias
        self.idx_token = idx_token
        self.ntype = ntype
        self.left = None
        self.right = None
        # indicates how in text it was extracted ('digit', 'str_token', 'enumeration')
        self.source = source

    def __str__(self):
        return 'Node({}({}))'.format(self.value, self.idx_token)


class BinaryTree(object):
    print_alias = True

    def __init__(self, root):
        self.root = root

    def print_tree(self, traversal_type):
        if traversal_type == 'preorder':
            return self.preorder_print(self.root, '')
        elif traversal_type == 'inorder':
            return self.inorder_print(self.root, '')
        elif traversal_type == 'postorder':
            return self.postorder_print(self.root, '')
        else:
            print('Traversal type {} is not supported.'.format(traversal_type))

    def preorder_print(self, start, traversal):
        """Root->Left->Right"""
        if start:
            traversal += (str(start.value) + ', ')
            traversal = self.preorder_print(start.left, traversal)
            traversal = self.preorder_print(start.right, traversal)
        return traversal

    def inorder_print(self, start: Node, traversal):
        """Left->Root->Right"""
        if start:
            if start.ntype == 'operator':
                traversal += '('
            traversal = self.inorder_print(start.left, traversal)
            if start.ntype == 'value':
                if self.print_alias and start.idx_token > -1 is not None:
                    traversal += (str(start.value) + '(' + str(start.idx_token) + ')')
                else:
                    traversal += (str(start.value) + '')
            else:
                traversal += (str(start.value) + '')

            traversal = self.inorder_print(start.right, traversal)
            if start.ntype == 'operator':
                traversal += ')'
        return traversal

    def inorder_print_alias(self, start: Node, traversal):
        """Print alias instead of value"""
        if start:
            if start.ntype == 'operator':
                traversal += '('
            traversal = self.inorder_print_alias(start.left, traversal)
            if start.ntype == 'value':
                if self.print_alias and start.idx_token > -1 is not None:
                    traversal += (str(start.alias) + '(' + str(start.idx_token) + ')')
                else:
                    traversal += (str(start.alias) + '')
            else:
                traversal += (str(start.alias) + '')

            traversal = self.inorder_print_alias(start.right, traversal)
            if start.ntype == 'operator':
                traversal += ')'
        return traversal

    def postorder_print(self, start, traversal):
        """Left->Right->Root"""
        if start:
            traversal = self.postorder_print(start.left, traversal)
            traversal = self.postorder_print(start.right, traversal)
            traversal += (str(start.value) + ', ')
        return traversal

    def __get_values__(self, start: Node):
        if start is None:
            return []
        if start.ntype == 'value':
            return [abs(start.value)] + self.__get_values__(start.left) + self.__get_values__(start.right)

        return [] + self.__get_values__(start.left) + self.__get_values__(start.right)

    def get_values(self):
        values = self.__get_values__(self.root)
        return values

    def __get_indices__(self, start: Node):
        if start is None:
            return []
        if start.ntype == 'value':
            return [start.idx_token]

        return [] + self.__get_indices__(start.left) + self.__get_indices__(start.right)

    def get_indices(self):
        """Gets the indices of the values in the original tokenized sentence"""
        indices = self.__get_indices__(self.root)
        return indices

    def __rearrange__(self, start: Node):
        if start.left.ntype == 'value' and start.left.value < 0 and start.value == '+':
            temp_left = start.left
            start.left = start.right
            temp_left.value = temp_left.value * (-1)
            if temp_left.alias is not None and temp_left.alias[:1] == '-':
                temp_left.alias = temp_left.alias[1:]
            elif temp_left.alias is not None:
                print("WARN!!: SOMETHING WRONG, ALIAS SHOULD BE NEGATIVE WHEN RE-ARRANGING")
            start.value = '-'
            start.alias = '-'
            start.right = temp_left

        if start.left.ntype == 'operator':
            self.__rearrange__(start.left)

        if start.right.ntype == 'operator':
            self.__rearrange__(start.right)

    def rearrange_tree(self):
        """ Re-arranges the tree, so the equations like (-6 + 16) become (16 - 6)"""
        self.__rearrange__(self.root)

    def __commute_tree__(self, start: Node, comm: tuple, operators: set, comm_idx=0):
        if start is None:
            return comm_idx
        if start.ntype == 'operator' and start.value in operators and comm[comm_idx] == 1:
            left_temp = start.left
            start.left = start.right
            start.right = left_temp
        if start.ntype == 'operator' and start.value in operators:
            comm_idx += 1

        comm_idx = self.__commute_tree__(start.left, comm, operators, comm_idx)
        comm_idx = self.__commute_tree__(start.right, comm, operators, comm_idx)
        return comm_idx

    def commute_tree(self, comm: tuple, operators=None):
        if operators is None:
            operators = {'*', '+'}
        self.__commute_tree__(self.root, comm, operators)

    def __count_operators__(self, start: Node, operators: set):
        if start is None:
            return 0
        if start.ntype == 'operator' and start.value in operators:
            return 1 + self.__count_operators__(start.left, operators) + \
                   self.__count_operators__(start.right, operators)
        else:
            return 0 + self.__count_operators__(start.left, operators) + \
                   self.__count_operators__(start.right, operators)

    def count_operators(self, operators: set):
        return self.__count_operators__(self.root, operators)

    def __enrich_indices__(self, start: Node, number_nodes: list, neighbour=None):
        if start.ntype != 'value':
            self.__enrich_indices__(start=start.left, number_nodes=number_nodes, neighbour=start.right)
            self.__enrich_indices__(start=start.right, number_nodes=number_nodes, neighbour=start.left)
        else:

            if start.idx_token > -1:
                # if index already assigned, then don't proceed, leave it as it is
                return

            value = start.value
            other_values = [(idx, node) for (idx, node) in enumerate(number_nodes) if cast_to_number(node.value)
                            == cast_to_number(value)]

            if len(other_values) == 1:
                start.idx_token = other_values[0][1].idx_token

            elif len(other_values) > 1:
                # first uses the numbers matched to the actual digits in the text, only then uses numbers
                # that were mapped (ex: weeks to 7, dimes, quarters, etc.)
                # if there is more tokens of the same value, then deletes the currently used
                other_values_priority = [v for v in other_values if v[1].source == 'digit']

                if len(other_values_priority) == 0:
                    other_values_priority = [v for v in other_values
                                             if (v[1].source == 'enumeration' or v[1].source == 'str_token')]

                if len(other_values_priority) > 1:
                    # here chooses the one that is the most closely located to its neighbour:
                    neighbour_idx_tkn = -1
                    if neighbour.idx_token > -1:
                        neighbour_idx_tkn = neighbour.idx_token
                        # print()
                    elif neighbour.ntype == 'value':
                        other_values_nbr = [(idx, node) for (idx, node) in enumerate(number_nodes) if
                                            cast_to_number(node.value) == cast_to_number(neighbour.value)]
                        if len(other_values_nbr) == 1:
                            neighbour_idx_tkn = other_values_nbr[0][1].idx_token

                    if neighbour_idx_tkn > -1:
                        idx_best = -1
                        lowest_found = 99999
                        for idx, curr_option in enumerate(other_values_priority):
                            abs_distance = abs(curr_option[1].idx_token - neighbour_idx_tkn)
                            if abs_distance < lowest_found:
                                lowest_found = abs_distance
                                idx_best = idx

                        if idx_best > -1:
                            other_values_priority = [other_values_priority[idx_best]]

                    start.idx_token = other_values_priority[0][1].idx_token
                    del number_nodes[other_values_priority[0][0]]
                else:
                    start.idx_token = other_values_priority[0][1].idx_token
                    del number_nodes[other_values_priority[0][0]]

    def enrich_indices(self, number_nodes: list):
        """Enriches the indices of the current tree using the indices loaded in the nodes"""
        self.__enrich_indices__(self.root, number_nodes=number_nodes)

    def __is_valid_tree__(self, start: Node):
        if start.ntype == 'value':
            if str(start.value)[:1] == '-':
                return False
            else:
                return True
        else:
            ret1 = self.__is_valid_tree__(start.left)
            ret2 = self.__is_valid_tree__(start.right)
            if not ret1 or not ret2:
                return False
            else:
                return True

    def is_valid_tree(self):
        """ If any node value starts with -, then is not valid """
        return self.__is_valid_tree__(self.root)

    def __align__(self, start: Node, alignments: list, numbers, curr_id: int):
        if start.ntype == 'value':
            if start.idx_token > -1:
                # if index already assigned, then don't proceed, leave it as it is, show warning
                print('WARNING, index already assigned inside __align__')
                return

            nr = numbers[alignments[curr_id]]['number']

            if start.value != nr:
                print('something wrong in __align__')

            assert start.value == nr

            start.idx_token = numbers[alignments[curr_id]]['idx_token']
            curr_id = curr_id + 1
            return curr_id
        else:
            curr_id = self.__align__(start=start.left, alignments=alignments, numbers=numbers, curr_id=curr_id)
            curr_id = self.__align__(start=start.right, alignments=alignments, numbers=numbers, curr_id=curr_id)
            return curr_id

    def align(self, alignments: list, numbers):
        self.__align__(start=self.root, alignments=alignments, numbers=numbers, curr_id=0)

    def __get_attention_weights__(self, start: Node, get_only_att_of_operators: bool):
        if start is None:
            return []
        att_weights = []
        left_weights = self.__get_attention_weights__(start.left, get_only_att_of_operators=get_only_att_of_operators)
        if start.ntype == 'operator':
            left_weights[0]['label'] = '(' + str(left_weights[0]['label'])

        att_weights.extend(left_weights)

        # if only operator's attention, then if value, sets the weights just to 0
        att_to_add = start.attention_weights
        if get_only_att_of_operators and start.ntype == 'value':
            att_to_add = torch.zeros(att_to_add.size())

        att_weights.extend([{'att_weights': att_to_add, 'label': start.value}])

        right_weights = self.__get_attention_weights__(start.right, get_only_att_of_operators=get_only_att_of_operators)
        if start.ntype == 'operator':
            right_weights[len(right_weights) - 1]['label'] = str(right_weights[len(right_weights) - 1]['label']) + ')'
        att_weights.extend(right_weights)
        return att_weights

    def get_attention_weights(self, get_only_att_of_operators=True):
        return self.__get_attention_weights__(self.root, get_only_att_of_operators=get_only_att_of_operators)

    def __evaluate__(self, start: Node):
        if start.ntype == 'value':
            return start.value
        else:
            val_left = self.__evaluate__(start.left)
            val_right = self.__evaluate__(start.right)
            if start.value == '+':
                return val_left + val_right
            elif start.value == '-':
                return val_left - val_right
            elif start.value == '*':
                return val_left * val_right
            elif start.value == '/':
                if not isclose(val_right, 0.0):
                    return val_left / val_right
                else:
                    return 999999999999
            else:
                raise NotImplementedError

    def evaluate(self):
        return self.__evaluate__(self.root)

    # cleans the tree from the tensors added during forward tree_lstm execution
    @staticmethod
    def clean_tree_tensors(start: Node):
        if start is not None:
            start.h_state = None
            start.c_state = None
            start.h_att_state = None
            start.c_att_state = None
            BinaryTree.clean_tree_tensors(start.left)
            BinaryTree.clean_tree_tensors(start.right)

    @staticmethod
    def get_min_idx(node: Node):
        if node is None:
            return 99999
        min_left = BinaryTree.get_min_idx(node.left)
        min_right = BinaryTree.get_min_idx(node.left)

        if min_left < min_right:
            mint = min_left
        else:
            mint = min_right
        if node.ntype == 'value' and node.idx_token < mint:
            mint = node.idx_token
        return mint

    @staticmethod
    def get_min_idx_left(node: Node):
        return BinaryTree.get_min_idx(node.left)

    @staticmethod
    def get_min_idx_right(node: Node):
        return BinaryTree.get_min_idx(node.right)

    def __eq__(self, other):
        return self.inorder_print(self.root, '') == \
               other.inorder_print(other.root, '')

    def __hash__(self):
        return hash(self.inorder_print(self.root, ''))

    def __str__(self):
        return self.inorder_print(self.root, '')

    def print_aliases(self):
        return self.inorder_print_alias(self.root, '')
