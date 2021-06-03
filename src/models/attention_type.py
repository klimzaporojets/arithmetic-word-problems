# TODO: module not needed, should be deleted, currently adding it because needed to load the saved models

from enum import Enum


class AttentionType(Enum):
    AFFINE = 1
    GRU_CELL = 2
    # for short summary of types below: https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html
    PARALLEL_TREE_DOT_PRODUCT = 3
    PARALLEL_TREE_ADDITIVE = 4
    PARALLEL_TREE_GENERAL = 5
