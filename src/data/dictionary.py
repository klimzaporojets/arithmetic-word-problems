""" implementation of the token dictionary """
import json


class Dictionary(object):

    def __init__(self, *symbols):
        self.sym2idx = {s: i for i, s in enumerate(symbols)}
        self.idx2sym = {v: k for k, v in self.sym2idx.items()}

    def add(self, *symbols):
        for s in symbols:
            if s not in self.sym2idx:
                # self.idx2sym.append(s)
                new_id = len(self.sym2idx)
                self.sym2idx[s] = new_id
                self.idx2sym[new_id] = s

    def __call__(self, *symbols, unknown_idx=-1):
        return [self.sym2idx[s] if s in self.sym2idx else unknown_idx for s in symbols]

    def decode(self, *idxs):
        return [self.idx2sym[idx] if idx in self.idx2sym else 'UNKNKNKN' for idx in idxs]

    def __len__(self):
        return len(self.sym2idx)

    def len(self):
        return self.__len__()

    def serialize_to_json(self, output_path=None):
        json.dump(self.sym2idx, open(output_path, 'wt'), indent=4, sort_keys=True)

    def load_from_json(self, input_path=None):
        self.sym2idx = json.load(open(input_path, 'r'))
        self.idx2sym = {value: key for key, value in self.sym2idx.items()}
