import re
import sys

import numpy as np
import sympy


from data.dictionary import Dictionary


def represents_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def delete_trailing_zeros(s):
    if type(s) != str:
        return s
    if '.' in s:
        dot_ind = s.index('.')
        if dot_ind == -1:
            return s
        trails = s[dot_ind + 1:]
        if trails.count('0') == len(trails):
            return s[:dot_ind]
    return s


def cast_to_number(s):
    if type(s) != str:
        return s
    s = s.replace(',', '')
    if represents_int(s):
        return int(s)
    else:
        return float(s)


def tokenize(str):
    sep = set(['+', '-', '*', '/', '(', ')', '='])
    output = []
    last = pos = 0
    while pos < len(str):
        if str[pos] in sep:
            if last < pos:
                output.append(str[last:pos])

            if not (str[pos] == '-' and (output[-1] == '=' or output[-1] == '(')):
                output.append(str[pos])
                pos += 1
                last = pos
            else:
                pos += 1
        else:
            pos += 1
    if last < len(str):
        output.append(str[last:])
    return [x.strip() for x in output]


def get_letters_mapping(equation):
    nr_regex = r"\d*\.\d+|\d+,\d+|\d+"
    p = re.compile(nr_regex)
    last_end = 0
    trans_eq = ''
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l']
    mappings = dict()
    for idx, m in enumerate(p.finditer(equation)):
        curr_nr = equation[m.start():m.end()]
        mappings[letters[idx]] = curr_nr
        trans_eq += equation[last_end:m.start()]
        trans_eq += letters[idx]
        last_end = m.end()

    trans_eq += equation[last_end:]

    return {'trans_eq': trans_eq, 'mappings': mappings}


def normalize(equation):
    let_mappings = get_letters_mapping(equation)
    trans_eq = let_mappings['trans_eq']
    eq_idx = trans_eq.index('=')
    lhs = trans_eq[0:eq_idx]
    rhs = trans_eq[eq_idx + 1:]
    x = sympy.Symbol('x')
    lhs = sympy.sympify(lhs)
    rhs = sympy.sympify(rhs)
    norm = sympy.solve(lhs - rhs, x)

    str_norm = sympy.sstr(norm[0], order='lex')

    for (letter, number) in let_mappings['mappings'].items():
        str_norm = str_norm.replace(letter, number)
    str_norm = 'x=' + str_norm
    return str_norm


def stream(string, variables):
    sys.stdout.write(f'\r{string}' % variables)


def load_word_embeddings(dictionary: Dictionary, hypers: dict) -> np.array:
    word_embeddings_path = hypers['word_embeddings_path']

    nr_embedding_words = sum(1 for line in open(word_embeddings_path))
    with open(word_embeddings_path) as infile:
        first_line = infile.readline()
        emb_dimensionality = len(first_line.split(' ')) - 1
    all_embeddings = np.zeros((nr_embedding_words, emb_dimensionality), dtype=np.float32)
    mask_non_empty = np.zeros((dictionary.len(),), dtype=np.int8)
    dict_embeddings = np.zeros((dictionary.len(), emb_dimensionality), dtype=np.float32)
    with open(word_embeddings_path) as infile:
        for idx, line in enumerate(infile):
            values = line.split(' ')
            word_embedding = np.asarray(values[1:])
            all_embeddings[idx, :] = word_embedding
            word: str = values[0]
            word_idx = dictionary(word.lower().strip(), unknown_idx=-1)
            if word_idx != -1:
                mask_non_empty[word_idx] = 1
                dict_embeddings[word_idx] = word_embedding

    emb_mean, emb_std = all_embeddings.mean(), all_embeddings.std()
    # unkn_embeddings = dict_embeddings[mask_non_empty == 0]
    dict_embeddings[mask_non_empty == 0] = \
        np.random.normal(emb_mean, emb_std, (dict_embeddings[mask_non_empty == 0].shape[0],
                                             dict_embeddings[mask_non_empty == 0].shape[1]))
    return dict_embeddings
