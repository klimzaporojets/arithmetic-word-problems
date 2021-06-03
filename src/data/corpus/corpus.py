import json
import os
import re
from abc import abstractmethod
from math import isclose

import numpy as np
from spacy.tokens.doc import Doc

from data.corpus.data import Question, nlp, load_into_binary_tree, get_variations, \
    get_ilp_trees_variations_last
from data.corpus.parser import parse
from data.dictionary import Dictionary
from tree import Node, BinaryTree
from utils.utils import normalize, tokenize, cast_to_number, delete_trailing_zeros

nr_regex = r"[-+]?\d*\.\d+|\d+,\d+|\d+"
str_to_numbers = {
    'one': 1,
    'two': 2,
    'three': 3,
    'four': 4,
    'five': 5,
    'six': 6,
    'seven': 7,
    'eight': 8,
    'nine': 9,
    'ten': 10,
    'dozen': 12,
    'dozens': 12,
    'week': 7,
    'weeks': 7
}

# one of these can be mapped to multiple numbers
str_to_numbers_units = {
    'dime': [0.1, 10],
    'dimes': [0.1, 10],
    'penny': [0.01],
    'pennies': [0.01],
    'quarter': [0.25, 25],
    'quarters': [0.25, 25],
    'nickel': [0.05],
    'nickels': [0.05],
    'cents': [100],
    'half': [0.5, 2],
    'minutes': [60]
}


class Corpus:

    def __init__(self, **args):
        self.PAD, self.BOW, self.EOW, self.UNK = '<pad>', '<bow>', '<eow>', '<UNK>'
        if len(args) > 0:
            loaded_ids = args['loaded_ids']
            params = args['params']
            dictionary = args['dictionary']

            data_path = params['data_path']['single_eq']
            debug_length = params['debug_length']
            max_falses_per_equation = params['max_falses_per_equation']
            max_equations_per_problem = params['max_equations_per_problem']
            ilp_path = params['ilp_path']['single_eq']
            include_asymmetric = params['include_asymmetric']
            length_asymmetric = params['length_asymmetric']
            manipulate_equation = params['manipulate_equation']

            train_ids = loaded_ids['train_ids']
            val_ids = loaded_ids['val_ids']
            test_ids = loaded_ids['test_ids']

            if data_path is None:
                return  # if no parameters passed to the constructor, just returns

            if dictionary is None:
                dictionary: Dictionary = Dictionary(self.PAD, self.UNK)

            self.dictionary = dictionary
            self.ilp_pad = 3
            self.include_asymmetric = include_asymmetric
            self.length_asymmetric = length_asymmetric

            if debug_length > 0:
                train_ids = train_ids[:debug_length]
                val_ids = val_ids[:debug_length]
                test_ids = test_ids[:debug_length]

            strain_ids = set(train_ids)
            sval_ids = set(val_ids)
            stest_ids = set(test_ids)

            # print('loading and tokenizing data')
            loaded_data: list = self.get_questions(data_path, stest_ids.union(sval_ids, strain_ids))

            # print('incorporating train corpus...')
            self.corpus_train = self.tokenize_data_ilp(loaded_data=loaded_data, fold_ids=train_ids,
                                                       sfold_ids=strain_ids,
                                                       max_falses_per_equation=max_falses_per_equation,
                                                       max_equations_per_problem=max_equations_per_problem,
                                                       ilp_out_path=ilp_path, include_gold_formula=True,
                                                       include_asymmetric=include_asymmetric,
                                                       length_asymmetric=length_asymmetric,
                                                       manipulate_equation=manipulate_equation)
            # print('incorporating validation corpus...')
            self.corpus_val = self.tokenize_data_ilp(loaded_data=loaded_data, fold_ids=val_ids, sfold_ids=sval_ids,
                                                     max_falses_per_equation=max_falses_per_equation,
                                                     max_equations_per_problem=max_equations_per_problem,
                                                     ilp_out_path=ilp_path, include_gold_formula=False,
                                                     include_asymmetric=include_asymmetric,
                                                     length_asymmetric=length_asymmetric,
                                                     manipulate_equation=manipulate_equation)

            # print('incorporating test corpus...')
            self.corpus_test = self.tokenize_data_ilp(loaded_data=loaded_data, fold_ids=test_ids, sfold_ids=stest_ids,
                                                      max_falses_per_equation=max_falses_per_equation,
                                                      max_equations_per_problem=max_equations_per_problem,
                                                      ilp_out_path=ilp_path, include_gold_formula=False,
                                                      include_asymmetric=include_asymmetric,
                                                      length_asymmetric=length_asymmetric,
                                                      manipulate_equation=manipulate_equation)

    def data_extraction(self, doc: Doc):
        numbers = []
        sentences = []
        tokens = []
        nr_propn_in_sentence = 0
        first_propn_pos = -1
        last_propn_pos = -1

        for idx_token, token in enumerate(doc):
            str_token = token.string.lower().strip()
            tokens.append(token)
            number = re.findall(nr_regex, str_token)
            if len(number) > 0:
                numbers.append(
                    {'number': cast_to_number(delete_trailing_zeros(number[0].replace(',', ''))),
                     'idx_token': idx_token,
                     'alias': 'n{}'.format(len(numbers) + 1), 'source': 'digit'})

            if token.pos_ == 'PROPN' and token.string[0].isupper():
                nr_propn_in_sentence += 1
                if first_propn_pos == -1:
                    first_propn_pos = idx_token

                last_propn_pos = idx_token

            # if more than 2 proper nouns in the sentence, then adds the number of proper nouns, this might solve
            # the situations where enumeration of people is given
            if str_token == '.' and nr_propn_in_sentence >= 2:
                propn_pos = int((last_propn_pos - first_propn_pos) / 2) + first_propn_pos
                numbers.append(
                    {'number': nr_propn_in_sentence,
                     'idx_token': propn_pos,
                     'alias': 'n{}'.format(len(numbers) + 1), 'source': 'enumeration'})

            if str_token == '.':
                nr_propn_in_sentence = 0
                first_propn_pos = -1

            if str_token in str_to_numbers_units:
                numbers_for_token = str_to_numbers_units[str_token]
                for curr_number in numbers_for_token:
                    numbers.append({'number': curr_number, 'idx_token': idx_token,
                                    'alias': 'n{}'.format(len(numbers) + 1),
                                    'source': 'str_token'})

            assert len(number) < 2

            if str_token in str_to_numbers:
                numbers.append({'number': str_to_numbers[str_token],
                                'idx_token': idx_token, 'alias': 'n{}'.format(len(numbers) + 1),
                                'source': 'str_token'})

        for sentence in doc.sents:
            sentences.append(sentence)
        return {'numbers': numbers, 'sentences': sentences, 'tokens': tokens}

    @abstractmethod
    def get_questions(self, path_questions: str, ids: set):
        """ Gets the questions from SingleEQ dataset """
        nr_to_qt = dict()

        with open(path_questions, 'r') as infile:
            res = json.load(infile)
        loaded_questions = list()
        nr_numbers = 0

        for curr_equation in res:
            question: Question = Question()
            question.id = curr_equation['iIndex']
            if question.id not in ids:
                continue

            # parses the equations and converts to tree and gets all the possible combinations
            doc = nlp(curr_equation['sQuestion'])
            data = self.data_extraction(doc)
            question.numbers = data['numbers']
            len_numbers = len(question.numbers)

            if len_numbers not in nr_to_qt:
                nr_to_qt[len_numbers] = 1
            else:
                nr_to_qt[len_numbers] = nr_to_qt[len_numbers] + 1

            if len_numbers > nr_numbers:
                nr_numbers = len_numbers
                # print('max numbers: ', nr_numbers)

            question.numbers_node = [Node(value=nr['number'], alias=nr['alias'], idx_token=nr['idx_token'],
                                          ntype='value', source=nr['source']) for nr in data['numbers']]
            question.solution = curr_equation['lSolutions'][0]

            formula = curr_equation['lEquations'][0].lower()
            question.formula = formula
            try:
                norm = normalize(formula)
                toks = tokenize(norm)
                tree = parse(toks)

                binary_tree: BinaryTree = load_into_binary_tree(tree)

                binary_tree.rearrange_tree()

                # generates all possible variations based on commutativity
                binary_tree_variations = get_variations(binary_tree)
            except Exception as e:
                # print('ERROR when processing: ', formula, ' assigning None to formula_trees')
                # print(e)
                binary_tree_variations = None

            # makes sure that all the variations get approximate the same answer:
            curr_answer = None
            if binary_tree_variations is not None:
                for curr_tree in binary_tree_variations:
                    if curr_answer is None:
                        curr_answer = curr_tree.evaluate()
                        new_answer = curr_tree.evaluate()
                    else:
                        new_answer = curr_tree.evaluate()

                    if not isclose(curr_answer, new_answer, rel_tol=1e-5, abs_tol=0.0):
                        print('ERROR IN THE EQUATION ', formula, ' SOME OF ALTERNATIVES IS DIFFERENT!!!!')
                        exit()

            question.formula_trees = binary_tree_variations

            question.question_sentences = data['sentences']
            question.question_tokens = data['tokens']
            question.orig_question = curr_equation['sQuestion']

            loaded_questions.append(question)

        return loaded_questions

    def tokenize_data_ilp(self, loaded_data: list, fold_ids: list, sfold_ids: set, max_falses_per_equation: int,
                          max_equations_per_problem: int, ilp_out_path: str, include_gold_formula: bool,
                          include_asymmetric: bool, length_asymmetric: int, manipulate_equation: bool):
        """Loads equations using the output produced by ALGES Integer Linear Programming (ILP) algorithm"""
        fold_data = [load_q for load_q in loaded_data if load_q.id in sfold_ids]
        id_dict_fold = dict([(load_q.id, load_q) for load_q in fold_data])

        content = []
        targets = []

        for idx, i in enumerate(fold_ids):
            question: Question = id_dict_fold[i]
            tokens_lowercase = [t.text.lower().strip() for t in question.question_tokens]
            self.dictionary.add(*tokens_lowercase)
            question_tok_ids = self.dictionary(*tokens_lowercase)
            question_tok_ids = np.asarray(question_tok_ids)

            to_format = 'q{:0' + str(self.ilp_pad) + 'd}.txt.out'
            ilp_file_path = os.path.join(ilp_out_path, to_format.format(question.id))

            gen_trees = get_ilp_trees_variations_last(question=question, ilp_out_path=ilp_file_path,
                                                      max_falses_per_equation=max_falses_per_equation,
                                                      max_equations_per_problem=max_equations_per_problem,
                                                      include_gold_formula=include_gold_formula,
                                                      include_asymmetric=include_asymmetric,
                                                      length_asymmetric=length_asymmetric,
                                                      manipulate_equation=manipulate_equation)

            loc_targets = []
            loc_content = []

            # for each of the generated trees, check whether it is correct one and set the respective label
            for gen_tree in gen_trees:
                loc_content.append({'tokens': question_tok_ids,
                                    'tree': gen_tree['tree'], 'solution': question.solution, 'id': question.id,
                                    'include_in_accuracy': gen_tree['include_in_accuracy']})

                if gen_tree['tree'] is not None:
                    gen_tree['tree'].print_alias = False

                loc_targets.append(gen_tree['label'])

            content.extend(loc_content)
            targets.extend(loc_targets)

        return {'content': content, 'targets': targets}

    def __str__(self):
        s = 'Corpus:\n'
        s += '\t%d different terms\n' % len(self.dictionary)
        s += '\tinstances: %d train, %d valid, %d test' % \
             (len(self.corpus_train), len(self.corpus_val), len(self.corpus_test))
        return s
