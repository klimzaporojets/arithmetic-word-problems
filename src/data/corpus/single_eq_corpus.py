from data.corpus.corpus import Corpus
from data.dictionary import Dictionary


class SingleEQCorpus(Corpus):
    def __init__(self, loaded_ids, dictionary:Dictionary, params):
        super().__init__(loaded_ids=loaded_ids, dictionary=dictionary, params=params)

    def __str__(self):
        s = 'SingleEQCorpus:\n'
        s += '\t%d different terms\n' % len(self.dictionary)
        s += '\tinstances: %d train, %d valid, %d test' % \
             (len(self.corpus_train), len(self.corpus_val), len(self.corpus_test))
        return s
