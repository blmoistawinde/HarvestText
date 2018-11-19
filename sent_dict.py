import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import combinations

class SentDict(object):
    def __init__(self, docs, method="PMI",min_times=5, pos_seeds=[],neg_seeds=[]):
        super(SentDict, self).__init__()
        self.doc_count = len(docs)
        if method == "PMI":
            self.co_occur, self.one_occur = self.get_word_stat(docs)
            self.words = [word for word in self.one_occur if self.one_occur[word]>=min_times]
            self.pos_seeds = pos_seeds
            self.neg_seeds = neg_seeds
            self.sent_dict = self.SO_PMI(self.words)
    def __getitem__(self,key):
        return self.sent_dict[key]
    def get_word_stat(self, docs):
        co_occur = dict()               # 由于defaultdict太占内存，还是使用dict
        one_occur = dict()
        for doc in docs:
            for word in doc:
                if not word in one_occur:
                    one_occur[word] = 1
                else:
                    one_occur[word] += 1
            for a,b in combinations(doc,2):
                if not (a,b) in co_occur:
                    co_occur[(a,b)] = 1
                    co_occur[(b,a)] = 1
                else:
                    co_occur[(a,b)] += 1
                    co_occur[(b,a)] += 1
        return co_occur,one_occur
    def PMI(self,w1,w2):
        if not((w1 in self.one_occur) and (w2 in self.one_occur)):
            raise Exception()
        if not (w1,w2) in self.co_occur:
            return 0
        c1, c2 = self.one_occur[w1], self.one_occur[w2]
        c3 = self.co_occur[(w1,w2)]
        return np.log2((c3*self.doc_count)/(c1*c2))
    def SO_PMI(self, words):
        ret = {}
        for word in words:
            ret[word] = sum(self.PMI(word,seed) for seed in self.pos_seeds) - \
                    sum(self.PMI(word,seed) for seed in self.neg_seeds)
        return ret
if __name__ == "__main__":
    docs = [["武磊","威武","，","中超","第一","射手","太","棒","了","！"],
          ["武磊","强","，","中超","最","棒","球员"],
          ["郜林","不行","，","只会","抱怨","的","球员","注定","上限","了"],
          ["郜林","看来","不行","，","已经","到","上限","了"]]
    sent_dict = SentDict(docs,min_times=1,pos_seeds=["棒"],neg_seeds=["不行"])
    print("威武", sent_dict["威武"])
    print("球员", sent_dict["球员"])
    print("上限", sent_dict["上限"])
    
