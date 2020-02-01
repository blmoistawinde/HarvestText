import numpy as np
import pandas as pd
from collections import defaultdict
from collections.abc import Iterable
from itertools import combinations

class SentDict(object):
    def __init__(self, docs=[], method="PMI",min_times=5, scale="None", pos_seeds=None,neg_seeds=None):
        super(SentDict, self).__init__()
        self.sent_dict = {}
        self.words = set()
        assert isinstance(pos_seeds, Iterable)
        assert isinstance(neg_seeds, Iterable)
        self.build_sent_dict(docs, method, min_times, scale, pos_seeds, neg_seeds)
    def __getitem__(self,key):
        return self.sent_dict[key]
    def set_pos_seeds(self, pos_seeds):
        self.pos_seeds = set(pos_seeds) & set(self.words)
    def set_neg_seed(self, neg_seeds):
        self.neg_seeds = set(neg_seeds) & set(self.words)
    def build_sent_dict(self ,docs=[], method="PMI",min_times=5, scale="None", pos_seeds=None,neg_seeds=None):
        self.doc_count = len(docs)
        self.method = method
        pos_seeds = set(pos_seeds)
        neg_seeds = set(neg_seeds)
        if self.doc_count > 0:
            if method == "PMI":
                self.co_occur, self.one_occur = self.get_word_stat(docs)
                self.words = set(word for word in self.one_occur if self.one_occur[word]>=min_times)
                if len(pos_seeds) > 0 or len(neg_seeds) > 0:     # 如果有新的输入，就更新种子词，否则默认已有（比如通过set已设定）
                    self.pos_seeds = (pos_seeds & self.words)
                    self.neg_seeds = (neg_seeds & self.words)
                if len(self.pos_seeds) > 0 or len(self.neg_seeds) > 0:
                    self.sent_dict = self.SO_PMI(self.words, scale)
                else:
                    raise Exception("你的文章中不包含种子词，SO-PMI算法无法执行")
            else:
                raise Exception("不支持的情感分析算法")
    def analyse_sent(self, words, avg):
        if self.method == "PMI":
            words = (set(words) & set(self.sent_dict))
            if avg:
                return sum(self.sent_dict[word] for word in words) / len(words) if len(words) > 0 else 0
            else:
                return [self.sent_dict[word] for word in words]
        else:
            raise Exception("不支持的情感分析算法")
        
    def get_word_stat(self, docs, co=True):
        co_occur = dict()               # 由于defaultdict太占内存，还是使用dict
        one_occur = dict()
        for doc in docs:
            for word in doc:
                if not word in one_occur:
                    one_occur[word] = 1
                else:
                    one_occur[word] += 1
                # 考虑自共现，否则如果一个负面词不与其他负面词共存，那么它就无法获得PMI，从而被认为是负面的，这不合情理
                if not (word,word) in co_occur:
                    co_occur[(word,word)] = 1
                else:
                    co_occur[(word,word)] += 1
            if co:
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
    def SO_PMI(self, words, scale="None"):
        ret = {}
        max0, min0 = 0, 0
        for word in words:
            tmp = sum(self.PMI(word,seed) for seed in self.pos_seeds) - \
                sum(self.PMI(word,seed) for seed in self.neg_seeds)
            max0, min0 = max(tmp,max0), min(tmp,min0)
            ret[word] = tmp
        if scale == "+-1":
            # 在正负两个区域分别做线性变换
            # 不采用统一线性变换2*(x-mid)/(max-min)的原因:
            # 要保留0作为中性情感的语义，否则当原来的最小值为0时，经过变换会变成-1
            for word, senti in ret.items():
                if senti > 0:      # 如果触发此条件，max0≥senti>0, 不用检查除数为0。下同
                    ret[word] /= max0
                elif senti < 0:
                    ret[word] /= (-min0)
        elif scale == "0-1":
            # 这里可以采用同一变换
            ret = {word:(senti-min0)/(max0-min0) for word, senti in ret.items()}
        return ret

if __name__ == "__main__":
    docs = [["武磊","威武","，","中超","第一","射手","太","棒","了","！"],
          ["武磊","强","，","中超","最","棒","球员"],
          ["郜林","不行","，","只会","抱怨","的","球员","注定","上限","了"],
          ["郜林","看来","不行","，","已经","到","上限","了"]]
    sent_dict = SentDict(docs,method="PMI",min_times=1,pos_seeds=["棒"],neg_seeds=["不行"])
    print("威武", sent_dict["威武"])
    print("球员", sent_dict["球员"])
    print("上限", sent_dict["上限"])
    print(sent_dict.analyse_sent(docs[0]))

