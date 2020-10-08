#coding=utf-8

import re
import time
import math
import numpy as np
import pandas as pd
from collections import defaultdict

def genSubstr(string, n):
    """
    Generate all substrings of max length n for string
    """
    length = len(string)
    res = [string[i: j] for i in range(0, length)
        for j in range(i + 1, min(i + n + 1, length + 1))]
    return res

def genSubparts(string):
    """
    Partition a string into all possible two parts, e.g.
    given "abcd", generate [("a", "bcd"), ("ab", "cd"), ("abc", "d")]
    For string of length 1, return empty list
    """
    length = len(string)
    res = [(string[0:i], string[i:]) for i in range(1, length)]
    return res

def entropyOfList(cnt_dict):
    length = sum(v for v in cnt_dict.values())
    return sum(-v/length*math.log(v/length) for v in cnt_dict.values())

def indexOfSortedSuffix(doc, max_word_len):
    """
    Treat a suffix as an index where the suffix begins.
    Then sort these indexes by the suffixes.
    """
    indexes = []
    length = len(doc)
    indexes = ((i,j) for i in range(0, length)
               for j in range(i + 1, min(i + 1 + max_word_len, length + 1)))
    return sorted(indexes, key=lambda i_j: doc[i_j[0]:i_j[1]])


class WordInfo(object):
    """
    Store information of each word, including its freqency, left neighbors and right neighbors
    """
    def __init__(self, text):
        super(WordInfo, self).__init__()
        self.text = text
        self.freq = 0.0
        self.left = defaultdict(int)
        self.right = defaultdict(int)
        self.aggregation = 0

    def update(self, left, right):
        """
        Increase frequency of this word, then append left/right neighbors
        @param left a single character on the left side of this word
        @param right as left is, but on the right side
        """
        self.freq += 1
        if left: self.left[left] += 1
        if right: self.right[right] += 1

    def compute(self, length):
        """
        Compute frequency and entropy of this word
        @param length length of the document for training to get words
        """
        self.freq /= length
        self.left = entropyOfList(self.left)
        self.right = entropyOfList(self.right)
        
    def computeAggregation(self, words_dict):
        """
        Compute aggregation of this word
        @param words_dict frequency dict of all candidate words
        """
        parts = genSubparts(self.text)
        if len(parts) > 0:
            self.aggregation = min(self.freq/words_dict[p1_p2[0]].freq/words_dict[p1_p2[1]].freq for p1_p2 in parts)



class WordDiscoverer(object):
    def __init__(self, doc, max_word_len=5, min_freq=0.00005, min_entropy=2.0, min_aggregation=50,
                 ent_threshold="both", mem_saving=False):
        super(WordDiscoverer, self).__init__()
        self.max_word_len = max_word_len
        self.min_freq = min_freq
        self.min_entropy = min_entropy
        self.min_aggregation = min_aggregation
        
        if mem_saving:
            self.word_infos = self.genWords(doc)
            # Filter out the results satisfy all the requirements
            if ent_threshold == "both":
                filter_func = lambda v: len(v.text) > 1 and v.aggregation > self.min_aggregation and\
                            v.freq > self.min_freq and v.left > self.min_entropy and v.right > self.min_entropy
            else:
                filter_func = lambda v: len(v.text) > 1 and v.aggregation > self.min_aggregation and\
                            v.freq > self.min_freq and (v.left+v.right)/2.0 > self.min_entropy
        else:
            # 对于太长的语料，因为每个候选词都保存了列表形式的left和right，导致内存容易爆炸，考虑优化算法。
            # 先根据简单的规则筛选词语（freq,agg）再重新扫描统计left,right
            # 因为需要两次扫描语料，所以速度会稍微慢一点
            self.word_infos = self.genWords2(doc)
            if ent_threshold == "both":
                filter_func = lambda v: v.left > self.min_entropy and v.right > self.min_entropy
            else:
                filter_func = lambda v: (v.left+v.right)/2.0 > self.min_entropy
        self.word_infos = list(filter(filter_func, self.word_infos))
        self.word_with_freq = [(w.text, w.freq) for w in self.word_infos]
        self.words = [w[0] for w in self.word_with_freq]
        # Result infomations, i.e., average data of all words
        word_count = float(len(self.word_with_freq))
        if word_count > 0:
            self.avg_len = sum([len(w.text) for w in self.word_infos])/word_count
            self.avg_freq = sum([w.freq for w in self.word_infos])/word_count
            self.avg_left_entropy = sum([w.left for w in self.word_infos])/word_count
            self.avg_right_entropy = sum([w.right for w in self.word_infos])/word_count
            self.avg_aggregation = sum([w.aggregation for w in self.word_infos])/word_count
        else:
            self.avg_len = 0
            self.avg_freq = 0
            self.avg_left_entropy = 0
            self.avg_right_entropy = 0
            self.avg_aggregation = 0

    def genWords(self, doc):
        """
        Generate all candidate words with their frequency/entropy/aggregation informations
        @param doc the document used for words generation
        """
        #pattern = re.compile('[\\s\\d,.<>/?:;\'\"[\\]{}()\\|~!@#$%^&*\\-_=+a-zA-Z，。《》、？：；“”‘’｛｝【】（）…￥！—┄－]+')
        # numbers preserved
        pattern = re.compile('[\\s,.<>/?:;\'\"[\\]{}()\\|~!@#$%^&*\\-_=+a-zA-Z，。《》、？：；“”‘’｛｝【】（）…￥！—┄－]+')
        doc = re.sub(pattern, ' ', doc)
        suffix_indexes = indexOfSortedSuffix(doc, self.max_word_len)
        word_cands = {}
        # compute frequency and neighbors
        for suf in suffix_indexes:
            word = doc[suf[0]:suf[1]]
            if " " in word : continue
            if word not in word_cands:
                word_cands[word] = WordInfo(word)
            word_cands[word].update(doc[suf[0] - 1:suf[0]], doc[suf[1]:suf[1] + 1])
            
        # compute probability and entropy
        length = len(doc)
        self.length = length
        for k in word_cands:
            word_cands[k].compute(length)
        # compute aggregation of words whose length > 1
        values = sorted(list(word_cands.values()), key=lambda x: len(x.text))
        for v in values:
            if len(v.text) == 1: continue
            v.computeAggregation(word_cands)
        return values

    def genWords2(self, doc):
        """
        Generate all candidate words with their frequency/entropy/aggregation informations
        @param doc the document used for words generation
        """
        #pattern = re.compile('[\\s\\d,.<>/?:;\'\"[\\]{}()\\|~!@#$%^&*\\-_=+a-zA-Z，。《》、？：；“”‘’｛｝【】（）…￥！—┄－]+')
        # numbers preserved
        pattern = re.compile('[\\s,.<>/?:;\'\"[\\]{}()\\|~!@#$%^&*\\-_=+a-zA-Z，。《》、？：；“”‘’｛｝【】（）…￥！—┄－]+')
        doc = re.sub(pattern, ' ', doc)
        suffix_indexes = indexOfSortedSuffix(doc, self.max_word_len)
        word_cands = {}
        # compute frequency and neighbors
        for suf in suffix_indexes:
            word = doc[suf[0]:suf[1]]
            if " " in word : continue
            if word not in word_cands:
                word_cands[word] = WordInfo(word)
            word_cands[word].freq += 1
            
        self.length = len(doc)
        for k in word_cands:
            word_cands[k].freq /= self.length
        values = word_cands.values()
        for v in values:
            if len(v.text) == 1: continue
            v.computeAggregation(word_cands)
        filter_func = lambda v: len(v.text) > 1 and v.aggregation > self.min_aggregation and\
                    v.freq > self.min_freq
        values = list(filter(filter_func, values))
        word_cands = {w:word_cands[w] for w in word_cands if word_cands[w] in values}
        for suf in suffix_indexes:
            word = doc[suf[0]:suf[1]]
            if word in word_cands:
                left, right = doc[suf[0] - 1:suf[0]], doc[suf[1]:suf[1] + 1]
                if left: word_cands[word].left[left] += 1
                if right: word_cands[word].right[right] += 1
        
        values = word_cands.values()
        for v in values:
            v.left = entropyOfList(v.left)
            v.right = entropyOfList(v.right)
        return values
    def get_df_info(self, ex_mentions, exclude_number=True):
        info = {"text":[],"freq":[],"left_ent":[],"right_ent":[],"agg":[]}
        for w in self.word_infos:
            if w.text in ex_mentions:
                continue
            if exclude_number and w.text.isdigit():
                continue
            info["text"].append(w.text)
            info["freq"].append(w.freq)
            info["left_ent"].append(w.left)
            info["right_ent"].append(w.right)
            info["agg"].append(w.aggregation)
        info = pd.DataFrame(info)
        info = info.set_index("text")
        # 词语质量评分
        info["score"] = np.log10(info["agg"])*info["freq"]*(info["left_ent"]+info["right_ent"])
        return info

if __name__ == '__main__':
   doc = '十四是十四四十是四十，，十四不是四十，，，，四十不是十四'
   doc = "恒大门前种两棵树都不至于这个比分"
   ws = WordDiscoverer(doc, max_word_len=2, min_aggregation=1.2, min_entropy=0.4)
   N = ws.length
   
   print("new words with freqency:")
   print(' '.join(['%s:%f'% (w,f) for (w,f) in ws.word_with_freq]))
   print("new words info:")
   print('\n'.join(['[%s] times:%d, left:%f, right:%f, aggregation:%f'%(w.text,w.freq*N,w.left,w.right,w.aggregation) for w in  ws.word_infos]))
   print('average len: ', ws.avg_len)
   print('average frequency: ', ws.avg_freq)
   print('average left entropy: ', ws.avg_left_entropy)
   print('average right entropy: ', ws.avg_right_entropy)
   print('average aggregation: ', ws.avg_aggregation)

        
    


