from .resources import get_qh_sent_dict
from .algorithms.sent_dict import SentDict

class SentimentMixin:
    """
    情感分析模块：
    - 基于SO-PMI的情感词典挖掘和情感分析算法
    """
    def build_sent_dict(self, sents, method="PMI", min_times=5, scale="None",
                        pos_seeds=None, neg_seeds=None, stopwords=None):
        '''利用种子词，构建情感词典

        :param sents: list of string, 一般建议为句子，是计算共现PMI的基本单元
        :param method: "PMI", 使用的算法，目前仅支持PMI
        :param min_times: int, 默认为5， 在所有句子中出现次数少于这个次数的词语将被过滤
        :param scale: {"None","0-1","+-1"}, 默认为"None"，否则将对情感值进行变换
            若为"0-1"，按照最大为1，最小为0进行线性伸缩，0.5未必是中性
            若为"+-1", 在正负区间内分别伸缩，保留0作为中性的语义
        :param pos_seeds: list of string, 积极种子词，如不填写将默认采用清华情感词典
        :param neg_seeds: list of string, 消极种子词，如不填写将默认采用清华情感词典
        :param stopwords: list of string, stopwords词，如不填写将不使用
        :return: sent_dict: dict,可以查询单个词语的情感值
        '''
        if pos_seeds is None and neg_seeds is None:
            sdict = get_qh_sent_dict()
            pos_seeds, neg_seeds = sdict["pos"], sdict["neg"]
        docs = [set(self.seg(sent)) for sent in sents]
        if not stopwords is None:
            stopwords = set(stopwords)
            for i in range(len(docs)):
                docs[i] = docs[i] - stopwords
            docs = list(filter(lambda x: len(x) > 0, docs))
        self.sent_dict = SentDict(docs, method, min_times, scale, pos_seeds, neg_seeds)
        return self.sent_dict.sent_dict

    def analyse_sent(self, sent, avg=True):
        """输入句子，输出其情感值，默认使用句子中，在情感词典中的词语的情感值的平均来计算

        :param sent: string, 句子
        :param avg: (default True) 是否使用平均值计算句子情感值
        :return: float情感值(if avg == True), 否则为词语情感值列表
        """
        return self.sent_dict.analyse_sent(self.seg(sent), avg)

