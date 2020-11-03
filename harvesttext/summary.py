import numpy as np
import scipy.special
from itertools import combinations
from .algorithms.utils import sent_sim_textrank, sent_sim_cos

class SummaryMixin:
    """
    文本摘要模块：
    - 基于textrank+MMR的无监督抽取式摘要方法
    """
    def get_summary(self, sents, topK=5, stopwords=None, with_importance=False, standard_name=True,
                    maxlen=None, avoid_repeat=False, sim_func='default'):
        '''使用Textrank算法得到文本中的关键句

        :param sents: str句子列表
        :param topK: 选取几个句子, 如果设置了maxlen，则优先考虑长度
        :param stopwords: 在算法中采用的停用词
        :param with_importance: 返回时是否包括算法得到的句子重要性
        :param standard_name: 如果有entity_mention_list的话，在算法中正规化实体名，一般有助于提升算法效果
        :param maxlen: 设置得到的摘要最长不超过多少字数，如果已经达到长度限制但未达到topK句也会停止
        :param avoid_repeat: 使用MMR principle惩罚与已经抽取的摘要重复的句子，避免重复
        :param sim_func: textrank使用的相似度量函数，默认为基于词重叠的函数（原论文），也可以是任意一个接受两个字符串列表参数的函数
        :return: 句子列表，或者with_importance=True时，（句子，分数）列表
        '''
        assert topK > 0
        import networkx as nx
        maxlen = float('inf') if maxlen is None else maxlen
        sim_func = sent_sim_textrank if sim_func == 'default' else sim_func
        # 使用standard_name,相似度可以基于实体链接的结果计算而更加准确
        sent_tokens = [self.seg(sent.strip(), standard_name=standard_name, stopwords=stopwords) for sent in sents]
        if self.language == "en":
            from pattern.en import lemma
            sent_tokens = [[lemma(wd) for wd in sent] for sent in sent_tokens]
        sent_tokens = [sent for sent in sent_tokens if len(sent) > 0]
        G = nx.Graph()
        for u, v in combinations(range(len(sent_tokens)), 2):
            G.add_edge(u, v, weight=sim_func(sent_tokens[u], sent_tokens[v]))

        try:
            pr = nx.pagerank_scipy(G)  # sometimes fail to converge
        except:
            pr = nx.pagerank_numpy(G)
        pr_sorted = sorted(pr.items(), key=lambda x: x[1], reverse=True)
        if not avoid_repeat:
            ret = []
            curr_len = 0
            for i, imp in pr_sorted[:topK]:
                curr_len += len(sents[i])
                if curr_len > maxlen: break
                ret.append((sents[i], imp) if with_importance else sents[i])
            return ret
        else:
            assert topK <= len(sent_tokens)
            ret = []
            curr_len = 0
            curr_sumy_words = []
            candidate_ids = list(range(len(sent_tokens)))
            i, imp = pr_sorted[0]
            curr_len += len(sents[i])
            if curr_len > maxlen:
                return ret
            ret.append((sents[i], imp) if with_importance else sents[i])
            curr_sumy_words.extend(sent_tokens[i])
            candidate_ids.remove(i)
            for iter in range(topK-1):
                importance = [pr[i] for i in candidate_ids]
                norm_importance = scipy.special.softmax(importance)
                redundancy = np.array([sent_sim_cos(curr_sumy_words, sent_tokens[i]) for i in candidate_ids])
                scores = 0.6*norm_importance - 0.4*redundancy
                id_in_cands = np.argmax(scores)
                i, imp = candidate_ids[id_in_cands], importance[id_in_cands]
                curr_len += len(sents[i])
                if curr_len > maxlen:
                    return ret
                ret.append((sents[i], imp) if with_importance else sents[i])
                curr_sumy_words.extend(sent_tokens[i])
                del candidate_ids[id_in_cands]
            return ret


