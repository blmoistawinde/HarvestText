#coding=utf-8
import os
import re
import numpy as np
import pandas as pd
from itertools import combinations
import jieba
import jieba.posseg as pseg
from collections import defaultdict
from .word_discoverer import WordDiscoverer
from .sent_dict import SentDict
import logging

class HarvestText:
    def __init__(self,standard_name=False):
        self.standard_name = standard_name             # 是否使用连接到的实体名来替换原文
        self.entity_types = set()
        self.trie_root = {}
        self.entity_mention_dict = defaultdict(list)
        self.entity_type_dict = {}
        self.type_entity_mention_dict = defaultdict(dict)
        self.prepared = False
        self.sent_dict = None
        #self.check_overlap = True                       # 是否检查重叠实体（市长江大桥），开启的话可能会较慢
        # 因为只有"freq"策略能够做，所以目前设定：指定freq策略时就默认检查重叠,否则不检查
        self.linking_strategy = "None"                 # 将字面值链接到实体的策略，默认为选取字典序第一个
        self.entity_count = defaultdict(int)            # 用于'freq'策略
        self.latest_mention = dict()                    # 用于'latest'策略
    #
    # 实体分词模块
    #
    def build_trie(self, new_word, entity, entity_type):
        type0 = "#%s#" % entity_type
        if not type0 in self.entity_types:
            self.entity_types.add(type0)
        trie_node = self.trie_root
        for ch in new_word:
            if not ch in trie_node:
                trie_node[ch] = {}
            trie_node = trie_node[ch]
        if not 'leaf' in trie_node:
            trie_node['leaf'] = {(entity,type0)}
        else:
            trie_node['leaf'].add((entity,type0))
    def add_entities(self, entity_mention_dict, entity_type_dict=None):
        self.entity_mention_dict = entity_mention_dict
        if entity_type_dict:
            self.entity_type_dict = entity_type_dict
        else:
            self.entity_type_dict = {entity: "添加词" for entity in entity_mention_dict}
        type_entity_mention_dict = defaultdict(dict)
        for entity0,type0 in self.entity_type_dict.items():
            type_entity_mention_dict[type0][entity0] = entity_mention_dict[entity0]
        self.type_entity_mention_dict = type_entity_mention_dict
        self._add_entities(type_entity_mention_dict)
        
    def _add_entities(self,type_entity_mention_dict):
        for type0 in type_entity_mention_dict:
            entity_mention_dict0 = type_entity_mention_dict[type0]
            for entity0 in entity_mention_dict0:
                mentions = entity_mention_dict0[entity0]
                for mention0 in mentions:
                    self.build_trie(mention0,entity0,type0)
        self.prepare()
    def prepare(self):
        self.prepared = True
        for type0 in self.entity_types:
            jieba.add_word(type0, freq=10000, tag=type0[1:-1])
    def deprepare(self):
        self.prepared = False
        for type0 in self.entity_types:
            del jieba.dt.FREQ[type0]
            tag0 = type0[1:-1]
            if tag0 in jieba.dt.user_word_tag_tab:
                del jieba.dt.user_word_tag_tab[tag0]
            jieba.dt.total -= 10000
    def check_prepared(self):
        if not self.prepared:
            self.prepare()
    def dig_trie(self, sent, l):  # 返回实体右边界r,实体范围
        trie_node = self.trie_root
        for i in range(l, len(sent)):
            if sent[i] in trie_node:
                trie_node = trie_node[sent[i]]
            else:
                if "leaf" in trie_node:
                    return i, trie_node["leaf"]
                else:
                    return -1, set()  # -1表示未找到
        if "leaf" in trie_node:
            return len(sent), trie_node["leaf"]
        else:
            return -1, set()  # -1表示未找到
    def set_linking_strategy(self,strategy,lastest_mention=None,entity_freq=None,type_freq=None):
        """
        为实体链接设定一些简单策略，目前可选的有：
        'None','freq','latest','latest&freq'
        'None': 默认选择候选实体字典序第一个
        'freq': 对于单个字面值，选择其候选实体中之前出现最频繁的一个。
                对于多个重叠字面值，选择其中候选实体出现最频繁的一个进行连接【每个字面值已经确定唯一映射】。
        'latest': 对于单个字面值，如果在最近有可以确定的映射，就使用最近的映射。
        'latest'- 对于职称等作为代称的情况可能会比较有用。
        比如"经理"可能代指很多人，但是第一次提到的时候应该会包括姓氏。
        我们就可以记忆这次信息，在后面用来消歧。
        'freq' - 单字面值例：'市长'+{'A市长':5,'B市长':3} -> 'A市长'
        重叠字面值例，'xx市长江yy'+{'xx市长':5,'长江yy':3}+{'市长':'xx市长'}+{'长江':'长江yy'} -> 'xx市长'
        :param strategy: 可选 'None','freq','latest','latest&freq' 中的一个
        :param lastest_mention: dict,用于'latest',预设
        :param entity_freq: dict,用于'freq',预设某实体的优先级（词频）
        :param type_freq: dict,用于'freq',预设类别所有实体的优先级（词频）
        :return:
        """
        self.linking_strategy = strategy
        if "latest" in strategy:
            if lastest_mention:
                for surface0, entity0 in lastest_mention.items():
                    self.latest_mention[surface0] = entity0
        if "freq" in strategy:
            if entity_freq:
                for entity0, freq0 in entity_freq.items():
                    self.entity_count[entity0] += freq0
            if type_freq:
                for type0, freq0 in type_freq.items():
                    for entity0 in self.type_entity_mention_dict[type0].keys():
                        self.entity_count[entity0] += freq0

    def choose_from(self,surface0, entity_types):
        if self.linking_strategy == "None":
            linked_entity_type = list(entity_types)[0]
        else:
            linked_entity_type = None
            if "latest" in self.linking_strategy:
                if surface0 in self.latest_mention:
                    entity0 = self.latest_mention[surface0]
                    for entity_type0 in entity_types:
                        if entity0 in entity_type0:
                            linked_entity_type = entity_type0
                            break
            if linked_entity_type is None:
                if "freq" in self.linking_strategy:
                    candidate, cnt_cand = None, 0
                    for i, entity_type0 in enumerate(entity_types):
                        entity0, cnt0 = entity_type0[0], 0
                        if entity0 in self.entity_count:
                            cnt0 = self.entity_count[entity0]
                        if i == 0 or cnt0 > cnt_cand:
                            candidate, cnt_cand = entity_type0, cnt0
                        linked_entity_type = candidate
            if linked_entity_type is None:
                linked_entity_type = list(entity_types)[0]
        if "latest" in self.linking_strategy:
            self.latest_mention[surface0] = linked_entity_type[0]
        if "freq" in self.linking_strategy:
            self.entity_count[linked_entity_type[0]] += 1
        return linked_entity_type
    def entity_linking(self,sent):
        self.check_prepared()
        entities_info = []
        l = 0
        while l < len(sent):
            r, entity_types = self.dig_trie(sent, l)
            if r != -1:
                surface0 = sent[l:r]         #字面值
                entity_type0 = self.choose_from(surface0, entity_types)
                if "freq" in self.linking_strategy:         # 处理重叠消歧，目前只有freq策略能够做到
                    overlap_surface_entity_with_pos = {}       # 获得每个待链接字面值的“唯一”映射
                    overlap_surface_entity_with_pos[surface0] = ([l,r],entity_type0)
                    for ll in range(l+1,r):
                        rr, entity_types_2 = self.dig_trie(sent, ll)
                        if rr != -1:
                            surface0_2 = sent[ll:rr]  # 字面值
                            entity_type0_2 = self.choose_from(surface0_2, entity_types_2)
                            overlap_surface_entity_with_pos[surface0_2] = ([ll,rr], entity_type0_2)
                    # 再利用频率比较这些映射
                    candidate, cnt_cand = None, 0
                    for i, ([ll,rr], entity_type00) in enumerate(overlap_surface_entity_with_pos.values()):
                        entity00, cnt0 = entity_type00[0], 0
                        if entity00 in self.entity_count:
                            cnt0 = self.entity_count[entity00]
                        if i == 0 or cnt0 > cnt_cand:
                            candidate, cnt_cand = ([ll,rr], entity_type00), cnt0
                    entities_info.append(candidate)
                    l = candidate[0][1]
                else:
                    entities_info.append(([l, r], entity_type0))  # 字典树能根据键找到实体范围，选择则依然需要根据历史等优化
                    l = r
            else:
                l += 1
        return entities_info
    def decoref(self, sent, entities_info):
        left = 0
        processed_text = ""
        for (beg, end), (entity,e_type) in entities_info:
            # processed_text += sent[left:beg] + entity
            processed_text += sent[left:beg] + e_type
            left = end
        processed_text += sent[left:]
        return processed_text
    def posseg(self, sent, standard_name = False):
        self.standard_name = standard_name
        entities_info = self.entity_linking(sent)
        sent2 = self.decoref(sent, entities_info)
        result = []
        i = 0
        for word, flag in pseg.cut(sent2):
            if word in self.entity_types:
                if self.standard_name:
                    word = entities_info[i][1][0]     # 使用链接的实体
                else:
                    l,r = entities_info[i][0]        # 或使用原文
                    word = sent[l:r]
                i +=1
            result.append((word, flag))
        return result
    def seg(self, sent, return_sent=False, standard_name = False):
        self.standard_name = standard_name
        entities_info = self.entity_linking(sent)
        sent2 = self.decoref(sent, entities_info)
        result = []
        i = 0
        for word in jieba.cut(sent2):
            if word in self.entity_types:
                if self.standard_name:
                    word = entities_info[i][1][0]     # 使用链接的实体
                else:
                    l,r = entities_info[i][0]        # 或使用原文
                    word = sent[l:r]
                i +=1
            result.append(word)
        if return_sent:
            return " ".join(result)
        else:
            return result
    def cut_sentences(self,para):             # 分句
        para = re.sub('([。！？\?])([^”])', r"\1\n\2", para)  # 单字符断句符
        para = re.sub('(\.{6})([^”])', r"\1\n\2", para)  # 英文省略号
        para = re.sub('(\…{2})([^”])', r"\1\n\2", para)  # 中文省略号
        para = re.sub('(”)', '”\n', para)  # 把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
        para = para.rstrip()  # 段尾如果有多余的\n就去掉它
        # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
        return para.split("\n")
    def clear(self):
        self.deprepare()
        self.__init__()
    #
    # 新词发现模块
    #
    def word_discover(self, doc, threshold_seeds=[], auto_param = True,
                      excluding_types = [], excluding_words=[],     # 可以排除已经登录的某些种类的实体，或者某些指定词
                      max_word_len=5, min_freq=0.00005, min_entropy=1.4, min_aggregation=50,
                      ent_threshold="both",mem_saving=0):
        # 采用经验参数，此时后面的参数设置都无效
        if auto_param:               # 根据自己的几个实验确定的参数估计值，没什么科学性，但是应该能得到还行的结果
            length = len(doc)
            min_entropy = np.log(length) / 10
            min_freq = min(0.00005,20.0/length)
            min_aggregation = np.sqrt(length) / 15
            mem_saving = int(length>300000)
            # ent_threshold: 确定左右熵的阈值对双侧都要求"both"，或者只要左右平均值达到"avg"
            # 对于每句话都很极短的情况（如长度<8），经常出现在左右边界的词语可能难以被确定，这时ent_threshold建议设为"avg"
        try:
            ws = WordDiscoverer(doc, max_word_len, min_freq, min_entropy, min_aggregation, ent_threshold, mem_saving)
        except Exception as e:
            logging.log(logging.ERROR,str(e))
            info = {"text":[],"freq":[],"left_ent":[],"right_ent":[],"agg":[]}
            info = pd.DataFrame(info)
            info = info.set_index("text")
            return info
        
        if len(excluding_types) > 0:
            if "#" in list(excluding_types)[0]:              # 化为无‘#’标签
                excluding_types = [x[1:-1] for x in excluding_types]
            ex_mentions = [x for enty in self.entity_mention_dict
                             if enty in self.entity_type_dict and
                             self.entity_type_dict[enty] in excluding_types
                             for x in self.entity_mention_dict[enty]]
        else:
            ex_mentions = []
        ex_mentions += excluding_words
        
        info = ws.get_df_info(ex_mentions)
        
        # 利用种子词来确定筛选优质新词的标准，种子词中最低质量的词语将被保留（如果一开始就被找到的话）
        if len(threshold_seeds) > 0:
            min_score = 100000
            for seed in threshold_seeds:
                if seed in info.index:
                    min_score = min(min_score,info.loc[seed,"score"])
            if (min_score >= 100000):
                min_score = 0
            else:
                min_score *= 0.9          # 留一些宽松的区间
                info = info[info["score"] > min_score]
        return info
    def add_new_words(self,new_words):
        self.entity_types.add("#新词#")
        for word in new_words:
            self.build_trie(word,word,"新词")
            self.entity_mention_dict[word] = word
            self.entity_type_dict[word] = "新词"
            if word not in self.type_entity_mention_dict["新词"]:
                self.type_entity_mention_dict["新词"][word] = [word]
            else:
                self.type_entity_mention_dict["新词"][word].append(word)
        self.prepare()
    def add_new_mentions(self,entity_mention_dict):   # 添加链接到已有实体的新别称，一般在新词发现的基础上筛选得到
        for entity0 in entity_mention_dict:
            type0 = self.entity_type_dict[entity0]
            for mention0 in entity_mention_dict[entity0]:
                self.entity_mention_dict[entity0].append(mention0)
                self.build_trie(mention0, entity0, type0)
            self.type_entity_mention_dict[type0][entity0] = self.entity_mention_dict[entity0]
    def add_new_entity(self,entity0,mention0,type0):
        self.entity_type_dict[entity0] = type0
        if entity0 in self.entity_mention_dict:
            self.entity_mention_dict[entity0].append(mention0)
        else:
            self.entity_mention_dict[entity0] = [mention0]
        self.build_trie(mention0,entity0,type0)
        if entity0 not in self.type_entity_mention_dict[type0]:
            self.type_entity_mention_dict[type0][entity0] = [mention0]
        else:
            self.type_entity_mention_dict[type0][entity0].append(mention0)
    #
    # 情感分析模块
    #
    def build_sent_dict(self, sents, method="PMI",min_times=5, ft_size=100, ft_epochs=15, ft_window=5, pos_seeds=[],neg_seeds=[]):
        docs = [self.seg(sent) for sent in sents]
        self.sent_dict = SentDict(docs, method, min_times,ft_size,ft_epochs,ft_window, pos_seeds, neg_seeds)
        return self.sent_dict
    def analyse_sent(self, sent):
        return self.sent_dict.analyse_sent(self.seg(sent))
    #
    # 实体检索模块
    #
    def build_index(self, docs, with_entity = True, with_type = True):
        inv_index = defaultdict(set)
        for i, sent in enumerate(docs):
            entities_info = self.entity_linking(sent)
            for span, (entity,type0) in entities_info:
                if with_entity:
                    inv_index[entity].add(i)
                if with_type:
                    inv_index[type0].add(i)
        return inv_index
    def get_entity_counts(self,docs,inv_index,used_type=[]):
        if len(used_type) > 0:
            entities = iter(x for x in self.entity_type_dict
                            if self.entity_type_dict[x] in used_type)
        else:
            entities = self.entity_type_dict.keys()
        cnt = {enty:len(inv_index[enty]) for enty in entities if enty in inv_index}
        return cnt
    def search_entity(self,query,docs,inv_index):
        words = query.split()
        if len(words) > 0:
            ids = inv_index[words[0]]
            for word in words[1:]:
                ids = ids & inv_index[word]
            np_docs = np.array(docs)[list(ids)]
            return np_docs.tolist()
        else:
            return []
    #
    # 文本摘要模块
    #
    def get_summary(self, docs, topK=5, with_importance = False, standard_name = True):
        import networkx as nx
        def sent_sim1(words1, words2):
            return (len(set(words1) & set(words2))) / (np.log2(len(words1)) + np.log2(len(words2)))

        # 使用standard_name,相似度可以基于实体链接的结果计算而更加准确
        sents = [self.seg(doc,standard_name=standard_name) for doc in docs]
        G = nx.Graph()
        for u, v in combinations(range(len(sents)), 2):
            G.add_edge(u, v, weight=sent_sim1(sents[u], sents[v]))

        pr = nx.pagerank_scipy(G)
        pr_sorted = sorted(pr.items(), key=lambda x: x[1], reverse=True)
        if with_importance:
            return [(docs[i],imp) for i, imp in pr_sorted[:topK]]
        else:
            return [docs[i] for i, rank in pr_sorted[:topK]]
    #
    # 实体网络模块
    #
    def build_entity_graph(self, docs,inv_index={},used_types=[]):
        import networkx as nx
        G = nx.Graph()
        links = {}
        if len(inv_index) == 0:
            for i, sent in enumerate(docs):
                entities_info = self.entity_linking(sent)
                if len(used_types) == 0:
                    entities = set(entity for span, (entity,type0) in entities_info)
                else:
                    entities = set(entity for span, (entity, type0) in entities_info if type0[1:-1] in used_types)
                for u,v in combinations(entities,2):
                    pair0 = tuple(sorted((u,v)))
                    if pair0 not in links:
                        links[pair0] = 1
                    else:
                        links[pair0] += 1
        else:                                        # 已经有倒排文档，可以更快速检索
            if len(used_types) == 0:
                entities = self.entity_type_dict.keys()
            else:
                entities = iter(entity for (entity, type0) in self.entity_type_dict.items() if type0 in used_types)
            for u, v in combinations(entities, 2):
                pair0 = tuple(sorted((u, v)))
                ids = inv_index[u] & inv_index[v]
                if len(ids) > 0:
                    links[pair0] = len(ids)
        for (u,v) in links:
            G.add_edge(u,v,weight=links[(u,v)])
        self.entity_graph = G
        return G
