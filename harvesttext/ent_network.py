import networkx as nx
from itertools import combinations

class EntNetworkMixin:
    """
    实体网络模块：
    - 根据实体在文档中的共现关系
        - 建立全局社交网络
        - 建立以某一个实体为中心的社交网络
    """
    def build_entity_graph(self, docs, min_freq=0, inv_index={}, used_types=[]):
        G = nx.Graph()
        links = {}
        if len(inv_index) == 0:
            for i, sent in enumerate(docs):
                entities_info = self.entity_linking(sent)
                if len(used_types) == 0:
                    entities = set(entity for span, (entity, type0) in entities_info)
                else:
                    entities = set(entity for span, (entity, type0) in entities_info if type0[1:-1] in used_types)
                for u, v in combinations(entities, 2):
                    pair0 = tuple(sorted((u, v)))
                    if pair0 not in links:
                        links[pair0] = 1
                    else:
                        links[pair0] += 1
        else:  # 已经有倒排文档，可以更快速检索
            if len(used_types) == 0:
                entities = self.entity_type_dict.keys()
            else:
                entities = iter(entity for (entity, type0) in self.entity_type_dict.items() if type0 in used_types)
            for u, v in combinations(entities, 2):
                pair0 = tuple(sorted((u, v)))
                ids = inv_index[u] & inv_index[v]
                if len(ids) > 0:
                    links[pair0] = len(ids)
        for (u, v) in links:
            if links[(u, v)] >= min_freq:
                G.add_edge(u, v, weight=links[(u, v)])
        self.entity_graph = G
        return G

    def build_word_ego_graph(self, docs, word, standard_name=True, min_freq=0, other_min_freq=-1, stopwords=None):
        '''根据文本和指定限定词，获得以限定词为中心的各词语的关系。
        限定词可以是一个特定的方面（衣食住行这类文档），这样就可以从词语中心图中获得关于这个方面的简要信息

        :param docs: 文本的列表
        :param word: 限定词
        :param standard_name: 把所有实体的指称化为标准实体名
        :param stopwords: 需要过滤的停用词
        :param min_freq: 作为边加入到图中的与中心词最小共现次数，用于筛掉可能过多的边
        :param other_min_freq: 中心词以外词语关系的最小共现次数
        :return: G（networxX中的Graph）

        '''
        G = nx.Graph()
        links = {}
        if other_min_freq == -1:
            other_min_freq = min_freq
        for doc in docs:
            if stopwords:
                words = set(x for x in self.seg(doc, standard_name=standard_name) if x not in stopwords)
            else:
                words = self.seg(doc, standard_name=standard_name)
            if word in words:
                for u, v in combinations(words, 2):
                    pair0 = tuple(sorted((u, v)))
                    if pair0 not in links:
                        links[pair0] = 1
                    else:
                        links[pair0] += 1

        used_nodes = set([word])  # 关系对中涉及的词语必须与实体有关（>= min_freq）
        for (u, v) in links:
            w = links[(u, v)]
            if word in (u, v) and w >= min_freq:
                used_nodes.add(v if word == u else u)
                G.add_edge(u, v, weight=w)
            elif w >= other_min_freq:
                G.add_edge(u, v, weight=w)
        G = G.subgraph(used_nodes).copy()
        return G

    def build_entity_ego_graph(self, docs, word, min_freq=0, other_min_freq=-1, inv_index={}, used_types=[]):
        '''Entity only version of build_word_ego_graph()
        '''
        G = nx.Graph()
        links = {}
        if other_min_freq == -1:
            other_min_freq = min_freq
        if len(inv_index) != 0:
            related_docs = self.search_entity(word, docs, inv_index)
        else:
            related_docs = []
            for doc in docs:
                entities_info = self.entity_linking(doc)
                entities = [entity0 for [[l,r], (entity0,type0)] in entities_info]
                if word in entities:
                    related_docs.append(doc)

        for i, sent in enumerate(related_docs):
            entities_info = self.entity_linking(sent)
            if len(used_types) == 0:
                entities = set(entity for span, (entity, type0) in entities_info)
            else:
                entities = set(entity for span, (entity, type0) in entities_info if type0[1:-1] in used_types)
            for u, v in combinations(entities, 2):
                pair0 = tuple(sorted((u, v)))
                if pair0 not in links:
                    links[pair0] = 1
                else:
                    links[pair0] += 1

        used_nodes = set([word])  # 关系对中涉及的词语必须与实体有关（>= min_freq）
        for (u, v) in links:
            w = links[(u, v)]
            if word in (u, v) and w >= min_freq:
                used_nodes.add(v if word == u else u)
                G.add_edge(u, v, weight=w)
            elif w >= other_min_freq:
                G.add_edge(u, v, weight=w)
        G = G.subgraph(used_nodes).copy()
        return G