import re
import json
import time
import numpy as np
import pandas as pd
import networkx as nx
import community
from pypinyin import lazy_pinyin
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import FastText

class NERPEntityDiscover:
    def __init__(self, sent_words, type_entity_dict, entity_count, pop_words_cnt, word2id, id2word,
                 min_count=5, pinyin_tolerance=0, pinyin_adjlist=None, **kwargs):
        self.type_entity_dict = type_entity_dict
        self.entity_count = entity_count
        self.pinyin_adjlist = pinyin_adjlist
        self.word2id, self.id2word = word2id, id2word
        self.mentions = set(x[:x.rfind("_")] for x in self.word2id)
        self.mention_count = {x[:x.rfind("_")]:cnt for x, cnt in self.entity_count.items()}
        partition = {i: i for i, word in enumerate(self.id2word)}
        partition, pattern_entity2mentions = self.postprocessing(partition, pinyin_tolerance, pop_words_cnt)
        self.entity_mention_dict, self.entity_type_dict = self.organize(partition, pattern_entity2mentions)

    def get_pinyin_correct_candidates(self, word, tolerance):  # 默认最多容忍一个拼音的变化
        assert tolerance in [0, 1]
        pinyins = lazy_pinyin(word)
        tmp = pinyins[:]
        pinyin_cands = {tuple(pinyins)}
        if tolerance == 1:
            for i, pinyin in enumerate(pinyins):
                if pinyin in self.pinyin_adjlist:
                    pinyin_cands |= {tuple(tmp[:i] + [neibr] + tmp[i + 1:]) for neibr in self.pinyin_adjlist[pinyin]}
        pinyin_cands = pinyin_cands & set(self.pinyin_mention_dict.keys())
        mention_cands = set()
        for pinyin in pinyin_cands:
            mention_cands |= self.pinyin_mention_dict[pinyin]
        return list(mention_cands)

    def postprocessing(self, partition, pinyin_tolerance, pop_words_cnt):
        """应用模式修复一些小问题

        :return: partition, pattern_entity2mentions
        """
        # simple postfix like removing parenthesis
        # “+?” parttern for lazy match so that "新区" can be matched instead of match
        re_patterns = {
            "parenthesis": (None, re.compile(r"[\[{\(<#【（《](\S+?)[\]}\)>#】）》]")),
            "person_postfix": ({"人名"}, re.compile(r"^(\S+?)(哥|姐|先生|女士|小姐|同志|同学|老师|教授)$")),
            "district": ({"地名"}, re.compile(r"^(\S+?)(国|省|市|区|县|村|镇|古镇|新区|特区|自治区|特别行政区|帝国|王国|共和国)$")),
            "organization": ({"地名", "机构名"}, re.compile(r"^(\S+?)(厂|公司|有限公司|协会|基金会|俱乐部|队|国家队|集团|联盟)$")),
        }
        pattern_entity2mentions = defaultdict(set)
        if pinyin_tolerance is not None:
            self.pinyin_mention_dict = defaultdict(set)
            for entity_type in self.id2word:
                new_word = entity_type[:entity_type.rfind("_")]
                self.pinyin_mention_dict[tuple(lazy_pinyin(new_word))].add(new_word)

        for eid1, entity_type in enumerate(self.id2word):
            tmp = entity_type.rfind("_")
            entity, etype = entity_type[:tmp], entity_type[tmp + 1:]
            # pattern_matching
            for pname, (allow_types, pat) in re_patterns.items():
                if (allow_types is None or (etype in allow_types)) and re.match(pat, entity):
                    trim_entity = re.sub(pat, r"\1", entity)
                    entity2 = trim_entity + "_" + etype
                    if entity2 in self.word2id:
                        eid2 = self.word2id[entity2]
                        partition[eid1] = partition[eid2]
                    if (pname in ["district", "organization"]) and len(trim_entity) > 1:
                        if trim_entity in self.mentions or trim_entity in pop_words_cnt:
                            pattern_entity2mentions[entity_type].add(trim_entity)
                            if trim_entity not in self.mention_count:
                                self.mention_count[trim_entity] = pop_words_cnt[trim_entity]

            # pinyin recheck
            if pinyin_tolerance is not None:
                candidates = self.get_pinyin_correct_candidates(entity, pinyin_tolerance)
                for cand in candidates:
                    entity2 = cand + "_" + etype
                    if entity2 in self.word2id:
                        eid2 = self.word2id[entity2]
                        partition[eid1] = partition[eid2]

        return partition, pattern_entity2mentions

    def organize(self, partition, pattern_entity2mentions):
        """把聚类结果组织成输出格式，每个聚类簇以出现频次最高的一个mention作为entity
        entity名中依然包括词性，但是mention要去掉词性

        :return: entity_mention_dict, entity_type_dict
        """
        num_entities1 = max(partition.values()) + 1
        cluster_mentions = [set() for i in range(num_entities1)]
        cluster_entities = [("entity", 0) for i in range(num_entities1)]
        for wid, cid in partition.items():
            entity0 = self.id2word[wid]
            mention0 = entity0[:entity0.rfind("_")]
            mention_cnt = self.entity_count[entity0]
            cluster_mentions[cid].add(mention0)
            cluster_entity, curr_cnt = cluster_entities[cid]
            if mention_cnt > curr_cnt:
                cluster_entities[cid] = (entity0, mention_cnt)

        entity_mention_dict, entity_type_dict = defaultdict(set), {}
        for mentions0, entity_infos in zip(cluster_mentions, cluster_entities):
            if entity_infos[0] == "entity" or entity_infos[1] <= 0:
                continue
            entity0 = entity_infos[0]
            etype0 = entity0[entity0.rfind("_") + 1:]
            mentions_pattern = set() if entity0 not in pattern_entity2mentions else pattern_entity2mentions[entity0]
            entity_mention_dict[entity0] = mentions0 | mentions_pattern
            entity_type_dict[entity0] = etype0

        return entity_mention_dict, entity_type_dict


class NFLEntityDiscoverer(NERPEntityDiscover):
    def __init__(self, sent_words, type_entity_dict, entity_count, pop_words_cnt, word2id, id2word,
                 min_count=5, pinyin_tolerance=0, pinyin_adjlist=None,
                 emb_dim=50, ft_iters=20, use_subword=True, threshold=0.98,
                 min_n=1, max_n=4, **kwargs):
        super(NFLEntityDiscoverer, self).__init__(sent_words, type_entity_dict, entity_count, pop_words_cnt, word2id, id2word,
                                                  min_count, pinyin_tolerance, pinyin_adjlist, **kwargs)
        self.type_entity_dict = type_entity_dict
        self.entity_count = entity_count
        self.pinyin_adjlist = pinyin_adjlist
        self.mentions = set(x[:x.rfind("_")] for x in self.word2id)
        self.mention_count = {x[:x.rfind("_")]:cnt for x, cnt in self.entity_count.items()}
        self.emb_mat, self.word2id, self.id2word = self.train_emb(sent_words, word2id, id2word,
                                                                  emb_dim, min_count, ft_iters, use_subword,
                                                                  min_n, max_n)
        partition = self.clustering(threshold)
        partition, pattern_entity2mentions = self.postprocessing(partition, pinyin_tolerance, pop_words_cnt)
        self.entity_mention_dict, self.entity_type_dict = self.organize(partition, pattern_entity2mentions)

    def train_emb(self, sent_words, word2id, id2word, emb_dim, min_count, ft_iters, use_subword, min_n, max_n):
        """因为fasttext的词频筛选策略(>=5)，word2id和id2word会发生改变，但是要保持按照词频的排序

        :return: emb_mat, word2id, id2word
            - emb_mat: np.array [num_entities, emb_dim]
            - word2id
            - id2word
        """
        print("Training fasttext")
        model = FastText(sent_words, size=emb_dim, min_count=min_count,
                         iter=ft_iters, word_ngrams=int(use_subword), min_n=min_n, max_n=max_n)
        id2word = [wd for wd in id2word if wd in model.wv.vocab]
        word2id = {wd: i for (i, wd) in enumerate(id2word)}
        emb_mat = np.zeros((len(id2word), emb_dim))
        for i, wd in enumerate(id2word):
            emb_mat[i, :] = model.wv[wd]

        return emb_mat, word2id, id2word

    # clustering
    def clustering(self, threshold):
        """分不同词性的聚类

        :return: partition: dict {word_id: cluster_id}
        """
        print("Louvain clustering")
        partition = {}
        part_offset = 0
        for etype, ners in self.type_entity_dict.items():
            sub_id_mapping = [self.word2id[ner0] for ner0 in ners if ner0 in self.word2id]
            if len(sub_id_mapping) == 0:
                continue
            emb_mat_sub = self.emb_mat[sub_id_mapping, :]
            cos_sims = cosine_similarity(emb_mat_sub)
            cos_sims -= np.eye(len(emb_mat_sub))
            adj_mat = (cos_sims > threshold).astype(int)
            G = nx.from_numpy_array(adj_mat)
            partition_sub = community.best_partition(G)
            for sub_id, main_id in enumerate(sub_id_mapping):
                sub_part_id = partition_sub[sub_id]
                partition[main_id] = sub_part_id + part_offset
            part_offset += max(partition_sub.values()) + 1
        return partition

