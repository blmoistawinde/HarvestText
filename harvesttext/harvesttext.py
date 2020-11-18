# coding=utf-8
import os
import re
import json
import numpy as np
import pandas as pd
import html
import urllib
import jieba
import jieba.posseg as pseg
import w3lib.html
import logging
import warnings
from tqdm import tqdm
from pypinyin import lazy_pinyin, pinyin
from opencc import OpenCC
from collections import defaultdict
from .ent_network import EntNetworkMixin
from .ent_retrieve import EntRetrieveMixin
from .parsing import ParsingMixin
from .sentiment import SentimentMixin
from .summary import SummaryMixin
from .word_discover import WordDiscoverMixin
from .resources import get_baidu_stopwords

class HarvestText(EntNetworkMixin, EntRetrieveMixin, ParsingMixin, SentimentMixin, SummaryMixin, WordDiscoverMixin):
    """
    主模块：
    - 主要保留了与实体分词、分句，预处理相关的代码
    - 还有存取、状态管理等基础代码
    - 其他功能在各个mixin里面
    - 主模块的功能是会被各个子模块最频繁调用的，也体现了本库以实体为核心，基于实体展开分析或改进算法的理念
    """
    def __init__(self, standard_name=False, language='zh_CN'):
        self.standard_name = standard_name  # 是否使用连接到的实体名来替换原文
        self.entity_types = set()
        self.trie_root = {}
        self.entity_mention_dict = defaultdict(set)
        self.entity_type_dict = {}
        self.type_entity_mention_dict = defaultdict(dict)
        self.pinyin_mention_dict = defaultdict(set)
        self.mentions = set()
        self.prepared = False
        self.hanlp_prepared = False
        self.sent_dict = None
        # self.check_overlap = True                       # 是否检查重叠实体（市长江大桥），开启的话可能会较慢
        # 因为只有"freq"策略能够做，所以目前设定：指定freq策略时就默认检查重叠,否则不检查
        self.linking_strategy = "None"  # 将字面值链接到实体的策略，默认为选取字典序第一个
        self.entity_count = defaultdict(int)  # 用于'freq'策略
        self.latest_mention = dict()  # 用于'latest'策略
        pwd = os.path.abspath(os.path.dirname(__file__))
        with open(pwd + "/resources/pinyin_adjlist.json", "r", encoding="utf-8") as f:
            self.pinyin_adjlist = json.load(f)
        self.language = language
        if language == "en":
            import nltk
            try:
                nltk.data.find('taggers/averaged_perceptron_tagger')
            except:
                nltk.download('averaged_perceptron_tagger')
            try:
                nltk.data.find('taggers/universal_tagset')
            except:
                nltk.download('universal_tagset')
            try:
                nltk.data.find('tokenizers/punkt')
            except:
                nltk.download('punkt')
            

    def build_trie(self, new_word, entity, entity_type):
        type0 = "#%s#" % entity_type
        if not type0 in self.entity_types:
            punct_regex = r"[、！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏!\"\#$%&\'\(\)\*\+,-\./:;<=>?@\[\\\]\^_`{\|}~]"
            matched = re.search(punct_regex, entity_type, re.MULTILINE | re.UNICODE)
            if matched:
                punct0 = matched.group()
                raise Exception("Your type input '{}' includes punctuation '{}', please remove them first".format(entity_type,punct0))
            self.entity_types.add(type0)
            self.prepared = False
            self.hanlp_prepared = False
        self.mentions.add(new_word)
        self.pinyin_mention_dict[tuple(lazy_pinyin(new_word))].add(new_word)

        trie_node = self.trie_root
        for ch in new_word:
            if not ch in trie_node:
                trie_node[ch] = {}
            trie_node = trie_node[ch]
        if not 'leaf' in trie_node:
            trie_node['leaf'] = {(entity, type0)}
        else:
            for (entity_orig, type_orig) in trie_node['leaf'].copy():
                if entity_orig == entity:           # 不允许同一实体有不同类型
                    trie_node['leaf'].remove((entity_orig, type_orig))
            trie_node['leaf'].add((entity, type0))

    def remove_mention(self, mention):
        trie_node = self.trie_root
        for ch in mention:
            if ch in trie_node:
                trie_node = trie_node[ch]
            else:
                return
        if not 'leaf' in trie_node:
            return
        else:
            del trie_node['leaf']

    def remove_entity(self, entity):
        mentions = self.entity_mention_dict[entity]
        for mention0 in mentions:
            trie_node = self.trie_root
            for ch in mention0:
                if ch in trie_node:
                    trie_node = trie_node[ch]
                else:
                    continue
            if not 'leaf' in trie_node:
                continue
            else:
                for (entity0, type0) in trie_node['leaf'].copy():
                    if entity0 == entity:
                        trie_node["leaf"].remove((entity0, type0))
                        break

    def _add_entities(self, type_entity_mention_dict):
        for type0 in type_entity_mention_dict:
            entity_mention_dict0 = type_entity_mention_dict[type0]
            for entity0 in entity_mention_dict0:
                mentions = entity_mention_dict0[entity0]
                for mention0 in mentions:
                    self.build_trie(mention0, entity0, type0)
        self.prepare()

    def add_entities(self, entity_mention_dict=None, entity_type_dict=None, override=False, load_path=None):
        '''登录的实体信息到ht，或者从save_entities保存的文件中读取（如果指定了load_path）

        :param entity_mention_dict: dict, {entity:[mentions]}格式，
        :param entity_type_dict: dict, {entity:entity_type}格式，
        :param override: bool, 是否覆盖已登录实体，默认False
        :param load_path: str, 要读取的文件路径（默认不使用）
        :return: None
        '''
        if load_path:
            self.load_entities(load_path, override)

        if override:
            self.clear()

        if entity_mention_dict is None and entity_type_dict is None:
            return

        if entity_mention_dict is None:         # 用实体名直接作为默认指称
            entity_mention_dict = dict(
                (entity0, {entity0}) for entity0 in entity_type_dict)
        else:
            entity_mention_dict = dict(
                (entity0, set(mentions0)) for (entity0, mentions0) in entity_mention_dict.items())
        if len(self.entity_mention_dict) == 0:
            self.entity_mention_dict = entity_mention_dict
        else:
            for entity, mentions in entity_type_dict.items():
                if entity in self.entity_mention_dict:
                    self.entity_mention_dict[entity] |= entity_mention_dict[entity]
                else:
                    self.entity_mention_dict[entity] = entity_mention_dict[entity]


        if entity_type_dict is None:
            entity_type_dict = {entity: "添加词" for entity in self.entity_mention_dict}
        if len(self.entity_type_dict) == 0:
            self.entity_type_dict = entity_type_dict
        else:
            for entity, type0 in entity_type_dict.items():
                if entity in self.entity_type_dict and type0 != self.entity_type_dict[entity]:
                    # 不允许同一实体有不同类型
                    warnings.warn("You've added an entity twice with different types, the later type will be used.")
                self.entity_type_dict[entity] = type0

        # 两个dict不对齐的情况下，以添加词作为默认词性
        for entity in self.entity_mention_dict:
            if entity not in self.entity_type_dict:
                self.entity_type_dict[entity] = "添加词"


        type_entity_mention_dict = defaultdict(dict)
        for entity0, type0 in self.entity_type_dict.items():
            if entity0 in self.entity_mention_dict:
                type_entity_mention_dict[type0][entity0] = self.entity_mention_dict[entity0]
        self.type_entity_mention_dict = type_entity_mention_dict
        self._add_entities(type_entity_mention_dict)

    def add_typed_words(self, type_word_dict):
        entity_type_dict = dict()
        for type0 in type_word_dict:
            for word in type_word_dict[type0]:
                entity_type_dict[word] = type0
        entity_mention_dict = dict(
            (entity0, set([entity0])) for entity0 in entity_type_dict.keys())
        self.entity_type_dict = entity_type_dict
        self.entity_mention_dict = entity_mention_dict

        type_entity_mention_dict = defaultdict(dict)
        for entity0, type0 in self.entity_type_dict.items():
            if entity0 in entity_mention_dict:
                type_entity_mention_dict[type0][entity0] = entity_mention_dict[entity0]
        self.type_entity_mention_dict = type_entity_mention_dict
        self._add_entities(type_entity_mention_dict)

    def add_new_words(self, new_words):
        for word in new_words:
            self.build_trie(word, word, "新词")
            self.entity_mention_dict[word] = set([word])
            self.entity_type_dict[word] = "新词"
            if word not in self.type_entity_mention_dict["新词"]:
                self.type_entity_mention_dict["新词"][word] = set([word])
            else:
                self.type_entity_mention_dict["新词"][word].add(word)
        self.check_prepared()

    def add_new_mentions(self, entity_mention_dict):  # 添加链接到已有实体的新别称，一般在新词发现的基础上筛选得到
        for entity0 in entity_mention_dict:
            type0 = self.entity_type_dict[entity0]
            for mention0 in entity_mention_dict[entity0]:
                self.entity_mention_dict[entity0].add(mention0)
                self.build_trie(mention0, entity0, type0)
            self.type_entity_mention_dict[type0][entity0] = self.entity_mention_dict[entity0]
        self.check_prepared()

    def add_new_entity(self, entity0, mention0=None, type0="添加词"):
        if mention0 is None:
            mention0 = entity0
        self.entity_type_dict[entity0] = type0
        if entity0 in self.entity_mention_dict:
            self.entity_mention_dict[entity0].add(mention0)
        else:
            self.entity_mention_dict[entity0] = set([mention0])
        self.build_trie(mention0, entity0, type0)
        if entity0 not in self.type_entity_mention_dict[type0]:
            self.type_entity_mention_dict[type0][entity0] = set([mention0])
        else:
            self.type_entity_mention_dict[type0][entity0].add(mention0)
        self.check_prepared()

    def prepare(self):
        self.prepared = True
        for type0 in self.entity_types:
            tag0 = "n"
            if "人名" in type0:
                tag0 = "nr"
            elif "地名" in type0:
                tag0 = "ns"
            elif "机构" in type0:
                tag0 = "nt"
            elif "其他专名" in type0:
                tag0 = "nz"
            jieba.add_word(type0, freq = 10000, tag=tag0)
    def hanlp_prepare(self):
        from pyhanlp import HanLP, JClass
        CustomDictionary = JClass("com.hankcs.hanlp.dictionary.CustomDictionary")
        StandardTokenizer = JClass("com.hankcs.hanlp.tokenizer.NLPTokenizer")

        self.hanlp_prepared = True
        for type0 in self.entity_types:
            tag0 = "n"
            if "人名" in type0:
                tag0 = "nr"
            elif "地名" in type0:
                tag0 = "ns"
            elif "机构" in type0:
                tag0 = "nt"
            elif "其他专名" in type0:
                tag0 = "nz"
            CustomDictionary.insert(type0, "%s 1000" % (tag0))  # 动态增加
        StandardTokenizer.ANALYZER.enableCustomDictionaryForcing(True)

    def deprepare(self):
        self.prepared = False
        self.hanlp_prepared = False
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
        # 需要记录：如果已经找到结果后还继续向前，但是在前面反而没有结果时，回溯寻找之前的记录
        # 例如：有mention("料酒"，"料酒 （焯水用）"), 在字符"料酒 花椒"中匹配时，已经经过"料酒"，但却会因为空格继续向前，最后错过结果
        records = []
        for i in range(l, len(sent)):
            if sent[i] in trie_node:
                trie_node = trie_node[sent[i]]
            else:
                break
            if "leaf" in trie_node:
                records.append((i + 1, trie_node["leaf"]))
        if len(records) > 0:
            return records[-1]
        else:
            return -1, set()  # -1表示未找到

    def search_word_trie(self, word, tolerance=1):
        """

        :param word:
        :param tolerance:
        :return:
        """
        results = set()
        def _visit(_trie, _word, _tolerance, _mention):
            if len(_word) > 0:
                ch = _word[0]
                if ch in _trie:
                    _visit(_trie[ch], _word[1:], _tolerance, _mention+ch)
                if _tolerance:
                    for ch in _trie:
                        if ch not in [_word[0], 'leaf']:
                            _visit(_trie[ch], _word[1:], _tolerance - 1, _mention+ch)
            else:
                if 'leaf' in _trie:
                    results.add(_mention)
        _visit(self.trie_root, word, tolerance,"")
        return list(results)

    def set_linking_strategy(self, strategy, lastest_mention=None, entity_freq=None, type_freq=None):
        """为实体链接设定一些简单策略，目前可选的有：'None','freq','latest','latest&freq'

            'None': 默认选择候选实体字典序第一个

            'freq': 对于单个字面值，选择其候选实体中之前出现最频繁的一个。对于多个重叠字面值，选择其中候选实体出现最频繁的一个进行连接【每个字面值已经确定唯一映射】。

            'latest': 对于单个字面值，如果在最近有可以确定的映射，就使用最近的映射。

            'latest'- 对于职称等作为代称的情况可能会比较有用。

            比如"经理"可能代指很多人，但是第一次提到的时候应该会包括姓氏。我们就可以记忆这次信息，在后面用来消歧。

            'freq' - 单字面值例：'市长'+{'A市长':5,'B市长':3} -> 'A市长'

            重叠字面值例，'xx市长江yy'+{'xx市长':5,'长江yy':3}+{'市长':'xx市长'}+{'长江':'长江yy'} -> 'xx市长'

        :param strategy: 可选 'None','freq','latest','latest&freq' 中的一个
        :param lastest_mention: dict,用于'latest',预设
        :param entity_freq: dict,用于'freq',预设某实体的优先级（词频）
        :param type_freq: dict,用于'freq',预设类别所有实体的优先级（词频）

        :return None

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
    def _link_record(self, surface0, entity0):
        if "latest" in self.linking_strategy:
            for surface0 in self.entity_mention_dict[entity0]:
                self.latest_mention[surface0] = entity0
        if "freq" in self.linking_strategy:
            self.entity_count[entity0] += 1

    def choose_from(self, surface0, entity_types):
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
        self._link_record(surface0, linked_entity_type[0])

        return linked_entity_type

    def mention2entity(self, mention):
        '''找到单个指称对应的实体

        :param mention:  指称
        :return: 如果存在对应实体，则返回（实体,类型），否则返回None, None
        '''
        for l in range(len(mention)-1):
            r, entity_types = self.dig_trie(mention, l)
            if r != -1 and r<=len(mention):
                surface0 = mention[0:r]  # 字面值
                (entity0, type0) = self.choose_from(surface0, entity_types)
                return entity0, type0
        return None, None

    def get_pinyin_correct_candidates(self, word, tolerance=1):  # 默认最多容忍一个拼音的变化
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

    def choose_from_multi_mentions(self, mention_cands, sent=""):
        surface0 = mention_cands[0]
        entity0, type0 = self.mention2entity(surface0)
        self._link_record(surface0, entity0)
        return entity0, type0

    def _entity_recheck(self, sent, entities_info, pinyin_tolerance, char_tolerance):
        sent2 = self.decoref(sent, entities_info)
        for word, flag in pseg.cut(sent2):
            if flag.startswith("n"):  # 对于名词，再检查是否有误差范围内匹配的其他指称
                entity0, type0 = None, None
                mention_cands = []
                if pinyin_tolerance is not None:
                    mention_cands += self.get_pinyin_correct_candidates(word, pinyin_tolerance)
                if char_tolerance is not None:
                    mention_cands += self.search_word_trie(word, char_tolerance)

                if len(mention_cands) > 0:
                    entity0, type0 = self.choose_from_multi_mentions(mention_cands, sent)
                if entity0:
                    l = sent.find(word)
                    entities_info.append([(l,l+len(word)),(entity0, type0)])

    def _entity_linking(self, sent, pinyin_tolerance=None, char_tolerance=None, keep_all=False):
        entities_info = []
        l = 0
        while l < len(sent):
            r, entity_types = self.dig_trie(sent, l)
            if r != -1 and r <= len(sent):
                surface0 = sent[l:r]  # 字面值
                if not keep_all:
                    entity_type0 = self.choose_from(surface0, entity_types)
                    if "freq" in self.linking_strategy:  # 处理重叠消歧，目前只有freq策略能够做到
                        overlap_surface_entity_with_pos = {}  # 获得每个待链接字面值的“唯一”映射
                        overlap_surface_entity_with_pos[surface0] = ([l, r], entity_type0)
                        for ll in range(l + 1, r):
                            rr, entity_types_2 = self.dig_trie(sent, ll)
                            if rr != -1 and rr <= len(sent):
                                surface0_2 = sent[ll:rr]  # 字面值
                                entity_type0_2 = self.choose_from(surface0_2, entity_types_2)
                                overlap_surface_entity_with_pos[surface0_2] = ([ll, rr], entity_type0_2)
                        # 再利用频率比较这些映射
                        candidate, cnt_cand = None, 0
                        for i, ([ll, rr], entity_type00) in enumerate(overlap_surface_entity_with_pos.values()):
                            entity00, cnt0 = entity_type00[0], 0
                            if entity00 in self.entity_count:
                                cnt0 = self.entity_count[entity00]
                            if i == 0 or cnt0 > cnt_cand:
                                candidate, cnt_cand = ([ll, rr], entity_type00), cnt0
                        entities_info.append(candidate)
                        l = candidate[0][1]
                    else:
                        entities_info.append(([l, r], entity_type0))  # 字典树能根据键找到实体范围，选择则依然需要根据历史等优化
                        l = r
                else:
                    entities_info.append(([l, r], entity_types))  # 字典树能根据键找到实体范围，选择则依然需要根据历史等优化
                    l = r
            else:
                l += 1
        return entities_info

    def entity_linking(self, sent, pinyin_tolerance=None, char_tolerance=None, keep_all=False, with_ch_pos=False):
        '''

        :param sent: 句子/文本
        :param pinyin_tolerance: {None, 0, 1} 搜索拼音相同(取0时)或者差别只有一个(取1时)的候选词链接到现有实体，默认不使用(None)
        :param char_tolerance: {None, 1} 搜索字符只差1个的候选词(取1时)链接到现有实体，默认不使用(None)
        :param keep_all: if True, keep all the possibilities of linked entities
        :param with_ch_pos: if True, also returns ch_pos
        :return: entities_info：依存弧,列表中的列表。
            if not keep_all: [([l, r], (entity, type)) for each linked mention m]
            else: [( [l, r], set((entity, type) for each possible entity of m) ) for each linked mention m]
            ch_pos: 每个字符对应词语的词性标注（不考虑登录的实体，可用来过滤实体，比如去掉都由名词组成的实体，有可能是错误链接）

        '''
        self.check_prepared()
        entities_info = self._entity_linking(sent, pinyin_tolerance, char_tolerance, keep_all)
        if (not keep_all) and (pinyin_tolerance is not None or char_tolerance is not None):
            self._entity_recheck(sent, entities_info, pinyin_tolerance, char_tolerance)
        if with_ch_pos:
            ch_pos = []
            for word, pos in pseg.cut(sent):
                ch_pos.extend([pos] * len(word))
            return entities_info, ch_pos
        else:
            return entities_info

    def get_linking_mention_candidates(self, sent, pinyin_tolerance=None, char_tolerance=None):
        mention_cands = defaultdict(list)
        cut_result = []
        self.check_prepared()
        entities_info = self._entity_linking(sent, pinyin_tolerance, char_tolerance)
        sent2 = self.decoref(sent, entities_info)
        l = 0
        i = 0
        for word, flag in pseg.cut(sent2):
            if word in self.entity_types:
                word = entities_info[i][1][0]  # 使用链接的实体
                i += 1
            cut_result.append(word)
            if flag.startswith("n"):  # 对于名词，再检查是否有误差范围内匹配的其他指称
                cands = []
                if pinyin_tolerance:
                    cands += self.get_pinyin_correct_candidates(word)
                if char_tolerance:
                    cands += self.search_word_trie(word)
                if len(cands) > 0:
                    mention_cands[(l, l + len(word))] = set(cands)
            l += len(word)
        sent2 =  "".join(cut_result)
        return sent2, mention_cands

    def decoref(self, sent, entities_info):
        left = 0
        processed_text = ""
        for (beg, end), (entity, e_type) in entities_info:
            # processed_text += sent[left:beg] + entity
            processed_text += sent[left:beg] + e_type
            left = end
        processed_text += sent[left:]
        return processed_text

    def posseg(self, sent, standard_name=False, stopwords=None):
        if self.language == 'en':
            from nltk import word_tokenize, pos_tag
            stopwords = set() if stopwords is None else stopwords
            tokens = [word for word in word_tokenize(sent) if word not in stopwords]
            return pos_tag(tokens, tagset='universal')
        else:
            self.standard_name = standard_name
            entities_info = self.entity_linking(sent)
            sent2 = self.decoref(sent, entities_info)
            result = []
            i = 0
            for word, flag in pseg.cut(sent2):
                if word in self.entity_types:
                    if self.standard_name:
                        word = entities_info[i][1][0]  # 使用链接的实体
                    else:
                        l, r = entities_info[i][0]  # 或使用原文
                        word = sent[l:r]
                    flag = entities_info[i][1][1][1:-1]
                    i += 1
                else:
                    if stopwords and word in stopwords:
                        continue
                result.append((word, flag))
            return result
    def seg(self, sent, standard_name=False, stopwords=None, return_sent=False):
        if self.language == "en":
            from nltk.tokenize import word_tokenize
            stopwords = set() if stopwords is None else stopwords
            words = [x for x in word_tokenize(sent) if not x in stopwords]
            return " ".join(words) if return_sent else words
        else:
            self.standard_name = standard_name
            entities_info = self.entity_linking(sent)
            sent2 = self.decoref(sent, entities_info)
            result = []
            i = 0
            for word in jieba.cut(sent2):
                if word in self.entity_types:
                    if self.standard_name:
                        word = entities_info[i][1][0]  # 使用链接的实体
                    else:
                        l, r = entities_info[i][0]  # 或使用原文
                        word = sent[l:r]
                    i += 1
                else:
                    if stopwords and word in stopwords:
                        continue
                result.append(word)
            if return_sent:
                return " ".join(result)
            else:
                return result
    def save_entity_info(self, save_path='./ht_entities.txt', entity_mention_dict=None, entity_type_dict=None):
        '''保存ht已经登录的实体信息，或者外部提供的相同格式的信息，目前保存的信息包括entity,mention,type.

        如果不提供两个dict参数，则默认使用模型自身已登录信息，否则使用提供的对应dict

        格式：


            entity||类别 mention||类别 mention||类别

            entity||类别 mention||类别

        每行第一个是实体名，其后都是对应的mention名，用一个空格分隔，每个名称后面都对应了其类别。

        保存这个信息的目的是为了便于手动编辑和导入:

        - 比如将某个mention作为独立的新entity，只需剪切到某一行的开头，并再复制一份再后面作为mention

        :param save_path: str, 要保存的文件路径（默认: ./ht_entities.txt）
        :param entity_mention_dict: dict, {entity:[mentions]}格式，
        :param entity_type_dict: dict, {entity:entity_type}格式，
        :return: None
        '''
        if entity_mention_dict is None and entity_type_dict is None:
            entity_mention_dict = self.entity_mention_dict
            entity_type_dict = self.entity_type_dict
        else:
            if entity_mention_dict is None:  # 用实体名直接作为默认指称
                entity_mention_dict = dict(
                    (entity0, {entity0}) for entity0 in entity_type_dict)
            else:
                entity_mention_dict = dict(
                    (entity0, set(mentions0)) for (entity0, mentions0) in entity_mention_dict.items())

            if entity_type_dict is None:
                entity_type_dict = {entity: "添加词" for entity in entity_mention_dict}

        # 两个dict不对齐的情况下，以添加词作为默认词性
        for entity in entity_mention_dict:
            if entity not in entity_type_dict:
                entity_type_dict[entity] = "添加词"

        if entity_mention_dict is None or entity_type_dict is None:
            return

        out_lines = []
        for entity, mentions0 in entity_mention_dict.items():
            etype = entity_type_dict[entity]
            enames = [entity] + list(mentions0)
            out_lines.append(" ".join("%s||%s" % (ename, etype) for ename in enames))

        dir0 = os.path.dirname(save_path)
        if dir0 != "":        # 如果在当前路径，则makedirs会报错
            os.makedirs(dir0, exist_ok=True)
        with open(save_path, "w", encoding='utf-8') as f:
            f.write("\n".join(out_lines))

    def load_entities(self, load_path='./ht_entities.txt', override=True):
        """从save_entities保存的文件读取实体信息

        :param load_path: str, 读取路径（默认：./ht_entities.txt）
        :param override: bool, 是否重写已登录实体，默认True
        :return: None, 实体已登录到ht中
        """
        # should have been inited at __init__(), but can override
        if override:
            self.clear()
        with open(load_path, encoding='utf-8') as f:
            for line in f:
                enames = line.strip().split()
                entity, etype = enames[0].split("||")
                mentions = set(x.split("||")[0] for x in enames[1:])
                self.entity_type_dict[entity] = etype
                self.entity_mention_dict[entity] = mentions

        type_entity_mention_dict = defaultdict(dict)
        for entity0, type0 in self.entity_type_dict.items():
            if entity0 in self.entity_mention_dict:
                type_entity_mention_dict[type0][entity0] = self.entity_mention_dict[entity0]
        self.type_entity_mention_dict = type_entity_mention_dict
        self._add_entities(type_entity_mention_dict)


    def cut_sentences(self, para, drop_empty_line=True, strip=True, deduplicate=False):
        '''cut_sentences

        :param para: 输入文本
        :param drop_empty_line: 是否丢弃空行
        :param strip: 是否对每一句话做一次strip
        :param deduplicate: 是否对连续标点去重，帮助对连续标点结尾的句子分句
        :return: sentences: list of str
        '''
        if deduplicate:
            para = re.sub(r"([。！？\!\?])\1+", r"\1", para)

        if self.language == 'en':
            from nltk import sent_tokenize
            sents = sent_tokenize(para)
            if strip:
                sents = [x.strip() for x in sents]
            if drop_empty_line:
                sents = [x for x in sents if len(x.strip()) > 0]
            return sents
        else:
            para = re.sub('([。！？\?!])([^”’])', r"\1\n\2", para)  # 单字符断句符
            para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
            para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
            para = re.sub('([。！？\?!][”’])([^，。！？\?])', r'\1\n\2', para)
            # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
            para = para.rstrip()  # 段尾如果有多余的\n就去掉它
            # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
            sentences = para.split("\n")
            if strip:
                sentences = [sent.strip() for sent in sentences]
            if drop_empty_line:
                sentences = [sent for sent in sentences if len(sent.strip()) > 0]
            return sentences

    def clean_text(self, text, remove_url=True, email=True, weibo_at=True, stop_terms=("转发微博",),
                   emoji=True, weibo_topic=False, deduplicate_space=True,
                   norm_url=False, norm_html=False, to_url=False,
                   remove_puncts=False, remove_tags=True, t2s=False,
                   expression_len=(1,6), linesep2space=False):
        '''
        进行各种文本清洗操作，微博中的特殊格式，网址，email，html代码，等等

        :param text: 输入文本
        :param remove_url: （默认使用）是否去除网址
        :param email: （默认使用）是否去除email
        :param weibo_at: （默认使用）是否去除微博的\@相关文本
        :param stop_terms: 去除文本中的一些特定词语，默认参数为("转发微博",)
        :param emoji: （默认使用）去除\[\]包围的文本，一般是表情符号
        :param weibo_topic: （默认不使用）去除##包围的文本，一般是微博话题
        :param deduplicate_space: （默认使用）合并文本中间的多个空格为一个
        :param norm_url: （默认不使用）还原URL中的特殊字符为普通格式，如(%20转为空格)
        :param norm_html: （默认不使用）还原HTML中的特殊字符为普通格式，如(\&nbsp;转为空格)
        :param to_url: （默认不使用）将普通格式的字符转为还原URL中的特殊字符，用于请求，如(空格转为%20)
        :param remove_puncts: （默认不使用）移除所有标点符号
        :param remove_tags: （默认使用）移除所有html块
        :param t2s: （默认不使用）繁体字转中文
        :param expression_len: 假设表情的表情长度范围，不在范围内的文本认为不是表情，不加以清洗，如[加上特别番外荞麦花开时共五册]。设置为None则没有限制
        :param linesep2space: （默认不使用）把换行符转换成空格
        :return: 清洗后的文本
        '''
        # unicode不可见字符
        # 未转义
        text = re.sub(r"[\u200b-\u200d]", "", text)
        # 已转义
        text = re.sub(r"(\\u200b|\\u200c|\\u200d)", "", text)
        # 反向的矛盾设置
        if norm_url and to_url:
            raise Exception("norm_url和to_url是矛盾的设置")
        if norm_html:
            text = html.unescape(text)
        if to_url:
            text = urllib.parse.quote(text)
        if remove_tags:
            text = w3lib.html.remove_tags(text)
        if remove_url:
            zh_puncts1 = "，；、。！？（）《》【】"
            URL_REGEX = re.compile(
                r'(?i)((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>' + zh_puncts1 + ']+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’' + zh_puncts1 + ']))',
                re.IGNORECASE)
            text = re.sub(URL_REGEX, "", text)
        if norm_url:
            text = urllib.parse.unquote(text)
        if email:
            EMAIL_REGEX = re.compile(r"[-a-z0-9_.]+@(?:[-a-z0-9]+\.)+[a-z]{2,6}", re.IGNORECASE)
            text = re.sub(EMAIL_REGEX, "", text)
        if weibo_at:
            text = re.sub(r"(回复)?(//)?\s*@\S*?\s*(:|：| |$)", " ", text)  # 去除正文中的@和回复/转发中的用户名
        if emoji:
            # 去除括号包围的表情符号
            # ? lazy match避免把两个表情中间的部分去除掉
            if type(expression_len) in {tuple, list} and len(expression_len) == 2:
                # 设置长度范围避免误伤人用的中括号内容，如[加上特别番外荞麦花开时共五册]
                lb, rb = expression_len
                text = re.sub(r"\[\S{"+str(lb)+r","+str(rb)+r"}?\]", "", text)  
            else:
                text = re.sub(r"\[\S+?\]", "", text)
            # text = re.sub(r"\[\S+\]", "", text)
            # 去除真,图标式emoji
            emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           "]+", flags=re.UNICODE)
            text = emoji_pattern.sub(r'', text)
        if weibo_topic:
            text = re.sub(r"#\S+#", "", text)  # 去除话题内容
        if linesep2space:
            text = text.replace("\n", " ")   # 不需要换行的时候变成1行
        if deduplicate_space:
            text = re.sub(r"(\s)+", r"\1", text)   # 合并正文中过多的空格
        if t2s:
            cc = OpenCC('t2s')
            text = cc.convert(text)
        assert hasattr(stop_terms, "__iter__"), Exception("去除的词语必须是一个可迭代对象")
        if type(stop_terms) == str:
            text = text.replace(stop_terms, "")
        else:
            for x in stop_terms:
                text = text.replace(x, "")
        if remove_puncts:
            allpuncs = re.compile(
                r"[，\_《。》、？；：‘’＂“”【「】」·！@￥…（）—\,\<\.\>\/\?\;\:\'\"\[\]\{\}\~\`\!\@\#\$\%\^\&\*\(\)\-\=\+]")
            text = re.sub(allpuncs, "", text)

        return text.strip()

    def clear(self):
        self.deprepare()
        self.__init__()

