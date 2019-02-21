# coding=utf-8
import os
import re
import json
import numpy as np
import pandas as pd
from itertools import combinations
import jieba
import jieba.posseg as pseg
from collections import defaultdict
from .word_discoverer import WordDiscoverer
from .sent_dict import SentDict
import logging
from pypinyin import lazy_pinyin, pinyin


class HarvestText:
    def __init__(self, standard_name=False):
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
    #
    # 实体分词模块
    #
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

    def add_entities(self, entity_mention_dict=None, entity_type_dict=None):
        if entity_mention_dict is None and entity_type_dict is None:
            return
        if entity_mention_dict is None:         # 用实体名直接作为默认指称
            entity_mention_dict = dict(
                (entity0, {entity0}) for entity0 in entity_type_dict)
        else:
            entity_mention_dict = dict(
                (entity0, set(mentions0)) for (entity0, mentions0) in entity_mention_dict.items())
        self.entity_mention_dict = entity_mention_dict
        if entity_type_dict:
            self.entity_type_dict = entity_type_dict
        else:
            self.entity_type_dict = {entity: "添加词" for entity in entity_mention_dict}
        type_entity_mention_dict = defaultdict(dict)
        for entity0, type0 in self.entity_type_dict.items():
            if entity0 in entity_mention_dict:
                type_entity_mention_dict[type0][entity0] = entity_mention_dict[entity0]
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

    def _add_entities(self, type_entity_mention_dict):
        for type0 in type_entity_mention_dict:
            entity_mention_dict0 = type_entity_mention_dict[type0]
            for entity0 in entity_mention_dict0:
                mentions = entity_mention_dict0[entity0]
                for mention0 in mentions:
                    self.build_trie(mention0, entity0, type0)
        self.prepare()

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
        '''
        找到单个指称对应的实体
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

    def get_pinyin_correct_candidates(self, word):  # 默认最多容忍一个拼音的变化
        pinyins = lazy_pinyin(word)
        tmp = pinyins[:]
        pinyin_cands = {tuple(pinyins)}
        for i, pinyin in enumerate(pinyins):
            if pinyin in self.pinyin_adjlist:
                pinyin_cands |= {tuple(tmp[:i] + [neibr] + tmp[i + 1:]) for neibr in self.pinyin_adjlist[pinyin]}
        pinyin_cands = pinyin_cands & set(self.pinyin_mention_dict.keys())
        mention_cands = set()
        for pinyin in pinyin_cands:
            mention_cands |= self.pinyin_mention_dict[pinyin]
        return list(mention_cands)

    def choose_from_multi_mentions(self,mention_cands,sent=""):
        surface0 = mention_cands[0]
        entity0, type0 = self.mention2entity(surface0)
        self._link_record(surface0, entity0)
        return entity0, type0



    def _entity_recheck(self, sent, entities_info, pinyin_recheck, char_recheck):
        sent2 = self.decoref(sent, entities_info)
        for word, flag in pseg.cut(sent2):
            if flag.startswith("n"):  # 对于名词，再检查是否有误差范围内匹配的其他指称
                entity0, type0 = None, None
                mention_cands = []
                if pinyin_recheck:
                    mention_cands += self.get_pinyin_correct_candidates(word)
                if char_recheck:
                    mention_cands += self.search_word_trie(word)

                if len(mention_cands) > 0:
                    entity0, type0 = self.choose_from_multi_mentions(mention_cands, sent)
                if entity0:
                    l = sent.find(word)
                    entities_info.append([(l,l+len(word)),(entity0, type0)])

    def _entity_linking(self, sent, pinyin_recheck=False, char_recheck=False):
        entities_info = []
        l = 0
        while l < len(sent):
            r, entity_types = self.dig_trie(sent, l)
            if r != -1 and r <= len(sent):
                surface0 = sent[l:r]  # 字面值
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
                l += 1
        return entities_info

    def entity_linking(self, sent, pinyin_recheck=False, char_recheck=False):
        self.check_prepared()
        entities_info = self._entity_linking(sent, pinyin_recheck, char_recheck)
        if pinyin_recheck or char_recheck:
            self._entity_recheck(sent, entities_info, pinyin_recheck, char_recheck)
        return entities_info

    def get_linking_mention_candidates(self, sent, pinyin_recheck=False, char_recheck=False):
        mention_cands = defaultdict(list)
        cut_result = []
        self.check_prepared()
        entities_info = self._entity_linking(sent, pinyin_recheck, char_recheck)
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
                if pinyin_recheck:
                    cands += self.get_pinyin_correct_candidates(word)
                if char_recheck:
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

    def cut_sentences(self, para, drop_empty_line = True):  # 分句
        para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
        para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
        para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
        para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
        # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
        para = para.rstrip()  # 段尾如果有多余的\n就去掉它
        # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
        sentences =  para.split("\n")
        if drop_empty_line:
            sentences = [sent for sent in sentences if len(sent.strip()) > 0]
        return sentences

    def named_entity_recognition(self, sent, standard_name=False):
        """
        利用pyhanlp的命名实体识别，找到句子中的（人名，地名，机构名）三种实体。harvesttext会预先链接已知实体
        :param sent:
        :param standard_name:
        :return: 发现的命名实体信息，字典 {实体名: 实体类型}
        """
        from pyhanlp import HanLP, JClass
        if not self.hanlp_prepared:
            self.hanlp_prepare()
        self.standard_name = standard_name
        entities_info = self.entity_linking(sent)
        sent2 = self.decoref(sent, entities_info)
        StandardTokenizer = JClass("com.hankcs.hanlp.tokenizer.StandardTokenizer")
        StandardTokenizer.SEGMENT.enableAllNamedEntityRecognize(True)
        entity_type_dict = {}
        try:
            for x in StandardTokenizer.segment(sent2):
                # 三种前缀代表：人名（nr），地名（ns），机构名（nt）
                tag0 = str(x.nature)
                if tag0.startswith("nr"):
                    entity_type_dict[x.word] = "人名"
                elif tag0.startswith("ns"):
                    entity_type_dict[x.word] = "地名"
                elif tag0.startswith("nt"):
                    entity_type_dict[x.word] = "机构名"
                elif tag0.startswith("nz"):
                    entity_type_dict[x.word] = "其他专名"
        except:
            pass
        return entity_type_dict
    def dependency_parse(self, sent, standard_name=False, stopwords=None):
        """
        依存句法分析，调用pyhanlp的接口，并且融入了harvesttext的实体识别机制。
        不保证高准确率。
        :param sent:
        :param standard_name:
        :param stopwords:
        :return: arcs：依存弧,列表中的列表。
        [[词语id,词语字面值或实体名(standard_name控制),词性，依存关系，依存子词语id] for 每个词语]
        """
        from pyhanlp import HanLP, JClass
        if not self.hanlp_prepared:
            self.hanlp_prepare()
        self.standard_name = standard_name
        entities_info = self.entity_linking(sent)
        sent2 = self.decoref(sent, entities_info)
        # [word.ID-1, word.LEMMA, word.POSTAG, word.DEPREL ,word.HEAD.ID-1]
        arcs = []
        i = 0
        sentence = HanLP.parseDependency(sent2)
        for word in sentence.iterator():
            word0, tag0 = word.LEMMA, word.POSTAG
            if stopwords and word0 in stopwords:
                continue
            if word0 in self.entity_types:
                if self.standard_name:
                    word0 = entities_info[i][1][0]  # 使用链接的实体
                else:
                    l, r = entities_info[i][0]  # 或使用原文
                    word0 = sent[l:r]
                tag0 = entities_info[i][1][1][1:-1]
                i += 1
            arcs.append([word.ID-1, word0, tag0, word.DEPREL, word.HEAD.ID-1])
        return arcs

    def triple_extraction(self, sent, standard_name=False, stopwords=None, expand = "all"):
        """
        利用主谓宾等依存句法关系，找到句子中有意义的三元组。
        很多代码参考：https://github.com/liuhuanyong/EventTriplesExtraction
        不保证高准确率。
        :param sent:
        :param standard_name:
        :param stopwords:
        :param expand: 默认"all"：扩展所有主谓词，"exclude_entity"：不扩展已知实体，可以保留标准的实体名，用于链接。"None":不扩展
        :return:
        """
        arcs = self.dependency_parse(sent, standard_name, stopwords)

        '''对找出的主语或者宾语进行扩展'''
        def complete_e(words, postags, child_dict_list, word_index):
            if expand == "all" or (expand == "exclude_entity" and "#"+postags[word_index]+"#" not in self.entity_types):
                child_dict = child_dict_list[word_index]
                prefix = ''
                if '定中关系' in child_dict:
                    for i in range(len(child_dict['定中关系'])):
                        prefix += complete_e(words, postags, child_dict_list, child_dict['定中关系'][i])
                postfix = ''
                if postags[word_index] == 'v':
                    if '动宾关系' in child_dict:
                        postfix += complete_e(words, postags, child_dict_list, child_dict['动宾关系'][0])
                    if '主谓关系' in child_dict:
                        prefix = complete_e(words, postags, child_dict_list, child_dict['主谓关系'][0]) + prefix

                return prefix + words[word_index] + postfix
            elif expand == "None":
                return words[word_index]
            else:            # (expand == "exclude_entity" and "#"+postags[word_index]+"#" in self.entity_types)
                return words[word_index]


        words, postags = ["" for i in range(len(arcs))], ["" for i in range(len(arcs))]
        child_dict_list = [defaultdict(list) for i in range(len(arcs))]
        for i, format_parse in enumerate(arcs):
            id0, words[i], postags[i], rel, headID = format_parse
            child_dict_list[headID][rel].append(i)
        svos = []
        for index in range(len(postags)):
            # 使用依存句法进行抽取
            if postags[index]:
                # 抽取以谓词为中心的事实三元组
                child_dict = child_dict_list[index]
                # 主谓宾
                if '主谓关系' in child_dict and '动宾关系' in child_dict:
                    r = words[index]
                    e1 = complete_e(words, postags, child_dict_list, child_dict['主谓关系'][0])
                    e2 = complete_e(words, postags, child_dict_list, child_dict['动宾关系'][0])
                    svos.append([e1, r, e2])

                # 定语后置，动宾关系
                relation = arcs[index][-2]
                head = arcs[index][-1]
                if relation == '定中关系':
                    if '动宾关系' in child_dict:
                        e1 = complete_e(words, postags, child_dict_list, head)
                        r = words[index]
                        e2 = complete_e(words, postags, child_dict_list, child_dict['动宾关系'][0])
                        temp_string = r + e2
                        if temp_string == e1[:len(temp_string)]:
                            e1 = e1[len(temp_string):]
                        if temp_string not in e1:
                            svos.append([e1, r, e2])
                # 含有介宾关系的主谓动补关系
                if '主谓关系' in child_dict and '动补结构' in child_dict:
                    e1 = complete_e(words, postags, child_dict_list, child_dict['主谓关系'][0])
                    CMP_index = child_dict['动补结构'][0]
                    r = words[index] + words[CMP_index]
                    if '介宾关系' in child_dict_list[CMP_index]:
                        e2 = complete_e(words, postags, child_dict_list, child_dict_list[CMP_index]['介宾关系'][0])
                        svos.append([e1, r, e2])
        return svos

    def clear(self):
        self.deprepare()
        self.__init__()

    #
    # 新词发现模块
    #
    def word_discover(self, doc, threshold_seeds=[], auto_param=True,
                      excluding_types=[], excluding_words=[],  # 可以排除已经登录的某些种类的实体，或者某些指定词
                      max_word_len=5, min_freq=0.00005, min_entropy=1.4, min_aggregation=50,
                      ent_threshold="both", mem_saving=0):
        # 采用经验参数，此时后面的参数设置都无效
        if auto_param:  # 根据自己的几个实验确定的参数估计值，没什么科学性，但是应该能得到还行的结果
            length = len(doc)
            min_entropy = np.log(length) / 10
            min_freq = min(0.00005, 20.0 / length)
            min_aggregation = np.sqrt(length) / 15
            mem_saving = int(length > 300000)
            # ent_threshold: 确定左右熵的阈值对双侧都要求"both"，或者只要左右平均值达到"avg"
            # 对于每句话都很极短的情况（如长度<8），经常出现在左右边界的词语可能难以被确定，这时ent_threshold建议设为"avg"
        try:
            ws = WordDiscoverer(doc, max_word_len, min_freq, min_entropy, min_aggregation, ent_threshold, mem_saving)
        except Exception as e:
            logging.log(logging.ERROR, str(e))
            info = {"text": [], "freq": [], "left_ent": [], "right_ent": [], "agg": []}
            info = pd.DataFrame(info)
            info = info.set_index("text")
            return info

        if len(excluding_types) > 0:
            if "#" in list(excluding_types)[0]:  # 化为无‘#’标签
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
                    min_score = min(min_score, info.loc[seed, "score"])
            if (min_score >= 100000):
                min_score = 0
            else:
                min_score *= 0.9  # 留一些宽松的区间
                info = info[info["score"] > min_score]
        return info

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

    def find_entity_with_rule(self, text, rulesets=[], add_to_dict=True, type0="添加词"):
        '''
        利用规则从分词结果中的词语找到实体，并可以赋予相应的类型再加入实体库
        :param text: string, 一段文本
        :param rulesets: list of (tuple of rules or single rule) from match_patterns,
        list中包含多个规则，满足其中一种规则的词就认为属于这个type
        而每种规则由tuple或单个条件(pattern)表示，一个词必须满足其中的一个或多个条件。
        :param add_to_dict: 是否把找到的结果直接加入词典
        :param type0: 赋予满足条件的词语的实体类型, 仅当add_to_dict时才有意义
        :return: found_entities

        '''
        found_entities = set()
        for word in self.seg(text):
            for ruleset in rulesets:  # 每个ruleset是或关系，只要满足一个就添加并跳过其他
                toAdd = True
                if type(ruleset) == type((1, 2)):  # tuple
                    for pattern0 in ruleset:
                        if not pattern0(word):
                            toAdd = False
                            break
                else:  # single rule
                    pattern0 = ruleset
                    if not pattern0(word):
                        toAdd = False
                if toAdd:
                    found_entities.add(word)
                    break
        if add_to_dict:
            for entity0 in found_entities:
                self.add_new_entity(entity0, entity0, type0)
            self.prepare()
        return found_entities

    #
    # 情感分析模块
    #
    def build_sent_dict(self, sents, method="PMI", min_times=5, ft_size=100, ft_epochs=15, ft_window=5, pos_seeds=[],
                        neg_seeds=[]):
        docs = [self.seg(sent) for sent in sents]
        self.sent_dict = SentDict(docs, method, min_times, ft_size, ft_epochs, ft_window, pos_seeds, neg_seeds)
        return self.sent_dict

    def analyse_sent(self, sent):
        return self.sent_dict.analyse_sent(self.seg(sent))

    #
    # 实体检索模块
    #
    def build_index(self, docs, with_entity=True, with_type=True):
        inv_index = defaultdict(set)
        for i, sent in enumerate(docs):
            entities_info = self.entity_linking(sent)
            for span, (entity, type0) in entities_info:
                if with_entity:
                    inv_index[entity].add(i)
                if with_type:
                    inv_index[type0].add(i)
        return inv_index

    def get_entity_counts(self, docs, inv_index, used_type=[]):
        if len(used_type) > 0:
            entities = iter(x for x in self.entity_type_dict
                            if self.entity_type_dict[x] in used_type)
        else:
            entities = self.entity_type_dict.keys()
        cnt = {enty: len(inv_index[enty]) for enty in entities if enty in inv_index}
        return cnt

    def search_entity(self, query, docs, inv_index):
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
    def get_summary(self, docs, topK=5, stopwords=None, with_importance=False, standard_name=True):
        import networkx as nx
        def sent_sim1(words1, words2):
            if len(words1) <= 1 or len(words2) <= 1:
                return 0.0
            return (len(set(words1) & set(words2))) / (np.log2(len(words1)) + np.log2(len(words2)))

        # 使用standard_name,相似度可以基于实体链接的结果计算而更加准确
        sents = [self.seg(doc.strip(), standard_name=standard_name, stopwords=stopwords) for doc in docs]
        sents = [sent for sent in sents if len(sent) > 0]
        G = nx.Graph()
        for u, v in combinations(range(len(sents)), 2):
            G.add_edge(u, v, weight=sent_sim1(sents[u], sents[v]))

        pr = nx.pagerank_scipy(G)
        pr_sorted = sorted(pr.items(), key=lambda x: x[1], reverse=True)
        if with_importance:
            return [(docs[i], imp) for i, imp in pr_sorted[:topK]]
        else:
            return [docs[i] for i, rank in pr_sorted[:topK]]

    #
    # 实体网络模块
    #
    def build_entity_graph(self, docs, min_freq=0, inv_index={}, used_types=[]):
        import networkx as nx
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
        '''
        根据文本和指定限定词，获得以限定词为中心的各词语的关系。
        限定词可以是一个特定的方面（衣食住行这类文档），这样就可以从词语中心图中获得关于这个方面的简要信息
        :param docs: 文本的列表
        :param word: 限定词
        :param standard_name: 把所有实体的指称化为标准实体名
        :param stopwords: 需要过滤的停用词
        :param min_freq: 作为边加入到图中的与中心词最小共现次数，用于筛掉可能过多的边
        :param other_min_freq: 中心词以外词语关系的最小共现次数
        :return: G（networxX中的Graph）

        '''
        import networkx as nx
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
        '''
        Entity only version of build_word_ego_graph()
        '''
        import networkx as nx
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
