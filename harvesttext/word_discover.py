import jieba
import jieba.analyse
import logging
import networkx as nx
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from .resources import get_baidu_stopwords
from .algorithms.word_discoverer import WordDiscoverer
from .algorithms.entity_discoverer import NFLEntityDiscoverer, NERPEntityDiscover
from .algorithms.keyword import textrank

class WordDiscoverMixin:
    """
    新词、关键词发现模块:
    - 基于凝聚度和左右熵的新词发现
    - 基于模式的专有名词发现
    - 命名实体识别
    - 实验性质的实体别名发现算法
    """
    def word_discover(self, doc, threshold_seeds=[], auto_param=True,
                      excluding_types=[], excluding_words='baidu_stopwords',  # 可以排除已经登录的某些种类的实体，或者某些指定词
                      max_word_len=5, min_freq=0.00005, min_entropy=1.4, min_aggregation=50,
                      ent_threshold="both", mem_saving=None, sort_by='freq', exclude_number=True):
        '''新词发现，基于 http://www.matrix67.com/blog/archives/5044 实现及微调

        :param doc: (string or list) 待进行新词发现的语料，如果是列表的话，就会自动用换行符拼接
        :param threshold_seeds: list of string, 设定能接受的“质量”最差的种子词，更差的词语将会在新词发现中被过滤
        :param auto_param: bool, 使用默认的算法参数
        :param excluding_types: list of str, 设定要过滤掉的特定词性或已经登录到ht的实体类别
        :param excluding_words: list of str, 设定要过滤掉的特定词
        :param max_word_len: 允许被发现的最长的新词长度
        :param min_freq: 被发现的新词，在给定文本中需要达到的最低频率
        :param min_entropy: 被发现的新词，在给定文本中需要达到的最低左右交叉熵
        :param min_aggregation: 被发现的新词，在给定文本中需要达到的最低凝聚度
        :param ent_threshold: "both": (默认)在使用左右交叉熵进行筛选时，两侧都必须超过阈值; "avg": 两侧的平均值达到阈值即可
        :param mem_saving: bool or None, 采用一些过滤手段来减少内存使用，但可能影响速度。如果不指定，对长文本自动打开，而对短文本不使用
        :param sort_by: 以下string之一: {'freq': 词频, 'score': 综合分数, 'agg':凝聚度} 按照特定指标对得到的词语信息排序，默认使用词频
        :param exclude_number: （默认True）过滤发现的纯数字新词
        :return: info: 包含新词作为index, 以及对应各项指标的DataFrame
        '''
        if type(doc) != str:
            doc = "\n".join(doc)
        # 采用经验参数，此时后面的参数设置都无效
        if auto_param:  # 根据自己的几个实验确定的参数估计值，没什么科学性，但是应该能得到还行的结果
            length = len(doc)
            min_entropy = np.log(length) / 10
            min_freq = min(0.00005, 20.0 / length)
            min_aggregation = np.sqrt(length) / 15
            mem_saving = bool(length > 300000) if mem_saving is None else mem_saving
            # ent_threshold: 确定左右熵的阈值对双侧都要求"both"，或者只要左右平均值达到"avg"
            # 对于每句话都很极短的情况（如长度<8），经常出现在左右边界的词语可能难以被确定，这时ent_threshold建议设为"avg"
        mem_saving = False if mem_saving is None else mem_saving

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
            ex_mentions = set(x for enty in self.entity_mention_dict
                           if enty in self.entity_type_dict and
                           self.entity_type_dict[enty] in excluding_types
                           for x in self.entity_mention_dict[enty])
        else:
            ex_mentions = set()
        assert excluding_words == 'baidu_stopwords' or (hasattr(excluding_words, '__iter__') and type(excluding_words) != str)
        if excluding_words == 'baidu_stopwords':
            ex_mentions |= get_baidu_stopwords()
        else:
            ex_mentions |= set(excluding_words)

        info = ws.get_df_info(ex_mentions, exclude_number)

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
        if sort_by:
            info.sort_values(by=sort_by, ascending=False, inplace=True)

        return info

    def find_entity_with_rule(self, text, rulesets=[], add_to_dict=True, type0="添加词"):
        '''利用规则从分词结果中的词语找到实体，并可以赋予相应的类型再加入实体库

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

    def named_entity_recognition(self, sent, standard_name=False, return_posseg=False):
        '''利用pyhanlp的命名实体识别，找到句子中的（人名，地名，机构名，其他专名）实体。harvesttext会预先链接已知实体

        :param sent: string, 文本
        :param standard_name: bool, 是否把连接到的已登录转化为标准名
        :param return_posseg: bool, 是否返回包括命名实体识别的，带词性分词结果
        :param book: bool, 预先识别
        :return: entity_type_dict: 发现的命名实体信息，字典 {实体名: 实体类型}
            (return_posseg=True时) possegs: list of (单词, 词性)
        '''
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
            possegs = []
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
                possegs.append((x.word, tag0))
        except:
            pass
        if return_posseg:
            return entity_type_dict, possegs
        else:
            return entity_type_dict
    def entity_discover(self, text, return_count=False, method="NFL", min_count=5, pinyin_tolerance=0, **kwargs):
        """无监督地从较大量文本中发现实体的类别和多个同义mention。建议对千句以上的文本来挖掘，并且文本的主题比较集中。
            效率：在测试环境下处理一个约10000句的时间大约是20秒。另一个约200000句的语料耗时2分半
            精度：算法准确率不高，但是可以初步聚类，建议先save_entities后, 再进行手动进行调整，然后load_entities再用于进一步挖掘

            ref paper: Mining Entity Synonyms with Efficient Neural Set Generation(https://arxiv.org/abs/1811.07032v1)

        :param text: string or list of string
        :param return_count: (default False) 是否再返回每个mention的出现次数
        :param method: 使用的算法， 目前可选 "NFL" (NER+Fasttext+Louvain+模式修复，基于语义和规则发现同义实体，但可能聚集过多错误实体), "NERP"(NER+模式修复, 仅基于规则发现同义实体)
        :param min_count: (default 5) mininum freq of word to be included
        :param pinyin_tolerance: {None, 0, 1} 合并拼音相同(取0时)或者差别只有一个(取1时)的候选词到同一组实体，默认使用(0)
        :param kwargs: 根据算法决定的参数，目前, "NERP"不需要额外参数，而"NFL"可接受的额外参数有：

            emb_dim: (default 50) fasttext embedding's dimensions

            threshold: (default 0.98) [比较敏感，调参重点]larger for more entities, threshold for add an edge between 2 entities if cos_dim exceeds

            ft_iters: (default 20) larger for more entities, num of iterations used by fasttext

            use_subword: (default True) whether to use fasttext's subword info

            min_n: (default 1) min length of used subword

            max_n: (default 4) max length of used subword

        :return: entity_mention_dict, entity_type_dict
        """
        text = text if type(text) == str else "\n".join(text)
        method = method.upper()
        assert method in {"NFL", "NERP"}
        # discover candidates with NER
        print("Doing NER")
        sent_words = []
        type_entity_dict = defaultdict(set)
        entity_count = defaultdict(int)
        wd_count = defaultdict(int)
        for sent in tqdm(self.cut_sentences(text)):
            NERs0, possegs = self.named_entity_recognition(sent, return_posseg=True)
            sent_wds0 = []
            for wd, pos in possegs:
                if wd in NERs0:
                    zh_pos = NERs0[wd]
                    entity_name = wd.lower() + "_" + zh_pos
                    type_entity_dict[zh_pos].add(entity_name)
                    sent_wds0.append(entity_name)
                    entity_count[entity_name] += 1
                else:
                    sent_wds0.append(wd)
                    wd_count[wd] += 1
            sent_words.append(sent_wds0)

        entity_count = pd.Series(entity_count)
        entity_count = entity_count[entity_count >= min_count]
        pop_words_cnt = {wd:cnt for wd, cnt in wd_count.items() if cnt >= min_count}
        id2word = entity_count.index.tolist()
        word2id = {wd: i for (i, wd) in enumerate(id2word)}

        type_entity_dict2 = {k: list(v) for k, v in type_entity_dict.items()}
        if method == "NFL":
            discoverer = NFLEntityDiscoverer(sent_words, type_entity_dict2, entity_count, pop_words_cnt, word2id, id2word,
                                             min_count, pinyin_tolerance, self.pinyin_adjlist, **kwargs)
        elif method == "NERP":
            discoverer = NERPEntityDiscover(sent_words, type_entity_dict2, entity_count, pop_words_cnt, word2id, id2word,
                                            min_count, pinyin_tolerance, self.pinyin_adjlist, **kwargs)
        entity_mention_dict, entity_type_dict = discoverer.entity_mention_dict, discoverer.entity_type_dict
        mention_count = discoverer.mention_count         # 新添加的mention的count在discoverer里更新
        if return_count:
            return entity_mention_dict, entity_type_dict, mention_count
        else:
            return entity_mention_dict, entity_type_dict
    
    def extract_keywords(self, text, topK, with_score=False, min_word_len=2, stopwords="baidu", allowPOS="default", method="jieba_tfidf", **kwargs):
        """用各种算法抽取关键词（目前均为无监督），结合了ht的实体分词来提高准确率

        目前支持的算法类型（及额外参数）：

        - jieba_tfidf: （默认）jieba自带的基于tfidf的关键词抽取算法，idf统计信息来自于其语料库
        - textrank: 基于textrank的关键词抽取算法
            - block_type: 默认"doc"。 支持三种级别，"sent", "para", "doc"，每个block之间的临近词语不建立连边
            - window: 默认2, 邻接的几个词语之内建立连边
            - weighted: 默认False, 时候使用加权图计算textrank
            - 构建词图时会过滤不符合min_word_len, stopwords, allowPOS要求的词语

        :params text: 从中挖掘关键词的文档
        :params topK: int, 从每个文档中抽取的关键词（最大）数量
        :params with_score: bool, 默认False, 是否同时返回算法提供的分数（如果有的话）
        :params min_word_len: 默认2, 被纳入关键词的词语不低于此长度
        :param stopwords: 字符串列表/元组/集合，或者'baidu'为默认百度停用词，在算法中引入的停用词，一般能够提升准确度
        :params allowPOS: iterable of str，关键词应当属于的词性，默认为"default" {'n', 'ns', 'nr', 'nt', 'nz', 'vn', 'v', 'an', 'a', 'i'}以及已登录的实体词类型
        :params method: 选择用于抽取的算法，目前支持"jieba_tfidf", "tfidf", "textrank"
        :params kwargs: 其他算法专属参数


        """
        assert method in {"jieba_tfidf", "textrank"}, print("目前不支持的算法")
        if allowPOS == 'default':
            # ref: 结巴分词标注兼容_ICTCLAS2008汉语词性标注集 https://www.cnblogs.com/hpuCode/p/4416186.html
            allowPOS = {'n', 'ns', 'nr', 'nt', 'nz', 'vn', 'v', 'an', 'a', 'i'}
        else:
            assert hasattr(allowPOS, "__iter__")
        # for HT, we consider registered entity types specifically
        allowPOS |= set(self.type_entity_mention_dict)

        assert stopwords == 'baidu' or (hasattr(stopwords, '__iter__') and type(stopwords) != str)
        stopwords = get_baidu_stopwords() if stopwords == 'baidu' else set(stopwords)
        
        if method == "jieba_tfidf":
            kwds = jieba.analyse.extract_tags(text, topK=int(2*topK), allowPOS=allowPOS, withWeight=with_score)
            if with_score:
                kwds = [(kwd, score) for (kwd, score) in kwds if kwd not in stopwords][:topK]
            else:
                kwds = kwds[:topK]
        elif method == "textrank":
            block_type = kwargs.get("block_type", "doc")
            assert block_type in {"sent", "para", "doc"}
            window = kwargs.get("window", 2)
            weighted = kwargs.get("weighted", True)
            if block_type == "doc":
                blocks = [text]
            elif block_type == "para":
                blocks = [para.strip() for para in text.split("\n") if para.strip() != ""]
            elif block_type == "sent":
                blocks = self.cut_sentences(text)
            block_pos = (self.posseg(block.strip(), stopwords=stopwords) for block in blocks)
            block_words = [[wd for wd, pos in x 
                               if pos in allowPOS and len(wd) >= min_word_len] 
                               for x in block_pos]
            kwds = textrank(block_words, topK, with_score, window, weighted)
        
        return kwds

            
            