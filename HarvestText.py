#coding=utf-8
import os
import numpy as np
import pandas as pd
import jieba
import jieba.posseg as pseg
from collections import defaultdict
from word_discoverer import WordDiscoverer
from sent_dict import SentDict


class HarvestText:
    def __init__(self,standard_name=False):
        self.standard_name = standard_name             # 是否使用连接到的实体名来替换原文
        self.entity_types = set()
        self.trie_root = {}

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
    def add_entities(self, entity_mention_dict, entity_type_dict):
        self.entity_mention_dict = entity_mention_dict
        self.entity_type_dict = entity_type_dict
        type_entity_mention_dict = defaultdict(dict)
        for entity0,type0 in entity_type_dict.items():
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
    def add_new_words(self,new_words):
        for word in new_words:
            self.build_trie(word,word,"新词")
        self.prepare()
    def prepare(self):
        for type0 in self.entity_types:
            jieba.add_word(type0, freq=10000, tag=type0[1:-1])
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
        # 收尾
        if "leaf" in trie_node:
            return len(sent), trie_node["leaf"]
        else:
            return -1, set()  # -1表示未找到
    def choose_from(self,name0, entities):
        # TODO: 加入更多同名不同实体的消歧策略
        return list(entities)[0]
    def entity_linking(self,sent):
        entities_info = []
        l = 0
        while l < len(sent):
            r, entities = self.dig_trie(sent, l)
            if r != -1:
                name0 = sent[l:r]
                entities_info.append(([l, r], self.choose_from(name0, entities)))  # 字典树能根据键找到实体范围，选择则依然需要根据历史等优化
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
    def word_discover(self, doc, threshold_seeds=[], auto_param = True,
                      excluding_types = [], excluding_words=[],     # 可以排除已经登录的某些种类的实体，或者某些指定词
                      max_word_len=5, min_freq=0.00005, min_entropy=1.4, min_aggregation=50,
                      ent_threshold="both",mem_saving=0):
        # 采用经验参数，此时后面的参数设置都无效
        if auto_param:               # 根据自己的几个实验确定的参数估计值，没什么科学性，但是应该能得到还行的结果
            length = len(doc)
            min_entropy = np.log(length) / 8
            min_freq = min(0.00005,20.0/length)
            min_aggregation = np.sqrt(length) / 10
            mem_saving = int(length>300000)
            # ent_threshold: 确定左右熵的阈值对双侧都要求"both"，或者只要左右平均值达到"avg"
            # 对于每句话都很极短的情况（如长度<8），经常出现在左右边界的词语可能难以被确定，这时ent_threshold建议设为"avg"
        ws = WordDiscoverer(doc,max_word_len,min_freq,min_aggregation, min_entropy, ent_threshold ,mem_saving)
        info = {"text":[],"freq":[],"left_ent":[],"right_ent":[],"agg":[]}
        if len(excluding_types) > 0:
            ex_mentions = [x for enty in self.entity_mention_dict
                             if self.entity_type_dict[enty] in excluding_types
                             for x in entity_mention_dict[enty]]
        else:
            ex_mentions = []
        ex_mentions += excluding_words
        for w in ws.word_infos:
            if w.text in ex_mentions:
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
    def build_sent_dict(self, sents, method="PMI",min_times=5, pos_seeds=[],neg_seeds=[]):
        docs = [self.seg(sent) for sent in sents]
        self.sent_dict = SentDict(docs, method, min_times, pos_seeds, neg_seeds)
        return self.sent_dict
    def analyse_sent(self, sent):
        words = [word for word in self.seg(sent) if word in self.sent_dict.words]
        if len(words) > 0:
            return sum(self.sent_dict[word] for word in words) / len(words)
        else:
            return 0.0
if __name__ == "__main__":
    ht = HarvestText()
    
    para = "上港的武磊和恒大的郜林，谁是中国最好的前锋？那当然是武球王，他是射手榜第一，原来是弱点的单刀也有了进步"
    
    
    # these 2 seeds can be obtained from some structured knowledge base
    print("add entity info(mention, type)")
    entity_mention_dict = {'武磊':['武磊','武球王'],'郜林':['郜林','郜飞机'],'前锋':['前锋'],'上海上港':['上港'],'广州恒大':['恒大'],'单刀球':['单刀']}
    entity_type_dict = {'武磊':'球员','郜林':'球员','前锋':'位置','上海上港':'球队','广州恒大':'球队','单刀球':'术语'}
    print(entity_mention_dict)
    print(entity_type_dict)
    ht.add_entities(entity_mention_dict,entity_type_dict)
    print("\nSentence segmentation")
    print(ht.seg(para,return_sent=True))
    print("\nPOS tagging with entity types")
    for word, flag in ht.posseg(para):
        print("%s:%s" % (word, flag),end = " ")
    print("\n\nentity_linking")
    for span, entity in ht.entity_linking(para):
        print(span, entity)

    #new_words_info = ht.word_discover(para,excluding_types=["球员"])  # 防止新词覆盖已知的球员名
    #new_words_info = ht.word_discover(para, threshold_seeds=["武磊"])
    new_words_info = ht.word_discover(para)
    print("\nnew words detection")
    print(new_words_info)
    new_words = new_words_info.index.tolist()
    # 添加识别到的新词，在后续的分词中将会优先分出这个词，词性为"新词"
    #ht.add_new_words(new_words)
    
    # 情感词典的构建及情感分析
    print("\nsentiment dictionary")
    sents = ["武磊威武，中超第一射手！",
          "武磊强，中超最第一本土球员！",
          "郜林不行，只会抱怨的球员注定上限了",
          "郜林看来不行，已经到上限了"]
    sent_dict = ht.build_sent_dict(sents,min_times=1,pos_seeds=["第一"],neg_seeds=["不行"])
    print("%s:%f" % ("威武",sent_dict["威武"]))
    print("%s:%f" % ("球员",sent_dict["球员"]))
    print("%s:%f" % ("上限",sent_dict["上限"]))
    print("\nsentence sentiment")
    sent = "武球王威武，中超最强球员！"
    print("%f:%s" % (ht.analyse_sent(sent),sent))
    
