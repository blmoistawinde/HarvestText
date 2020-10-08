import re
from .resources import get_baidu_stopwords
from collections import defaultdict
from .algorithms.texttile import TextTile

class ParsingMixin:
    """
    文本解析模块：
    - 依存句法分析
    - 基于依存句法分析的三元组抽取
    - 基于Texttile的文本自动分段算法
    """
    def dependency_parse(self, sent, standard_name=False, stopwords=None):
        '''依存句法分析，调用pyhanlp的接口，并且融入了harvesttext的实体识别机制。不保证高准确率。

        :param sent:
        :param standard_name:
        :param stopwords:
        :return: arcs：依存弧,列表中的列表。
            [[词语id,词语字面值或实体名(standard_name控制),词性，依存关系，依存子词语id] for 每个词语]
        '''
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
        '''利用主谓宾等依存句法关系，找到句子中有意义的三元组。
        很多代码参考：https://github.com/liuhuanyong/EventTriplesExtraction
        不保证高准确率。

        :param sent:
        :param standard_name:
        :param stopwords:
        :param expand: 默认"all"：扩展所有主谓词，"exclude_entity"：不扩展已知实体，可以保留标准的实体名，用于链接。"None":不扩展
        :return:
        '''
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

    def cut_paragraphs(self, text, num_paras=None, block_sents=3, std_weight=0.5,
                       align_boundary=True, stopwords='baidu', remove_puncts=True,
                       seq_chars=-1, **kwargs):
        '''

        :param text:
        :param num_paras: (默认为None)可以手动设置想要划分的段落数，也可以保留默认值None，让算法自动确定
        :param block_sents: 算法的参数，将几句句子分为一个block。一般越大，算法自动划分的段落越少
        :param std_weight: 算法的参数。一般越大，算法自动划分的段落越多
        :param align_boundary: 新划分的段落是否要与原有的换行处对齐
        :param stopwords: 字符串列表/元组/集合，或者'baidu'为默认百度停用词，在算法中引入的停用词，一般能够提升准确度
        :param remove_puncts: （默认为True）是否在算法中去除标点符号，一般能够提升准确度
        :param seq_chars: （默认为-1）如果设置为>=1的值，则以包含这个数量的字符为基本单元，代替默认的句子。
        :param **kwargs: passed to ht.cut_sentences, like deduplicate
        :return:
        '''
        if num_paras is not None:
            assert num_paras > 0, "Should give a positive number of num_paras"
        assert stopwords == 'baidu' or (hasattr(stopwords, '__iter__') and type(stopwords) != str)
        stopwords = get_baidu_stopwords() if stopwords == 'baidu' else set(stopwords)
        if seq_chars < 1:
            cut_seqs = lambda x: self.cut_sentences(x, **kwargs)
        else:
            seq_chars = int(seq_chars)

            def _cut_seqs(text, len0, strip=True, deduplicate=False):
                if deduplicate:
                    text = re.sub(r"([。！？\!\?])\1+", r"\1", text)
                if strip:
                    text = text.strip()
                seqs = [text[i:i + len0] for i in range(0, len(text), len0)]
                return seqs

            cut_seqs = lambda x: _cut_seqs(x, seq_chars, **kwargs)

        if align_boundary:
            paras = [para.strip() for para in text.split("\n") if len(para.strip()) > 0]
            if num_paras is not None:
                # assert num_paras <= len(paras), "The new segmented paragraphs must be no less than the original ones"
                if num_paras >= len(paras):
                    return paras
            original_boundary_ids = []
            sentences = []
            for para in paras:
                sentences.extend(cut_seqs(para))
                original_boundary_ids.append(len(sentences))
        else:
            original_boundary_ids = None
            sentences = cut_seqs(text, **kwargs)
        # with entity resolution, can better decide similarity
        if remove_puncts:
            allpuncs = re.compile(
                r"[，\_《。》、？；：‘’＂“”【「】」、·！@￥…（）—\,\<\.\>\/\?\;\:\'\"\[\]\{\}\~\`\!\@\#\$\%\^\&\*\(\)\-\=\+]")
            sent_words = [re.sub(allpuncs, "",
                                 self.seg(sent, standard_name=True, stopwords=stopwords, return_sent=True)
                                 ).split()
                          for sent in sentences]
        else:
            sent_words = [self.seg(sent, standard_name=True, stopwords=stopwords)
                          for sent in sentences]
        texttiler = TextTile()
        predicted_boundary_ids = texttiler.cut_paragraphs(sent_words, num_paras, block_sents, std_weight,
                                                          align_boundary, original_boundary_ids)
        jointer = " " if (self.language == 'en' and seq_chars > 1) else ""
        predicted_paras = [jointer.join(sentences[l:r]) for l, r in
                           zip([0] + predicted_boundary_ids[:-1], predicted_boundary_ids)]
        return predicted_paras
