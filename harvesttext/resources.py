#coding=utf-8
#!/usr/bin/env python

# Resources

# 褒贬义词典   清华大学 李军
#
# 此资源被用于以下论文中:
# Jun Li and Maosong Sun, Experimental Study on Sentiment Classification of Chinese Review using Machine Learning Techniques, in Proceding of IEEE NLPKE 2007
# 李军 中文评论的褒贬义分类实验研究 硕士论文 清华大学 2008
import os
import json

def get_qh_sent_dict():
    """
    获得参考褒贬义词典：
    褒贬义词典   清华大学 李军

    此资源被用于以下论文中:
    Jun Li and Maosong Sun, Experimental Study on Sentiment Classification of Chinese Review using Machine Learning Techniques, in Proceding of IEEE NLPKE 2007
    李军 中文评论的褒贬义分类实验研究 硕士论文 清华大学 2008
    :return: qh_sent_dict = {"pos":[words],"neg":[words]}

    """

    pwd = os.path.abspath(os.path.dirname(__file__))
    with open(pwd+"/resources/qh_sent_dict.json","r",encoding="utf-8") as f:
        qh_sent_dict = json.load(f)
    return qh_sent_dict

def get_baidu_stopwords():
    """
        获得百度停用词列表
        来源，网上流传的版本：https://wenku.baidu.com/view/98c46383e53a580216fcfed9.html
        包含了中英文常见词及部分标点符号
        :return: stopwords: set of string

        """
    pwd = os.path.abspath(os.path.dirname(__file__))
    with open(pwd + "/resources/bd_stopwords.json", "r", encoding="utf-8") as f:
        stopwords = json.load(f)
    return set(stopwords)

def get_qh_typed_words(used_types = ['IT', '动物', '医药', '历史人名', '地名', '成语', '法律', '财经', '食物']):
    """
    THUOCL：清华大学开放中文词库
    http://thuocl.thunlp.org/
    IT	财经	成语	地名	历史名人	诗词	医学	饮食	法律	汽车	动物
    :param used_types:
    :return: typed_words: 字典，键为类型，值为该类的词语组成的set

    """
    pwd = os.path.abspath(os.path.dirname(__file__))
    with open(pwd + "/resources/THUOCL.json", "r", encoding="utf-8") as f:
        typed_words0 = json.load(f)
    typed_words = dict()
    for type0 in typed_words0:
        if type0 in used_types:
            typed_words[type0] = set(typed_words0[type0])
    return typed_words

def get_sanguo():
    """
    获得三国演义原文
    :return: ["章节1文本","章节2文本",...]

    """
    pwd = os.path.abspath(os.path.dirname(__file__))
    with open(pwd+"/resources/sanguo_docs.json","r",encoding="utf-8") as f:
        docs = json.load(f)
    return docs

def get_sanguo_entity_dict():
    """
    获得三国演义中的人名、地名、势力名的知识库。
    自行搭建的简单版，一定有遗漏和错误，仅供参考使用
    :return: entity_mention_dict,entity_type_dict

    """
    import json
    pwd = os.path.abspath(os.path.dirname(__file__))
    with open(pwd+"/resources/sanguo_entity_dict.json","r",encoding="utf-8") as f:
        entity_dict = json.load(f)
    return entity_dict["mention"], entity_dict["type"]

def get_english_senti_lexicon(type="LH"):
    """
    获得英语情感词汇表

    目前默认为来自这里的词汇表
    https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html#lexicon

    If you use this list, please cite the following paper:

       Minqing Hu and Bing Liu. "Mining and Summarizing Customer Reviews."
           Proceedings of the ACM SIGKDD International Conference on Knowledge
           Discovery and Data Mining (KDD-2004), Aug 22-25, 2004, Seattle,
           Washington, USA,

    :return: sent_dict = {"pos":[words],"neg":[words]}
    """
    pwd = os.path.abspath(os.path.dirname(__file__))
    with open(pwd + "/resources/LH_senti_lexicon.json", "r", encoding="utf-8") as f:
        senti_lexicon = json.load(f)
    return senti_lexicon


