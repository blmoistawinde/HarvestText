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
from collections import defaultdict

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

def get_nltk_en_stopwords():
    """
    来自nltk的英语停用词

    :return: stopwords: set of string
    """
    import nltk
    try:
        nltk.data.find('corpora/stopwords')
    except:
        nltk.download('stopwords')
    from nltk.corpus import stopwords
    return set(stopwords.words('english'))

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

def get_jieba_dict(min_freq=0, max_freq=float('inf'), with_pos=False, use_proxy=False, proxies=None):
    """
    获得jieba自带的中文词语词频词典
    
    :params min_freq: 选取词语需要的最小词频
    :params max_freq: 选取词语允许的最大词频
    :params with_pos: 返回结果是否包括词性信息
    :return if not with_pos, dict of {wd: freq}, else, dict of {(wd, pos): freq} 
    """
    from .download_utils import RemoteFileMetadata, check_download_resource
    remote = RemoteFileMetadata(
        filename='jieba_dict.txt',
        url='https://github.com/blmoistawinde/HarvestText/releases/download/V0.8/jieba_dict.txt',
        checksum='7197c3211ddd98962b036cdf40324d1ea2bfaa12bd028e68faa70111a88e12a8')
    file_path = check_download_resource(remote, use_proxy, proxies)
    ret = defaultdict(int)
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if len(line.strip().split()) == 3:
                wd, freq, pos = line.strip().split()
                freq = int(freq)
            if freq > min_freq and freq < max_freq:
                if not with_pos:
                    ret[wd] = freq
                else:
                    ret[(wd, pos)] = freq
    return ret
        