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


