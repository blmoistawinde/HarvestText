#coding=utf-8
#!/usr/bin/env python

# Resources

# 褒贬义词典   清华大学 李军
#
# 此资源被用于以下论文中:
# Jun Li and Maosong Sun, Experimental Study on Sentiment Classification of Chinese Review using Machine Learning Techniques, in Proceding of IEEE NLPKE 2007
# 李军 中文评论的褒贬义分类实验研究 硕士论文 清华大学 2008

def get_qh_sent_dict():
    """
    获得参考褒贬义词典：
    褒贬义词典   清华大学 李军

    此资源被用于以下论文中:
    Jun Li and Maosong Sun, Experimental Study on Sentiment Classification of Chinese Review using Machine Learning Techniques, in Proceding of IEEE NLPKE 2007
    李军 中文评论的褒贬义分类实验研究 硕士论文 清华大学 2008
    :return: qh_sent_dict = {"pos":[words],"neg":[words]}
    """
    qh_sent_dict = {"pos":[],"neg":[]}
    with open("../resources/sentiment.dict.v1.0/tsinghua.positive.gb.txt","r") as f:
        qh_sent_dict["pos"] = f.read().split()

    with open("../resources/sentiment.dict.v1.0/tsinghua.negative.gb.txt","r") as f:
        qh_sent_dict["neg"] = f.read().split()
    return qh_sent_dict

def get_sanguo():
    """
    获得三国演义原文
    :return: ["章节1文本","章节2文本",...]
    """
    import os
    basedir = "../resources/三国演义/"
    docs = ["" for i in range(120)]
    for i in range(1,121):
        with open(basedir + f"{i}.txt", "r" ,encoding="utf-8") as f:
            docs[i-1] = f.read().strip()
    return docs

def get_sanguo_entity_dict():
    """
    获得三国演义中的人名、地名、势力名的知识库。
    自行搭建的简单版，一定有遗漏和错误，仅供参考使用
    :return: entity_mention_dict,entity_type_dict
    """
    import json
    with open("../resources/sanguo_entity_dict.json","r",encoding="utf-8") as f:
        entity_dict = json.load(f)
    return entity_dict["mention"], entity_dict["type"]

