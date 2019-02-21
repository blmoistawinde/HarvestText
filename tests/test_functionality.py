#coding=utf-8
import os
import sys
import inspect

def get_current_function_name():
    return os.path.dirname(os.path.abspath(__file__))+"/"+inspect.stack()[1][3]
import _locale
_locale._getdefaultlocale = (lambda *args: ['zh_CN', 'utf8'])

from harvesttext import HarvestText
ht = HarvestText()

def test_new_word_discover():
    sys.stdout = open(get_current_function_name()+"_current","w")
    expected = open(get_current_function_name()+"_expected").read()
    para = "上港的武磊和恒大的郜林，谁是中国最好的前锋？那当然是武磊武球王了，他是射手榜第一，原来是弱点的单刀也有了进步"
    # 返回关于新词质量的一系列信息，允许手工改进筛选(pd.DataFrame型)
    new_words_info = ht.word_discover(para)
    # new_words_info = ht.word_discover(para, threshold_seeds=["武磊"])
    new_words = new_words_info.index.tolist()
    print(new_words)
    sys.stdout.close()
    assert open(get_current_function_name() + "_current").read() == expected

def test_new_word_register():
    sys.stdout, expected = open(get_current_function_name()+"_current","w"), open(get_current_function_name()+"_expected").read()
    new_words = ["落叶球","666"]
    ht.add_new_words(new_words)        # 作为广义上的"新词"登录
    ht.add_new_entity("落叶球", mention0="落叶球", type0="术语") # 作为特定类型登录
    print(ht.seg("这个落叶球踢得真是666", return_sent=True))
    for word, flag in ht.posseg("这个落叶球踢得真是666"):
        print("%s:%s" % (word, flag), end=" ")
    sys.stdout.close()
    assert open(get_current_function_name() + "_current").read() == expected

def test_entity_segmentation():
    sys.stdout, expected = open(get_current_function_name()+"_current","w"), open(get_current_function_name()+"_expected").read()
    para = "上港的武磊和恒大的郜林，谁是中国最好的前锋？那当然是武磊武球王了，他是射手榜第一，原来是弱点的单刀也有了进步"
    print("\nadd entity info(mention, type)")
    entity_mention_dict = {'武磊': ['武磊', '武球王'], '郜林': ['郜林', '郜飞机'], '前锋': ['前锋'], '上海上港': ['上港'], '广州恒大': ['恒大'],
                           '单刀球': ['单刀']}
    entity_type_dict = {'武磊': '球员', '郜林': '球员', '前锋': '位置', '上海上港': '球队', '广州恒大': '球队', '单刀球': '术语'}
    ht.add_entities(entity_mention_dict, entity_type_dict)

    print("\nWord segmentation")
    print(ht.seg(para, return_sent=True))  # return_sent=False时，则返回词语列表

    print("\nPOS tagging with entity types")
    for word, flag in ht.posseg(para):
        print("%s:%s" % (word, flag), end=" ")

    print("\n\nentity_linking")
    for span, entity in ht.entity_linking(para):
        print(span, entity)

    print("Sentence segmentation")
    print(ht.cut_sentences(para))

    sys.stdout.close()
    assert open(get_current_function_name() + "_current").read() == expected

def test_sentiment_dict():
    sys.stdout, expected = open(get_current_function_name()+"_current","w"), open(get_current_function_name()+"_expected").read()
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
    print("%f:%s" % (ht.analyse_sent(sent), sent))

    sys.stdout.close()
    assert open(get_current_function_name() + "_current").read() == expected

def test_entity_search():
    sys.stdout, expected = open(get_current_function_name()+"_current","w"), open(get_current_function_name()+"_expected").read()
    print("\nentity search")
    docs = ["武磊威武，中超第一射手！",
            "郜林看来不行，已经到上限了。",
            "武球王威武，中超最强前锋！",
            "武磊和郜林，谁是中国最好的前锋？"]
    inv_index = ht.build_index(docs)
    print(ht.get_entity_counts(docs, inv_index))  # 获得文档中所有实体的出现次数
    print(ht.search_entity("武磊", docs, inv_index))  # 单实体查找
    print(ht.search_entity("武磊 郜林", docs, inv_index))  # 多实体共现

    # 谁是最被人们热议的前锋？用这里的接口可以很简便地回答这个问题
    subdocs = ht.search_entity("#球员# 前锋", docs, inv_index)
    print(subdocs)  # 实体、实体类型混合查找
    inv_index2 = ht.build_index(subdocs)
    print(ht.get_entity_counts(subdocs, inv_index2, used_type=["球员"]))  # 可以限定类型

    sys.stdout.close()
    assert open(get_current_function_name() + "_current").read() == expected

def test_text_summarization():
    sys.stdout, expected = open(get_current_function_name()+"_current","w"), open(get_current_function_name()+"_expected").read()
    # 文本摘要
    print("\nText summarization")
    docs = ["武磊威武，中超第一射手！",
            "郜林看来不行，已经到上限了。",
            "武球王威武，中超最强前锋！",
            "武磊和郜林，谁是中国最好的前锋？"]
    for doc in ht.get_summary(docs, topK=2):
        print(doc)

    sys.stdout.close()
    assert open(get_current_function_name() + "_current").read() == expected

def test_entity_network():
    sys.stdout, expected = open(get_current_function_name()+"_current","w"), open(get_current_function_name()+"_expected").read()
    print("\nentity network")
    # 在现有实体库的基础上随时新增，比如从新词发现中得到的漏网之鱼
    ht.add_new_entity("颜骏凌", "颜骏凌", "球员")
    docs = ["武磊和颜骏凌是队友",
            "武磊和郜林都是国内顶尖前锋"]
    G = ht.build_entity_graph(docs)
    print(dict(G.edges.items()))
    G = ht.build_entity_graph(docs, used_types=["球员"])
    print(dict(G.edges.items()))

    sys.stdout.close()
    assert open(get_current_function_name() + "_current").read() == expected

def test_save_load_clear():
    sys.stdout, expected = open(get_current_function_name()+"_current","w"), open(get_current_function_name()+"_expected").read()
    from harvesttext import loadHT,saveHT
    para = "上港的武磊和恒大的郜林，谁是中国最好的前锋？那当然是武磊武球王了，他是射手榜第一，原来是弱点的单刀也有了进步"
    saveHT(ht,"ht_model1")
    ht2 = loadHT("ht_model1")
    print("cut with loaded model")
    print(ht2.seg(para))
    ht2.clear()
    print("cut with cleared model")
    print(ht2.seg(para))

    sys.stdout.close()
    assert open(get_current_function_name() + "_current").read() == expected

def test_load_resources():
    sys.stdout, expected = open(get_current_function_name()+"_current","w"), open(get_current_function_name()+"_expected").read()
    from harvesttext.resources import get_qh_sent_dict,get_baidu_stopwords,get_sanguo,get_sanguo_entity_dict
    sdict = get_qh_sent_dict()              # {"pos":[积极词...],"neg":[消极词...]}
    print("pos_words:",list(sdict["pos"])[10:15])
    print("neg_words:",list(sdict["neg"])[5:10])

    stopwords = get_baidu_stopwords()
    print("stopwords:", list(stopwords)[5:10])

    docs = get_sanguo()                 # 文本列表，每个元素为一章的文本
    print("三国演义最后一章末16字:\n",docs[-1][-16:])
    entity_mention_dict, entity_type_dict = get_sanguo_entity_dict()
    print("刘备 指称：",entity_mention_dict["刘备"])
    print("刘备 类别：",entity_type_dict["刘备"])
    print("蜀 类别：", entity_type_dict["蜀"])
    print("益州 类别：", entity_type_dict["益州"])

    sys.stdout.close()
    assert open(get_current_function_name() + "_current").read() == expected

def test_linking_strategy():
    sys.stdout, expected = open(get_current_function_name()+"_current","w"), open(get_current_function_name()+"_expected").read()
    ht0 = HarvestText()
    def test_case(text0,entity_mention_dict,strategy,entity_type_dict=None,**kwargs):
        ht0.add_entities(entity_mention_dict,entity_type_dict)
        ht0.set_linking_strategy(strategy,**kwargs)
        print(ht0.entity_linking(text0))
        ht0.clear()
    # latest 例
    test_case('X老师您好。请问老师这题怎么做？',
              entity_mention_dict={"X老师": ["X老师", "老师"], "Y老师": ["Y老师", "老师"]},
              strategy="latest"
              )

    test_case('谢谢老师',
              entity_mention_dict={"X老师": ["X老师", "老师"], "Y老师": ["Y老师", "老师"]},
              strategy="latest",
              lastest_mention={"老师": "X老师"})

    # freq 单字面值例
    test_case('市长',
              entity_mention_dict={"A市长": ["市长"], "B市长": ["长江"]},
              strategy="freq",
              entity_freq={"A市长": 5, "B市长": 3})

    # freq 重叠字面值例
    test_case('xx市长江yy',
              entity_mention_dict={"xx市长":["xx市长"],"长江yy":["长江yy"]},
              strategy="freq",
              entity_freq={"xx市长":3,"长江yy":5})

    test_case('我叫小沈阳',
              entity_mention_dict={"沈阳": ["沈阳"], "小沈阳": ["小沈阳"]},
              strategy="freq",
              entity_type_dict={"沈阳": "地名", "小沈阳": "人名"},
              type_freq={"地名": -1})

    sys.stdout.close()
    assert open(get_current_function_name() + "_current").read() == expected


def test_find_with_rules():
    sys.stdout, expected = open(get_current_function_name()+"_current","w"), open(get_current_function_name()+"_expected").read()
    from harvesttext.match_patterns import UpperFirst, AllEnglish, Contains, StartsWith, EndsWith
    # some more patterns is provided
    text0 = "我喜欢Python，因为requests库很适合爬虫"
    ht0 = HarvestText()

    found_entities = ht0.find_entity_with_rule(text0, rulesets=[AllEnglish()], type0="英文名")
    print(found_entities)
    print(ht0.posseg(text0))
    print(ht0.mention2entity("Python"))


    # Satisfying one of the rules
    ht0.clear()
    found_entities = ht0.find_entity_with_rule(text0,rulesets=[AllEnglish(),Contains("爬")],type0="技术")
    print(found_entities)
    print(ht0.posseg(text0))

    # Satisfying a couple of rules [using tuple]
    ht0.clear()
    found_entities = ht0.find_entity_with_rule(text0, rulesets=[(AllEnglish(),UpperFirst())], type0="专有英文词")
    print(found_entities)
    print(ht0.posseg(text0))

    sys.stdout.close()
    assert open(get_current_function_name() + "_current").read() == expected

def test_build_word_ego_graph():
    sys.stdout, expected = open(get_current_function_name()+"_current","w"), open(get_current_function_name()+"_expected").read()
    import networkx as nx
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 步骤一（替换sans-serif字体）
    plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）
    from harvesttext import get_sanguo, get_sanguo_entity_dict, get_baidu_stopwords

    ht0 = HarvestText()
    entity_mention_dict, entity_type_dict = get_sanguo_entity_dict()
    ht0.add_entities(entity_mention_dict, entity_type_dict)
    sanguo1 = get_sanguo()[0]
    stopwords = get_baidu_stopwords()
    docs = ht0.cut_sentences(sanguo1)
    G = ht0.build_word_ego_graph(docs,"刘备",min_freq=3,other_min_freq=2,stopwords=stopwords)
    pos = nx.kamada_kawai_layout(G)
    nx.draw(G,pos)
    nx.draw_networkx_labels(G,pos)

    G = ht0.build_entity_ego_graph(docs, "刘备", min_freq=3, other_min_freq=2)
    pos = nx.spring_layout(G)
    nx.draw(G, pos)
    nx.draw_networkx_labels(G, pos)


    sys.stdout.close()
    assert open(get_current_function_name() + "_current").read() == expected

def test_using_typed_words():
    sys.stdout, expected = open(get_current_function_name()+"_current","w"), open(get_current_function_name()+"_expected").read()
    from harvesttext.resources import get_qh_typed_words,get_baidu_stopwords
    ht0 = HarvestText()
    typed_words, stopwords = get_qh_typed_words(), get_baidu_stopwords()
    ht0.add_typed_words(typed_words)
    print("加载清华领域词典，并使用停用词")
    print("全部类型",typed_words.keys())
    sentence = "THUOCL是自然语言处理的一套中文词库，词表来自主流网站的社会标签、搜索热词、输入法词库等。"
    print(sentence)
    print(ht0.posseg(sentence,stopwords=stopwords))
    print("一些词语被赋予特殊类型IT,而“是”等词语被筛出。")

    sys.stdout.close()
    assert open(get_current_function_name() + "_current").read() == expected

def test_entity_error_check():
    sys.stdout, expected = open(get_current_function_name()+"_current","w"), open(get_current_function_name()+"_expected").read()
    ht0 = HarvestText()
    typed_words = {"人名":["武磊"]}
    ht0.add_typed_words(typed_words)
    sent1 = "武磊和吴力只差一个拼音"
    print(sent1)
    print(ht0.entity_linking(sent1, pinyin_recheck=True))
    sent2 = "武磊和吴磊只差一个字"
    print(sent2)
    print(ht0.entity_linking(sent2, char_recheck=True))
    sent3 = "吴磊和吴力都可能是武磊的代称"
    print(sent3)
    print(ht0.get_linking_mention_candidates(sent3, pinyin_recheck=True, char_recheck=True))

    sys.stdout.close()
    assert open(get_current_function_name() + "_current").read() == expected

def test_depend_parse():
    sys.stdout, expected = open(get_current_function_name()+"_current","w"), open(get_current_function_name()+"_expected").read()
    ht0 = HarvestText()
    para = "上港的武磊武球王是中国最好的前锋。"
    entity_mention_dict = {'武磊': ['武磊', '武球王'], "上海上港":["上港"]}
    entity_type_dict = {'武磊': '球员', "上海上港":"球队"}
    ht0.add_entities(entity_mention_dict, entity_type_dict)
    for arc in ht0.dependency_parse(para):
        print(arc)
    print(ht0.triple_extraction(para))

    sys.stdout.close()
    assert open(get_current_function_name() + "_current").read() == expected

def test_named_entity_recognition():
    sys.stdout, expected = open(get_current_function_name()+"_current","w"), open(get_current_function_name()+"_expected").read()
    ht0 = HarvestText()
    sent = "上海上港足球队的武磊是中国最好的前锋。"
    print(ht0.named_entity_recognition(sent))

    sys.stdout.close()
    assert open(get_current_function_name()+"_current").read() == expected


if __name__ == "__main__":
    test_new_word_discover()
    test_new_word_register()
    test_entity_segmentation()
    test_sentiment_dict()
    test_entity_search()
    test_text_summarization()
    test_entity_network()
    test_save_load_clear()
    test_load_resources()
    test_linking_strategy()
    test_find_with_rules()
    test_load_resources()
    test_using_typed_words()
    test_build_word_ego_graph()
    test_entity_error_check()
    test_depend_parse()
    test_named_entity_recognition()