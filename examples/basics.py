#coding=utf-8
import re
from harvesttext import HarvestText
ht = HarvestText()

def new_word_discover():
    para = "上港的武磊和恒大的郜林，谁是中国最好的前锋？那当然是武磊武球王了，他是射手榜第一，原来是弱点的单刀也有了进步"
    # 返回关于新词质量的一系列信息，允许手工改进筛选(pd.DataFrame型)
    new_words_info = ht.word_discover(para)
    # new_words_info = ht.word_discover(para, threshold_seeds=["武磊"])
    new_words = new_words_info.index.tolist()
    print(new_words)

def new_word_register():
    new_words = ["落叶球","666"]
    ht.add_new_words(new_words)        # 作为广义上的"新词"登录
    ht.add_new_entity("落叶球", mention0="落叶球", type0="术语") # 作为特定类型登录
    print(ht.seg("这个落叶球踢得真是666", return_sent=True))
    for word, flag in ht.posseg("这个落叶球踢得真是666"):
        print("%s:%s" % (word, flag), end=" ")

def entity_segmentation():
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

def sentiment_dict():
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

def sentiment_dict_default():
    print("\nsentiment dictionary using default seed words")
    docs = ["张市筹设兴华实业公司外区资本家踊跃投资晋察冀边区兴华实业公司，自筹备成立以来，解放区内外企业界人士及一般商民，均踊跃认股投资",
            "打倒万恶的资本家",
        "该公司原定资本总额为二十五万万元，现已由各界分认达二十万万元，所属各厂、各公司亦募得股金一万万余元",
        "连日来解放区以外各工商人士，投函向该公司询问经营性质与范围以及股东权限等问题者甚多，络绎抵此的许多资本家，于参观该公司所属各厂经营状况后，对民主政府扶助与奖励私营企业发展的政策，均极表赞同，有些资本家因款项未能即刻汇来，多向筹备处预认投资的额数。由平津来张的林明棋先生，一次即以现款入股六十余万元"
       ]
    # scale: 将所有词语的情感值范围调整到[-1,1]
    # 省略pos_seeds, neg_seeds,将采用默认的情感词典 get_qh_sent_dict()
    print("scale=\"0-1\", 按照最大为1，最小为0进行线性伸缩，0.5未必是中性")
    sent_dict = ht.build_sent_dict(docs,min_times=1,scale="0-1")
    print("%s:%f" % ("赞同",sent_dict["赞同"]))
    print("%s:%f" % ("二十万",sent_dict["二十万"]))
    print("%s:%f" % ("万恶",sent_dict["万恶"]))
    print("%f:%s" % (ht.analyse_sent(docs[0]), docs[0]))
    print("%f:%s" % (ht.analyse_sent(docs[1]), docs[1]))

    print("scale=\"+-1\", 在正负区间内分别伸缩，保留0作为中性的语义")
    sent_dict = ht.build_sent_dict(docs,min_times=1,scale="+-1")
    print("%s:%f" % ("赞同",sent_dict["赞同"]))
    print("%s:%f" % ("二十万",sent_dict["二十万"]))
    print("%s:%f" % ("万恶",sent_dict["万恶"]))
    print("%f:%s" % (ht.analyse_sent(docs[0]), docs[0]))
    print("%f:%s" % (ht.analyse_sent(docs[1]), docs[1]))
    

def entity_search():
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

def text_summarization():
    # 文本摘要
    print("\nText summarization")
    docs = ["武磊威武，中超第一射手！",
            "郜林看来不行，已经到上限了。",
            "武球王威武，中超最强前锋！",
            "武磊和郜林，谁是中国最好的前锋？"]
    for doc in ht.get_summary(docs, topK=2):
        print(doc)

def entity_network():
    print("\nentity network")
    # 在现有实体库的基础上随时新增，比如从新词发现中得到的漏网之鱼
    ht.add_new_entity("颜骏凌", "颜骏凌", "球员")
    docs = ["武磊和颜骏凌是队友",
            "武磊和郜林都是国内顶尖前锋"]
    G = ht.build_entity_graph(docs)
    print(dict(G.edges.items()))
    G = ht.build_entity_graph(docs, used_types=["球员"])
    print(dict(G.edges.items()))

def save_load_clear():
    from harvesttext import loadHT,saveHT
    para = "上港的武磊和恒大的郜林，谁是中国最好的前锋？那当然是武磊武球王了，他是射手榜第一，原来是弱点的单刀也有了进步"
    saveHT(ht,"ht_model1")
    ht2 = loadHT("ht_model1")
    print("cut with loaded model")
    print(ht2.seg(para))
    ht2.clear()
    print("cut with cleared model")
    print(ht2.seg(para))

def load_resources():
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

def linking_strategy():
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


def find_with_rules():
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

def build_word_ego_graph():
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
    plt.show()
    G = ht0.build_entity_ego_graph(docs, "刘备", min_freq=3, other_min_freq=2)
    pos = nx.spring_layout(G)
    nx.draw(G, pos)
    nx.draw_networkx_labels(G, pos)
    plt.show()

def using_typed_words():
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

def entity_error_check():
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

def depend_parse():
    ht0 = HarvestText()
    para = "上港的武磊武球王是中国最好的前锋。"
    entity_mention_dict = {'武磊': ['武磊', '武球王'], "上海上港":["上港"]}
    entity_type_dict = {'武磊': '球员', "上海上港":"球队"}
    ht0.add_entities(entity_mention_dict, entity_type_dict)
    for arc in ht0.dependency_parse(para):
        print(arc)
    print(ht0.triple_extraction(para))

def named_entity_recognition():
    ht0 = HarvestText()
    sent = "上海上港足球队的武磊是中国最好的前锋。"
    print(ht0.named_entity_recognition(sent))

def el_keep_all():
    ht0 = HarvestText()
    entity_mention_dict = {'李娜1': ['李娜'], "李娜2":['李娜']}
    entity_type_dict = {'李娜1': '运动员', '李娜2': '歌手'}
    ht0.add_entities(entity_mention_dict, entity_type_dict)
    print(ht0.entity_linking("打球的李娜和唱歌的李娜不是一个人", keep_all=True))

def filter_el_with_rule():
    # 当候选实体集很大的时候，实体链接得到的指称重可能有很多噪声，可以利用一些规则进行筛选
    # 1. 词性：全部由动词v，形容词a, 副词d, 连词c，介词p等组成的，一般不是传统意义上会关心的实体
    # 2. 词长：指称长度只有1的，一般信息不足
    # 由于这些规则可以高度定制化，所以不直接写入库中，而在外部定义。这段代码提供一个示例：
    def el_filtering(entities_info, ch_pos):
        return [([l, r], (entity0, type0)) for [l, r], (entity0, type0) in entities_info
                if not all(bool(re.search("^(v|a|d|c|p|y|z)", pos)) for pos in ch_pos[l:r])
                and (r-l) > 1]
    ht0 = HarvestText()
    text = "《记得》：谁还记得 是谁先说 永远的爱我"
    entity_mention_dict = {'记得（歌曲）': ['记得', '《记得》'], "我（张国荣演唱歌曲）": ['我', '《我》']}
    entity_type_dict = {'记得（歌曲）': '歌名', '我（张国荣演唱歌曲）': '歌名'}
    ht0.add_entities(entity_mention_dict, entity_type_dict)
    entities_info, ch_pos = ht0.entity_linking(text, with_ch_pos=True)  # 显式设定了with_ch_pos=True才有
    print("filter_el_with_rule")
    print("Sentence:", text)
    print("Original Entities:", entities_info)
    filtered_entities = el_filtering(entities_info, ch_pos)
    # 我 因为词长被过滤，而 记得 因为是纯动词而被过滤，但是《记得》包括了标点，不会被过滤
    print("filtered_entities:", filtered_entities)

def clean_text():
    print("各种清洗文本")
    ht0 = HarvestText()
    # 默认的设置可用于清洗微博文本
    text1 = "回复@钱旭明QXM:[嘻嘻][嘻嘻] //@钱旭明QXM:杨大哥[good][good]"
    print("清洗微博【@和表情符等】")
    print("原：", text1)
    print("清洗后：", ht0.clean_text(text1))
    # URL的清理
    text1 = "【#赵薇#：正筹备下一部电影 但不是青春片....http://t.cn/8FLopdQ"
    print("清洗网址URL")
    print("原：", text1)
    print("清洗后：", ht0.clean_text(text1, remove_url=True))
    # 清洗邮箱
    text1 = "我的邮箱是abc@demo.com，欢迎联系"
    print("清洗邮箱")
    print("原：", text1)
    print("清洗后：", ht0.clean_text(text1, email=True))
    # 处理URL转义字符
    text1 = "www.%E4%B8%AD%E6%96%87%20and%20space.com"
    print("URL转正常字符")
    print("原：", text1)
    print("清洗后：", ht0.clean_text(text1, norm_url=True, remove_url=False))
    text1 = "www.中文 and space.com"
    print("正常字符转URL[含有中文和空格的request需要注意]")
    print("原：", text1)
    print("清洗后：", ht0.clean_text(text1, to_url=True, remove_url=False))
    # 处理HTML转义字符
    text1 = "&lt;a c&gt;&nbsp;&#x27;&#x27;"
    print("HTML转正常字符")
    print("原：", text1)
    print("清洗后：", ht0.clean_text(text1, norm_html=True))

if __name__ == "__main__":
    new_word_discover()
    new_word_register()
    entity_segmentation()
    sentiment_dict()
    sentiment_dict_default()
    entity_search()
    text_summarization()
    entity_network()
    save_load_clear()
    load_resources()
    linking_strategy()
    find_with_rules()
    load_resources()
    using_typed_words()
    build_word_ego_graph()
    entity_error_check()
    depend_parse()
    named_entity_recognition()
    el_keep_all()
    filter_el_with_rule()
    clean_text()
