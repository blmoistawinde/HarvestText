import pickle
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
    new_words = ["落叶球", "666"]
    ht.add_new_words(new_words)
    print(ht.seg("这个落叶球踢得真是666", return_sent=True))
    for word, flag in ht.posseg("这个落叶球踢得真是666"):
        print("%s:%s" % (word, flag), end=" ")

def entity_segmentation():
    para = "上港的武磊和恒大的郜林，谁是中国最好的前锋？那当然是武磊武球王了，他是射手榜第一，原来是弱点的单刀也有了进步"
    print("add entity info(mention, type)")
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
    para = "上港的武磊和恒大的郜林，谁是中国最好的前锋？那当然是武磊武球王了，他是射手榜第一，原来是弱点的单刀也有了进步"
    with open("ht_model1", "wb") as f:
        pickle.dump(ht, f)
    with open("ht_model1", "rb") as f:
        ht2 = pickle.load(f)
    print("cut with loaded model")
    print(ht2.seg(para))
    ht2.clear()
    print("cut with cleared model")
    print(ht2.seg(para))


if __name__ == "__main__":
    new_word_discover()
    new_word_register()
    entity_segmentation()
    sentiment_dict()
    entity_search()
    text_summarization()
    entity_network()
    save_load_clear()