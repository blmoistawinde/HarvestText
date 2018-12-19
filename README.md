# HarvestText

Sow with little data seed, harvest much from a text field.

播撒几多种子词，收获万千领域实

![PyPI - Python Version](https://img.shields.io/badge/python-3.6-blue.svg) ![GitHub](https://img.shields.io/github/license/mashape/apistatus.svg) ![Version](https://img.shields.io/badge/version-V0.3-red.svg)

## 用途
HarvestText是一个基于少量种子词和背景知识完成一些领域自适应文本挖掘任务（如新词发现、情感分析、实体链接等）的工具。
	
在很多领域文本分析中，我们往往已经了解其中的一些关键词语或实体，例如小说文本分析中的人物名，电影评论中的演员名、角色名、影片名，足球评论文本中的球员、球队、乃至一些术语。在后面的分析中，它们可能是我们的重点关注对象，或者是可以利用它们来改进分词等基础任务、提供机器学习的一些基础特征。
	
本库就旨在于提供解决这些问题的一个简单易用的方案。

## 依赖
- jieba
- numpy, pandas
- networkx(可选)
	
## 用法


首先安装，
使用pip
```
pip install harvesttext
```

或进入setup.py所在目录，然后命令行:
```
python setup.py install
```

随后在代码中：

```python3
from harvesttext import HarvestText
ht = HarvestText()
```

即可调用本库的功能接口。

<a id="目录">目录:</a>
- [新词发现](#新词发现)
- [实体链接](#实体链接)
- [情感分析](#情感分析)
- [信息检索（考虑实体消歧）](#信息检索（考虑实体消歧）)
- [文本摘要](#文本摘要)
- [实体网络](#实体网络)
- [内置资源](#内置资源)
- [存取与消除](#存取与消除)
	
<a id="新词发现"> </a>
### 新词发现
从比较大量的文本中利用一些统计指标发现新词。（可选）通过提供一些种子词语来确定怎样程度质量的词语可以被发现。（即至少所有的种子词会被发现，在满足一定的基础要求的前提下。）
```python3
para = "上港的武磊和恒大的郜林，谁是中国最好的前锋？那当然是武磊武球王了，他是射手榜第一，原来是弱点的单刀也有了进步"
#返回关于新词质量的一系列信息，允许手工改进筛选(pd.DataFrame型)
new_words_info = ht.word_discover(para)
#new_words_info = ht.word_discover(para, threshold_seeds=["武磊"])  
new_words = new_words_info.index.tolist()
print(new_words)
```

> ["武磊"]

具体的方法和指标含义，参考：http://www.matrix67.com/blog/archives/5044

可以把找到的新词登录，后续的分词中将会优先分出这些词，并把词性标注为"新词"
```python3
new_words = ["落叶球","666"]
ht.add_new_words(new_words)
print(ht.seg("这个落叶球踢得真是666",return_sent=True))
for word, flag in ht.posseg("这个落叶球踢得真是666"):
	print("%s:%s" % (word, flag),end = " ")
```
> 这个 落叶球 踢 得 真是 666

> 这个:r 落叶球:新词 踢:v 得:ud 真是:d 666:新词 
	
<a id="实体链接"> </a>
### 实体链接
给定某些实体及其可能的代称，以及实体对应类型。将其登录到词典中，在分词时优先切分出来，并且以对应类型作为词性。也可以单独获得语料中的所有实体及其位置：

```python3
print("add entity info(mention, type)")
entity_mention_dict = {'武磊':['武磊','武球王'],'郜林':['郜林','郜飞机'],'前锋':['前锋'],'上海上港':['上港'],'广州恒大':['恒大'],'单刀球':['单刀']}
entity_type_dict = {'武磊':'球员','郜林':'球员','前锋':'位置','上海上港':'球队','广州恒大':'球队','单刀球':'术语'}
ht.add_entities(entity_mention_dict,entity_type_dict)
print("\nSentence segmentation")
print(ht.seg(para,return_sent=True))    # return_sent=False时，则返回词语列表
```
	
> 上港 的 武磊 和 恒大 的 郜林 ， 谁 是 中国 最好 的 前锋 ？ 那 当然 是 武磊 武球王 了， 他 是 射手榜 第一 ， 原来 是 弱点 的 单刀 也 有 了 进步

采用传统的分词工具很容易把“武球王”拆分为“武 球王”
	
```python3
print("\nPOS tagging with entity types")
for word, flag in ht.posseg(para):
	print("%s:%s" % (word, flag),end = " ")
```

> 上港:球队 的:uj 武磊:球员 和:c 恒大:球队 的:uj 郜林:球员 ，:x 谁:r 是:v 中国:ns 最好:a 的:uj 前锋:位置 ？:x 那:r 当然:d 是:v 武磊:球员 武球王:球员 了:ul ，:x 他:r 是:v 射手榜:n 第一:m ，:x 原来:d 是:v 弱点:n 的:uj 单刀:术语 也:d 有:v 了:ul 进步:d 

```python3
print("\n\nentity_linking")
for span, entity in ht.entity_linking(para):
	print(span, entity)
```

> [0, 2] ('上海上港', '#球队#')
[3, 5] ('武磊', '#球员#')
[6, 8] ('广州恒大', '#球队#')
[9, 11] ('郜林', '#球员#')
[19, 21] ('前锋', '#位置#')
[26, 28] ('武磊', '#球员#')
[28, 31] ('武磊', '#球员#')
[47, 49] ('单刀球', '#术语#')

这里把“武球王”转化为了标准指称“武磊”，可以便于标准统一的统计工作。

(V0.3) 现在本库能够用一些基本策略来处理复杂的实体消歧任务（比如一词多义【"老师"是指"A老师"还是"B老师"？】、候选词重叠【xx市长/江yy？、xx市长/江yy？】）。
具体可见[linking_strategy()](./examples/basics.py)

<a id="情感分析"> </a>
### 情感分析
本库采用情感词典方法进行情感分析，通过提供少量标准的褒贬义词语（“种子词”），从语料中自动学习其他词语的情感倾向，形成情感词典。对句中情感词的加总平均则用于判断句子的情感倾向：
```python3
print("\nsentiment dictionary")
sents = ["武磊威武，中超第一射手！",
      "武磊强，中超最第一本土球员！",
      "郜林不行，只会抱怨的球员注定上限了",
      "郜林看来不行，已经到上限了"]
sent_dict = ht.build_sent_dict(sents,min_times=1,pos_seeds=["第一"],neg_seeds=["不行"])
print("%s:%f" % ("威武",sent_dict["威武"]))
print("%s:%f" % ("球员",sent_dict["球员"]))
print("%s:%f" % ("上限",sent_dict["上限"]))
```

> sentiment dictionary 
> 威武:1.000000 
> 球员:0.000000 
> 上限:-1.000000

```python3
print("\nsentence sentiment")
sent = "武球王威武，中超最强球员！"
print("%f:%s" % (ht.analyse_sent(sent),sent))
```
> 0.600000:武球王威武，中超最强球员！

如果没想好选择哪些词语作为“种子词”，本库中也内置了一个通用情感词典[内置资源](#内置资源)，可以从中挑选。

<a id="信息检索（考虑实体消歧）"> </a>
### 信息检索（考虑实体消歧）
可以从文档列表中查找出包含对应实体（及其别称）的文档，以及统计包含某实体的文档数。使用倒排索引的数据结构完成快速检索。
```python3
docs = ["武磊威武，中超第一射手！",
		"郜林看来不行，已经到上限了。",
		"武球王威武，中超最强前锋！",
		"武磊和郜林，谁是中国最好的前锋？"]
inv_index = ht.build_index(docs)
print(ht.get_entity_counts(docs, inv_index))  # 获得文档中所有实体的出现次数
# {'武磊': 3, '郜林': 2, '前锋': 2}

print(ht.search_entity("武磊", docs, inv_index))  # 单实体查找
# ['武磊威武，中超第一射手！', '武球王威武，中超最强前锋！', '武磊和郜林，谁是中国最好的前锋？']

print(ht.search_entity("武磊 郜林", docs, inv_index))  # 多实体共现
# ['武磊和郜林，谁是中国最好的前锋？']

# 谁是最被人们热议的前锋？用这里的接口可以很简便地回答这个问题
subdocs = ht.search_entity("#球员# 前锋", docs, inv_index)
print(subdocs)  # 实体、实体类型混合查找
# ['武球王威武，中超最强前锋！', '武磊和郜林，谁是中国最好的前锋？']
inv_index2 = ht.build_index(subdocs)
print(ht.get_entity_counts(subdocs, inv_index2, used_type=["球员"]))  # 可以限定类型
# {'武磊': 2, '郜林': 1}
```

<a id="文本摘要"> </a>
### 文本摘要
(使用networkx实现)
使用Textrank算法，得到从文档集合中抽取代表句作为摘要信息：
```python3
print("\nText summarization")
docs = ["武磊威武，中超第一射手！",
		"郜林看来不行，已经到上限了。",
		"武球王威武，中超最强前锋！",
		"武磊和郜林，谁是中国最好的前锋？"]
for doc in ht.get_summary(docs, topK=2):
	print(doc)
# 武球王威武，中超最强前锋！
# 武磊威武，中超第一射手！	
```

<a id="实体网络"> </a>
### 实体网络
(使用networkx实现)
利用词共现关系，建立其实体间图结构的网络关系(返回networkx.Graph类型)。可以用来建立人物之间的社交网络等。
```python3
# 在现有实体库的基础上随时新增，比如从新词发现中得到的漏网之鱼
ht.add_new_entity("颜骏凌", "颜骏凌", "球员")
docs = ["武磊和颜骏凌是队友",
		"武磊和郜林都是国内顶尖前锋"]
G = ht.build_entity_graph(docs)
print(dict(G.edges.items()))
G = ht.build_entity_graph(docs, used_types=["球员"])
print(dict(G.edges.items()))
```

<a id="内置资源"> </a>
### 内置资源
现在本库内集成了一些资源，方便使用和建立demo。

资源包括：
- 褒贬义词典 清华大学 李军 整理自http://nlp.csai.tsinghua.edu.cn/site2/index.php/13-sms
- 三国演义文言文文本
- 三国演义人名、州名、势力知识库

```python3
def load_resources():
    from harvesttext import get_qh_sent_dict, get_sanguo, get_sanguo_entity_dict
    
	sdict = get_qh_sent_dict()  # {"pos":[积极词...],"neg":[消极词...]}
    print("pos_words:",sdict["pos"][:5])
    print("neg_words:",sdict["neg"][:5])
    
	docs = get_sanguo()     # 文本列表，每个元素为一章的文本
    print("三国演义最后一章末16字:\n",docs[-1][-16:])
    
	entity_mention_dict, entity_type_dict = get_sanguo_entity_dict()
    print("刘备 指称：",entity_mention_dict["刘备"])
    print("刘备 类别：",entity_type_dict["刘备"])
	print("蜀 类别：", entity_type_dict["蜀"])
    print("益州 类别：", entity_type_dict["益州"])
load_resources()
```

```
pos_words: ['遂意', '得救', '稳帖', '谦诚', '赞成']
neg_words: ['乱离', '下流', '挑刺儿', '憾事', '日暮途穷']
三国演义最后一章末16字:
 鼎足三分已成梦，后人凭吊空牢骚。
刘备 指称： ['刘备', '刘玄德', '玄德']
刘备 类别： 人名
蜀 类别： 势力
益州 类别： 州名
```

<a id="存取与消除"> </a>
### 存取与消除
可以本地保存模型再读取复用(pickle)，也可以消除当前模型的记录。
```python3
from harvesttext import loadHT,saveHT
para = "上港的武磊和恒大的郜林，谁是中国最好的前锋？那当然是武磊武球王了，他是射手榜第一，原来是弱点的单刀也有了进步"
saveHT(ht,"ht_model1")
ht2 = loadHT("ht_model1")
print("cut with loaded model")
print(ht2.seg(para))
ht2.clear()
print("cut with cleared model")
print(ht2.seg(para))

# 消除记录
ht2.clear()
print("cut with cleared model")
print(ht2.seg(para))
```
## More
本库正在开发中，关于现有功能的改善和更多功能的添加可能会陆续到来。欢迎在issues里提供意见建议。觉得好用的话，也不妨来个Star~