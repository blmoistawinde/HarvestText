# HarvestText

Sow with little data seed, harvest much from a text field.

播撒几多种子词，收获万千领域实

![PyPI - Python Version](https://img.shields.io/badge/python-3.6-blue.svg) ![GitHub](https://img.shields.io/github/license/mashape/apistatus.svg) ![Version](https://img.shields.io/badge/version-V0.3-red.svg)

## 用途
HarvestText是一个专注无（弱）监督方法，能够整合领域知识（如类型，别名）对特定领域文本进行简单高效地处理和分析的库。

具体功能如下：

<a id="目录">目录:</a>
- [精细分词分句](#实体链接)
	- 可包含指定词和类别的分词。充分考虑省略号，双引号等特殊标点的分句。
- [实体链接](#实体链接)
	- 把别名，缩写与他们的标准名联系起来。 
- [情感分析](#情感分析)
	- 给出少量种子词（通用的褒贬义词语），得到语料中各个词语和语段的褒贬度。
- [信息检索](#信息检索)
	- 统计特定实体出现的位置，次数等。
- [关系网络](#关系网络)
	- 利用共现关系，获得关键词之间的网络。或者以一个给定词语为中心，探索与其相关的词语网络。
- [内置资源](#内置资源)
	- 通用停用词，通用情感词，IT、财经、饮食、法律等领域词典。可直接用于以上任务。
- [新词发现](#新词发现)
	- 利用统计规律（或规则）发现语料中可能会被传统分词遗漏的特殊词汇。也便于从文本中快速筛选出关键词。
- [文本摘要](#文本摘要)
	- 基于Textrank得到一系列句子中的代表性句子中。
- [存取消除](#存取与消除)
	- 可以本地保存模型再读取复用(pickle)，也可以消除当前模型的记录。
	
在很多领域文本分析中，我们往往已经了解其中的一些关键词语或实体，例如小说文本分析中的人物名，电影评论中的演员名、角色名、影片名，足球评论文本中的球员、球队、乃至一些术语。在后面的分析中，它们可能是我们的重点关注对象，或者是可以利用它们来改进分词等基础任务、提供机器学习的一些基础特征。

内置停用词，特殊类型词，情感词等资源，并以简单清晰的方式与本库的分词、分析等流程结合。

本库就旨在于提供解决这些问题的一个简单易用的方案。

## 依赖
- jieba
- numpy, pandas
- networkx
	
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

<a id="实体链接"> </a>
### 实体链接
给定某些实体及其可能的代称，以及实体对应类型。将其登录到词典中，在分词时优先切分出来，并且以对应类型作为词性。也可以单独获得语料中的所有实体及其位置：

```python3
para = "上港的武磊和恒大的郜林，谁是中国最好的前锋？那当然是武磊武球王了，他是射手榜第一，原来是弱点的单刀也有了进步"
entity_mention_dict = {'武磊':['武磊','武球王'],'郜林':['郜林','郜飞机'],'前锋':['前锋'],'上海上港':['上港'],'广州恒大':['恒大'],'单刀球':['单刀']}
entity_type_dict = {'武磊':'球员','郜林':'球员','前锋':'位置','上海上港':'球队','广州恒大':'球队','单刀球':'术语'}
ht.add_entities(entity_mention_dict,entity_type_dict)
print("\nSentence segmentation")
print(ht.seg(para,return_sent=True))    # return_sent=False时，则返回词语列表
```
	
> 上港 的 武磊 和 恒大 的 郜林 ， 谁 是 中国 最好 的 前锋 ？ 那 当然 是 武磊 武球王 了， 他 是 射手榜 第一 ， 原来 是 弱点 的 单刀 也 有 了 进步

采用传统的分词工具很容易把“武球王”拆分为“武 球王”

词性标注，包括指定的特殊类型。
```python3
print("\nPOS tagging with entity types")
for word, flag in ht.posseg(para):
	print("%s:%s" % (word, flag),end = " ")
```

> 上港:球队 的:uj 武磊:球员 和:c 恒大:球队 的:uj 郜林:球员 ，:x 谁:r 是:v 中国:ns 最好:a 的:uj 前锋:位置 ？:x 那:r 当然:d 是:v 武磊:球员 武球王:球员 了:ul ，:x 他:r 是:v 射手榜:n 第一:m ，:x 原来:d 是:v 弱点:n 的:uj 单刀:术语 也:d 有:v 了:ul 进步:d 

```python3
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

分句：
```python3
print(ht.cut_sentences(para))
```

> ['上港的武磊和恒大的郜林，谁是中国最好的前锋？', '那当然是武磊武球王了，他是射手榜第一，原来是弱点的单刀也有了进步']

如果手头暂时没有可用的词典，不妨看看本库[内置资源](#内置资源)中的领域词典是否适合你的需要。

\*现在本库能够也用一些基本策略来处理复杂的实体消歧任务（比如一词多义【"老师"是指"A老师"还是"B老师"？】、候选词重叠【xx市长/江yy？、xx市长/江yy？】）。
具体可见[linking_strategy()](./examples/basics.py#linking_strategy)

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

<a id="信息检索"> </a>
### 信息检索
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

<a id="关系网络"> </a>
### 关系网络
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

获得以一个词语为中心的词语网络，下面以三国第一章为例，探索主人公刘备的遭遇（下为主要代码，例子见[build_word_ego_graph()](./examples/basics.py#linking_strategy)）。
```python3
entity_mention_dict, entity_type_dict = get_sanguo_entity_dict()
ht0.add_entities(entity_mention_dict, entity_type_dict)
sanguo1 = get_sanguo()[0]
stopwords = get_baidu_stopwords()
docs = ht0.cut_sentences(sanguo1)
G = ht0.build_word_ego_graph(docs,"刘备",min_freq=3,other_min_freq=2,stopwords=stopwords)
```
![word_ego_net](file:///./images/word_ego_net.jpg)

刘关张之情谊，刘备投奔的靠山，以及刘备讨贼之经历尽在于此。

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


<a id="内置资源"> </a>
### 内置资源
现在本库内集成了一些资源，方便使用和建立demo。

资源包括：
- 褒贬义词典 清华大学 李军 整理自http://nlp.csai.tsinghua.edu.cn/site2/index.php/13-sms
- 百度停用词词典 来自网络：https://wenku.baidu.com/view/98c46383e53a580216fcfed9.html
- 领域词典 来自清华THUNLP： http://thuocl.thunlp.org/ 全部类型`['IT', '动物', '医药', '历史人名', '地名', '成语', '法律', '财经', '食物']`


此外，还提供了一个特殊资源——《三国演义》，包括：
- 三国演义文言文文本
- 三国演义人名、州名、势力知识库

大家可以探索从其中能够得到什么有趣发现😁。

```python3
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
load_resources()
```

```
pos_words: ['宰相肚里好撑船', '查实', '忠实', '名手', '聪明']
neg_words: ['散漫', '谗言', '迂执', '肠肥脑满', '出卖']
stopwords: ['apart', '左右', '结果', 'probably', 'think']
三国演义最后一章末16字:
 鼎足三分已成梦，后人凭吊空牢骚。
刘备 指称： ['刘备', '刘玄德', '玄德']
刘备 类别： 人名
蜀 类别： 势力
益州 类别： 州名
```

加载清华领域词典，并使用停用词。
```python3
def using_typed_words():
    from harvesttext.resources import get_qh_typed_words,get_baidu_stopwords
    ht0 = HarvestText()
    typed_words, stopwords = get_qh_typed_words(), get_baidu_stopwords()
    ht0.add_typed_words(typed_words)
    sentence = "THUOCL是自然语言处理的一套中文词库，词表来自主流网站的社会标签、搜索热词、输入法词库等。"
    print(sentence)
    print(ht0.posseg(sentence,stopwords=stopwords))
using_typed_words()
```

```
THUOCL是自然语言处理的一套中文词库，词表来自主流网站的社会标签、搜索热词、输入法词库等。
[('THUOCL', 'eng'), ('自然语言处理', 'IT'), ('一套', 'm'), ('中文', 'nz'), ('词库', 'n'), ('词表', 'n'), ('来自', 'v'), ('主流', 'b'), ('网站', 'n'), ('社会', 'n'), ('标签', '财经'), ('搜索', 'v'), ('热词', 'n'), ('输入法', 'IT'), ('词库', 'n')]
```

一些词语被赋予特殊类型IT,而“是”等词语被筛出。


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

发现的新词很多都可能是文本中的特殊关键词，故可以把找到的新词登录，使后续的分词优先分出这些词。
```python3
def new_word_register():
    new_words = ["落叶球","666"]
    ht.add_new_words(new_words)   # 作为广义上的"新词"登录
    ht.add_new_entity("落叶球", mention0="落叶球", type0="术语")  # 作为特定类型登录
    print(ht.seg("这个落叶球踢得真是666", return_sent=True))
    for word, flag in ht.posseg("这个落叶球踢得真是666"):
        print("%s:%s" % (word, flag), end=" ")
```
> 这个 落叶球 踢 得 真是 666

> 这个:r 落叶球:术语 踢:v 得:ud 真是:d 666:新词 

也可以使用一些特殊的*规则*来找到所需的关键词，并直接赋予类型，比如全英文，或者有着特定的前后缀等。
```python3
# find_with_rules()
from harvesttext.match_patterns import UpperFirst, AllEnglish, Contains, StartsWith, EndsWith
text0 = "我喜欢Python，因为requests库很适合爬虫"
ht0 = HarvestText()

found_entities = ht0.find_entity_with_rule(text0, rulesets=[AllEnglish()], type0="英文名")
print(found_entities)
print(ht0.posseg(text0))
```

```
{'Python', 'requests'}
[('我', 'r'), ('喜欢', 'v'), ('Python', '英文名'), ('，', 'x'), ('因为', 'c'), ('requests', '英文名'), ('库', 'n'), ('很', 'd'), ('适合', 'v'), ('爬虫', 'n')]
```
	

<a id="存取与消除"> </a>
### 存取消除
可以本地保存模型再读取复用(pickle)，也可以消除当前模型的记录。
```python3
from harvesttext import loadHT,saveHT
para = "上港的武磊和恒大的郜林，谁是中国最好的前锋？那当然是武磊武球王了，他是射手榜第一，原来是弱点的单刀也有了进步"
saveHT(ht,"ht_model1")
ht2 = loadHT("ht_model1")

# 消除记录
ht2.clear()
print("cut with cleared model")
print(ht2.seg(para))
```
## More
本库正在开发中，关于现有功能的改善和更多功能的添加可能会陆续到来。欢迎在issues里提供意见建议。觉得好用的话，也不妨来个Star~

感谢以下repo带来的启发：
[snownlp](https://github.com/isnowfy/snownlp)

[funNLP](https://github.com/fighting41love/funNLP)

[ChineseWordSegmentation](https://github.com/Moonshile/ChineseWordSegmentation)