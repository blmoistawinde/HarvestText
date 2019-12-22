# HarvestText

Sow with little data seed, harvest much from a text field.

播撒几多种子词，收获万千领域实

![PyPI - Python Version](https://img.shields.io/badge/python-3.6-blue.svg) ![GitHub](https://img.shields.io/github/license/mashape/apistatus.svg) ![Version](https://img.shields.io/badge/version-V0.5-red.svg)

## 用途
HarvestText是一个专注无（弱）监督方法，能够整合领域知识（如类型，别名）对特定领域文本进行简单高效地处理和分析的库。适用于许多文本预处理和初步探索性分析任务，在小说分析，网络文本，专业文献等领域都有潜在应用价值。

使用案例:
- [分析《三国演义》中的社交网络](https://blog.csdn.net/blmoistawinde/article/details/85344906)（实体分词，文本摘要，关系网络等）
![网络建模过程示意.png](https://img-blog.csdnimg.cn/20181229200533159.png?x-oss-)
- [2018中超舆情展示系统](https://blmoistawinde.github.io/SuperLegal2018Display/index.html)（实体分词，情感分析，新词发现\[辅助绰号识别\]等）
相关文章：[一文看评论里的中超风云](https://blog.csdn.net/blmoistawinde/article/details/83443196)
![2018中超舆情展示系统](https://img-blog.csdnimg.cn/20181027084021173.png)

【注：本库仅完成实体分词和情感分析，可视化使用matplotlib】
- [近代史纲要信息抽取及问答系统](https://blog.csdn.net/blmoistawinde/article/details/86557070)(命名实体识别，依存句法分析，简易问答系统)

具体功能如下：

<a id="目录">目录:</a>
- 基本处理
	- [精细分词分句](#实体链接)
		- 可包含指定词和类别的分词。充分考虑省略号，双引号等特殊标点的分句。
	- [文本清洗(更新)](#文本清洗)
	    - 处理URL, email, 微博等文本中的特殊符号和格式
	- [实体链接](#实体链接)
		- 把别名，缩写与他们的标准名联系起来。 
	- [命名实体识别](#命名实体识别)
		- 找到一句句子中的人名，地名，机构名等命名实体。
	- [依存句法分析](#依存句法分析)
		- 分析语句中各个词语（包括链接到的实体）的主谓宾语修饰等语法关系，
	- [内置资源](#内置资源)
		- 通用停用词，通用情感词，IT、财经、饮食、法律等领域词典。可直接用于以上任务。
	- [信息检索](#信息检索)
		- 统计特定实体出现的位置，次数等。
	- [新词发现](#新词发现)
		- 利用统计规律（或规则）发现语料中可能会被传统分词遗漏的特殊词汇。也便于从文本中快速筛选出关键词。
	- [字符拼音纠错](#字符拼音纠错)
		- 把语句中有可能是已知实体的错误拼写（误差一个字符或拼音）的词语链接到对应实体。
	- [存取消除](#存取与消除)
		- 可以本地保存模型再读取复用，也可以消除当前模型的记录。
- 高层应用
	- [情感分析](#情感分析)
		- 给出少量种子词（通用的褒贬义词语），得到语料中各个词语和语段的褒贬度。
	- [关系网络](#关系网络)
		- 利用共现关系，获得关键词之间的网络。或者以一个给定词语为中心，探索与其相关的词语网络。
	- [文本摘要](#文本摘要)
		- 基于Textrank算法，得到一系列句子中的代表性句子。
	- [事实抽取](#依存句法分析)
		- 利用句法分析，提取可能表示事件的三元组。
	- [简易问答系统](#简易问答系统)
		- 从三元组中建立知识图谱并应用于问答，可以定制一些问题模板。效果有待提升，仅作为示例。

	
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

如果同一个名字有多个可能对应的实体（"打球的李娜和唱歌的李娜不是一个人"），可以设置`keep_all=True`来保留多个候选，后面可以再采用别的策略消歧，见[el_keep_all()](./examples/basics.py#L277)

如果连接到的实体过多，其中有一些明显不合理，可以采用一些策略来过滤，这里给出了一个例子[filter_el_with_rule()](./examples/basics.py#L284)

本库能够也用一些基本策略来处理复杂的实体消歧任务（比如一词多义【"老师"是指"A老师"还是"B老师"？】、候选词重叠【xx市长/江yy？、xx市长/江yy？】）。
具体可见[linking_strategy()](./examples/basics.py#L151)

<a id="文本清洗"> </a>

### 文本清洗

可以处理文本中的特殊字符，或者去掉文本中不希望出现的一些特殊格式。

包括：微博的@，表情符；网址；email；html代码中的&nbsp;一类的特殊字符；网址内的%20一类的特殊字符

例子如下：
```python
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
```

```
各种清洗文本
清洗微博【@和表情符等】
原： 回复@钱旭明QXM:[嘻嘻][嘻嘻] //@钱旭明QXM:杨大哥[good][good]
清洗后： 杨大哥
清洗网址URL
原： 【#赵薇#：正筹备下一部电影 但不是青春片....http://t.cn/8FLopdQ
清洗后： 【#赵薇#：正筹备下一部电影 但不是青春片....
清洗邮箱
原： 我的邮箱是abc@demo.com，欢迎联系
清洗后： 我的邮箱是，欢迎联系
URL转正常字符
原： www.%E4%B8%AD%E6%96%87%20and%20space.com
清洗后： www.中文 and space.com
正常字符转URL[含有中文和空格的request需要注意]
原： www.中文 and space.com
清洗后： www.%E4%B8%AD%E6%96%87%20and%20space.com
HTML转正常字符
原： &lt;a c&gt;&nbsp;&#x27;&#x27;
清洗后： <a c> ''
```

<a id="命名实体识别"> </a>
### 命名实体识别
找到一句句子中的人名，地名，机构名等命名实体。使用了 [pyhanLP](https://github.com/hankcs/pyhanlp) 的接口实现。

```python
ht0 = HarvestText()
sent = "上海上港足球队的武磊是中国最好的前锋。"
print(ht0.named_entity_recognition(sent))
```

```
{'上海上港足球队': '机构名', '武磊': '人名', '中国': '地名'}
```

<a id="依存句法分析"> </a>

### 依存句法分析
分析语句中各个词语（包括链接到的实体）的主谓宾语修饰等语法关系，并以此提取可能的事件三元组。使用了 [pyhanLP](https://github.com/hankcs/pyhanlp) 的接口实现。

```python
ht0 = HarvestText()
para = "上港的武磊武球王是中国最好的前锋。"
entity_mention_dict = {'武磊': ['武磊', '武球王'], "上海上港":["上港"]}
entity_type_dict = {'武磊': '球员', "上海上港":"球队"}
ht0.add_entities(entity_mention_dict, entity_type_dict)
for arc in ht0.dependency_parse(para):
    print(arc)
print(ht0.triple_extraction(para))
```

```
[0, '上港', '球队', '定中关系', 3]
[1, '的', 'u', '右附加关系', 0]
[2, '武磊', '球员', '定中关系', 3]
[3, '武球王', '球员', '主谓关系', 4]
[4, '是', 'v', '核心关系', -1]
[5, '中国', 'ns', '定中关系', 8]
[6, '最好', 'd', '定中关系', 8]
[7, '的', 'u', '右附加关系', 6]
[8, '前锋', 'n', '动宾关系', 4]
[9, '。', 'w', '标点符号', 4]
```
```python
print(ht0.triple_extraction(para))
```
```
[['上港武磊武球王', '是', '中国最好前锋']]
```

<a id="字符拼音纠错"> </a>

### 字符拼音纠错

把语句中有可能是已知实体的错误拼写（误差一个字符或拼音）的词语链接到对应实体。
```python
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
entity_error_check()
```

```
武磊和吴力只差一个拼音
[([0, 2], ('武磊', '#人名#')), [(3, 5), ('武磊', '#人名#')]]
武磊和吴磊只差一个字
[([0, 2], ('武磊', '#人名#')), [(3, 5), ('武磊', '#人名#')]]
吴磊和吴力都可能是武磊的代称
('吴磊和吴力都可能是武磊的代称', defaultdict(<class 'list'>, {(0, 2): {'武磊'}, (3, 5): {'武磊'}}))
```
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

如果没想好选择哪些词语作为“种子词”，本库中也内置了一个通用情感词典[内置资源](#内置资源)，在不指定情感词时作为默认的选择，也可以根据需要从中挑选。

默认使用的SO-PMI算法对于情感值没有上下界约束，如果需要限制在[0,1]或者[-1,1]这样的区间的话，可以调整scale参数，例子如下：

```python3
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
```

```
sentiment dictionary using default seed words
scale="0-1", 按照最大为1，最小为0进行线性伸缩，0.5未必是中性
赞同:1.000000
二十万:0.153846
万恶:0.000000
0.449412:张市筹设兴华实业公司外区资本家踊跃投资晋察冀边区兴华实业公司，自筹备成立以来，解放区内外企业界人士及一般商民，均踊跃认股投资
0.364910:打倒万恶的资本家
scale="+-1", 在正负区间内分别伸缩，保留0作为中性的语义
赞同:1.000000
二十万:0.000000
万恶:-1.000000
0.349305:张市筹设兴华实业公司外区资本家踊跃投资晋察冀边区兴华实业公司，自筹备成立以来，解放区内外企业界人士及一般商民，均踊跃认股投资
-0.159652:打倒万恶的资本家
```


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

获得以一个词语为中心的词语网络，下面以三国第一章为例，探索主人公刘备的遭遇（下为主要代码，例子见[build_word_ego_graph()](./examples/basics.py#)）。
```python3
entity_mention_dict, entity_type_dict = get_sanguo_entity_dict()
ht0.add_entities(entity_mention_dict, entity_type_dict)
sanguo1 = get_sanguo()[0]
stopwords = get_baidu_stopwords()
docs = ht0.cut_sentences(sanguo1)
G = ht0.build_word_ego_graph(docs,"刘备",min_freq=3,other_min_freq=2,stopwords=stopwords)
```
![word_ego_net](/images/word_ego_net.jpg)

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

可以本地保存模型再读取复用，也可以消除当前模型的记录。

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

<a id="简易问答系统"> </a>

### 简易问答系统

具体实现及例子在[naiveKGQA.py](./examples/naiveKGQA.py)中，下面给出部分示意：

```python
QA = NaiveKGQA(SVOs, entity_type_dict=entity_type_dict)
questions = ["你好","孙中山干了什么事？","谁发动了什么？","清政府签订了哪些条约？",
			 "英国与鸦片战争的关系是什么？","谁复辟了帝制？"]
for question0 in questions:
	print("问："+question0)
	print("答："+QA.answer(question0))
```

```
问：孙中山干了什么事？
答：就任临时大总统、发动护法运动、让位于袁世凯
问：谁发动了什么？
答：英法联军侵略中国、国民党人二次革命、英国鸦片战争、日本侵略朝鲜、孙中山护法运动、法国侵略越南、英国侵略中国西藏战争、慈禧太后戊戌政变
问：清政府签订了哪些条约？
答：北京条约、天津条约
问：英国与鸦片战争的关系是什么？
答：发动
问：谁复辟了帝制？
答：袁世凯
```

## More

本库正在开发中，关于现有功能的改善和更多功能的添加可能会陆续到来。欢迎在issues里提供意见建议。觉得好用的话，也不妨来个Star~

感谢以下repo带来的启发：

[snownlp](https://github.com/isnowfy/snownlp)

[pyhanLP](https://github.com/hankcs/pyhanlp)

[funNLP](https://github.com/fighting41love/funNLP)

[ChineseWordSegmentation](https://github.com/Moonshile/ChineseWordSegmentation)

[EventTriplesExtraction](https://github.com/liuhuanyong/EventTriplesExtraction)

