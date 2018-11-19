# HarvestText

Sow with little data seed, harvest much from a text field.

播撒几多种子词，收获万千领域实

![PyPI - Python Version](https://img.shields.io/badge/python-3.6-blue.svg) ![GitHub](https://img.shields.io/github/license/mashape/apistatus.svg) ![Version](https://img.shields.io/badge/version-V0.1-red.svg)

## 用途
HarvestText是一个基于少量种子词和背景知识完成一些领域自适应文本挖掘任务（如新词发现、情感分析、实体链接等）的工具。
	
在很多领域文本分析中，我们往往已经了解其中的一些关键词语或实体，例如小说文本分析中的人物名，电影评论中的演员名、角色名、影片名，足球评论文本中的球员、球队、乃至一些术语。在后面的分析中，它们可能是我们的重点关注对象，或者是可以利用它们来改进分词等基础任务、提供机器学习的一些基础特征。
	
本库就旨在于提供解决这些问题的一个简单易用的方案。

## 依赖
- jieba
- numpy, pandas
	
## 用法
目前还没有实现安装功能，使用方法是把本目录下的3个py文件放入你当前文件的工作目录，然后使用：

```python3
from HarvestText import HarvestText
ht = HarvestText()
```

即可调用本库的功能接口。
	
1.新词发现
从比较大量的文本中利用一些统计指标发现新词。（可选）通过提供一些种子词语来确定怎样程度质量的词语可以被发现。（即至少所有的种子词会被发现，在满足一定的基础要求的前提下。）
```python3
para = "上港的武磊和恒大的郜林，谁是中国最好的前锋？武磊吧，他是射手榜第一，原来是弱点的单刀也有了进步"
#返回关于新词质量的一系列信息，允许手工改进筛选(pd.DataFrame型)
new_words_info = ht.word_discover(para)
#new_words_info = ht.word_discover(para, threshold_seeds=["武磊"])  
new_words = new_words_info.index.tolist()
print(new_words)
```

> ["武磊"]

具体的方法和指标含义，参考：http://www.matrix67.com/blog/archives/5044
	
2.实体链接
给定某些实体及其可能的代称，以及实体对应类型。将其登录到词典中，在分词时优先切分出来，并且以对应类型作为词性。也可以单独获得语料中的所有实体及其位置：

```python3
print("add entity info(mention, type)")
entity_mention_dict = {'武磊':['武磊','武球王'],'郜林':['郜林','郜飞机'],'前锋':['前锋'],'上海上港':['上港'],'广州恒大':['恒大'],'单刀球':['单刀']}
entity_type_dict = {'武磊':'球员','郜林':'球员','前锋':'位置','上海上港':'球队','广州恒大':'球队','单刀球':'术语'}
ht.add_entities(entity_mention_dict,entity_type_dict)
print("\nSentence segmentation")
print(ht.seg(para,return_sent=True))    # return_sent=False时，则返回词语列表
```
	
> 上港 的 武磊 和 恒大 的 郜林 ， 谁 是 中国 最好 的 前锋 ？ 那 当然 是 武球王 ， 他 是 射手榜 第一 ， 原来 是 弱点 的 单刀 也 有 了 进步

采用传统的分词工具很容易把“武球王”拆分为“武 球王”
	
```python3
print("\nPOS tagging with entity types")
for word, flag in ht.posseg(para):
	print("%s:%s" % (word, flag),end = " ")
```

> 上港:球队 的:uj 武磊:球员 和:c 恒大:球队 的:uj 郜林:球员 ，:x 谁:r 是:v 中国:ns 最好:a 的:uj 前锋:位置 ？:x 武磊:球员 吧:y ，:x 他:r 是:v 射手榜:n 第一:m ，:x 原来:d 是:v 弱点:n 的:uj 单刀:术语 也:d 有:v 了:ul 进步:d 

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
[26, 29] ('武磊', '#球员#')
[44, 46] ('单刀球', '#术语#')

这里把“武球王”转化为了标准指称“武磊”，可以便于标准统一的统计工作。

3. 情感分析
本库采用情感词典方法进行情感分析，通过提供少量标准的褒贬义词语，从语料中自动学习其他词语的情感倾向，形成情感词典。对句中情感词的加总平均则用于判断句子的情感倾向：
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

## More
本库正在开发中，关于现有功能的改善和更多功能的添加可能会陆续到来。欢迎在issues里提供意见建议。觉得好用的话，也不妨来个Star~