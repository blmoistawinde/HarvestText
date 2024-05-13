import re
from harvesttext import HarvestText

def test_hard_text_cleaning():
    ht = HarvestText()
    # 不可见字符
    text1 = "捧杀！干得漂亮！[doge] \\u200b\\u200b\\u200b"
    text2 = ht.clean_text(text1)
    print("清洗前：", [text1])
    print("清洗后：", [text2])
    assert text2 == "捧杀！干得漂亮！"
    text1 = "捧杀！干得漂亮！[doge] \u200b\u200b\u200b"
    text2 = ht.clean_text(text1)
    assert text2 == "捧杀！干得漂亮！"
    print("清洗前：", [text1])
    print("清洗后：", [text2])
    # 两个表情符号中间有内容
    text1 = "#缺钱找新浪# 瞎找不良网贷不如用新浪官方借款，不查负债不填联系人。  http://t.cn/A643boyi \n新浪[浪]用户专享福利，[浪]新浪产品用的越久额度越高，借万元日利率最低至0.03%，最长可分12期慢慢还！ http://t.cn/A643bojv  http://t.cn/A643bKHS \u200b\u200b\u200b"
    text2 = ht.clean_text(text1)
    print("清洗前：", [text1])
    print("清洗后：", [text2])
    assert text2 == "#缺钱找新浪# 瞎找不良网贷不如用新浪官方借款，不查负债不填联系人。\n新浪用户专享福利，新浪产品用的越久额度越高，借万元日利率最低至0.03%，最长可分12期慢慢还！"
    # 包含emoji
    text1 = "各位大神们🙏求教一下这是什么动物呀！[疑问]\n\n为什么它同时长得有点吓人又有点可爱[允悲]\n\n#thosetiktoks# http://t.cn/A6bXIC44 \u200b\u200b\u200b"
    text2 = ht.clean_text(text1)
    print("清洗前：", [text1])
    print("清洗后：", [text2])
    assert text2 == "各位大神们求教一下这是什么动物呀！\n为什么它同时长得有点吓人又有点可爱\n#thosetiktoks#"
    text1 = "JJ棋牌数据4.3万。数据链接http://www.jj.cn/，数据第一个账号，第二个密码，95%可登录，可以登录官网查看数据是否准确"
    text2 = ht.clean_text(text1)
    assert text2 == "JJ棋牌数据4.3万。数据链接，数据第一个账号，第二个密码，95%可登录，可以登录官网查看数据是否准确"
    # 复杂网页清洗
    text1 = "发布了头条文章：《【XT】每日开工链新事儿 06.30 星期二》  [http://t.cn/A6LsKirA#区块链[超话]#](http://t.cn/A6LsKirA#%E5%8C%BA%E5%9D%97%E9%93%BE[%E8%B6%85%E8%AF%9D]#) #数字货币[超话]# #买价值币，只选XT# #比特币[超话]# #XT每日开工链新事儿? 06.30# #腾讯回应起诉老干妈#"
    text2 = ht.clean_text(text1, markdown_hyperlink=True, weibo_topic=True)
    print("清洗前：", [text1])
    print("清洗后：", [text2])
    assert text2 == "发布了头条文章：《【XT】每日开工链新事儿 06.30 星期二》"
    # 自定义正则表达式补充清洗
    text1 = "【#马化腾状告陶华碧#，#腾讯请求查封贵州老于妈公司1624万财产#】6月30日，据中国裁判文书网，【】广东省深圳市南山区人民法院发布一则民事裁定书" 
    text2 = ht.clean_text(text1, custom_regex=r"【.*?】")
    print("清洗前：", [text1])
    print("清洗后：", [text2])
    assert text2 == "6月30日，据中国裁判文书网，广东省深圳市南山区人民法院发布一则民事裁定书"
    text1 = "#嘎龙[超话]#【云次方/嘎龙】 回忆录?!1-2 http://t.cn/A6yvkujb 3 http://t.cn/A6yvkGO 4 http://t.cn/A6yZ59m0" 
    text2 = ht.clean_text(text1, weibo_topic=True, custom_regex=[r"【.*?】", r'[0-9\-]* +http[s]?://(?:[a-zA-Z]|[0-9]|[#$%*-;=?&@~.&+]|[!*,])+'])
    print("清洗前：", [text1])
    print("清洗后：", [text2])
    assert text2 == "回忆录?!"

if __name__ == "__main__":
    test_hard_text_cleaning()