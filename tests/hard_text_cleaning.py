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
    

if __name__ == "__main__":
    test_hard_text_cleaning()