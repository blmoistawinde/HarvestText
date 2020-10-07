import re
import jieba
import jieba.posseg
from harvesttext import HarvestText

def AllEnglish():
    rule = lambda x: bool(re.fullmatch(r"[a-zA-Z]*",x))
    return rule

def AllEnglishOrNum():
    rule = lambda x: bool(re.fullmatch(r"[a-zA-Z0-9]*",x))
    return rule

def UpperFirst():
    rule = lambda x: bool(re.fullmatch(r"[A-Z]",x[0]))
    return rule

def StartsWith(prefix):
    return (lambda x: x.startswith(prefix))

def EndsWith(suffix):
    return (lambda x: x.endswith(suffix))

def Contains(span):
    rule = lambda x: bool(re.search(span,x))
    return rule

def WithLength(length):
    return (lambda x: len(x) == length)

