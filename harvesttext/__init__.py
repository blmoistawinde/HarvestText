#coding=utf-8
#!/usr/bin/env python
from .harvesttext import HarvestText
from .resources import *

import pickle
def saveHT(htModel,filename):
    with open(filename, "wb") as f:
        htModel.prepared = False
        htModel.hanlp_prepared = False
        pickle.dump(htModel,f)

def loadHT(filename):
    with open(filename, "rb") as f:
        ht = pickle.load(f)
    ht.prepare()
    return ht

