import pytest
def f():
    from harvesttext import HarvestText
    ht = HarvestText()
    entity_mention_dict = {'武磊': ['武磊', '武球王'], '郜林': ['郜林', '郜飞机'], '前锋': ['前锋'], '上海上港': ['上港'], '广州恒大': ['恒大'],
                               '单刀球': ['单刀']}
    entity_type_dict = {'武磊': '球员', '郜林': '球员', '前锋': '位,置', '上海上港': '球队', '广州恒大': '球队', '单刀球': '术语'}
    ht.add_entities(entity_mention_dict, entity_type_dict)

def test_mytest():
    with pytest.raises(Exception):
        f()
