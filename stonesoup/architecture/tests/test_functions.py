
from .._functions import _dict_set, _default_label, _default_letters
from ..node import RepeaterNode


def test_dict_set():
    d = dict()
    assert d == {}

    inc, d = _dict_set(d, "c", "cow")
    assert inc
    assert d == {"cow": {"c"}}
    inc, d = _dict_set(d, "o", "cow")
    assert inc
    assert d == {"cow": {"c", "o"}}
    inc, d = _dict_set(d, "c", "cow")
    assert not inc
    assert d == {"cow": {"c", "o"}}

    d2 = dict()
    assert d2 == {}

    inc, d2 = _dict_set(d2, "africa", "lion", "yes")
    assert inc
    assert d2 == {"lion": {"yes": {"africa"}}}

    inc, d2 = _dict_set(d2, "europe", "polar bear", "no")
    assert inc
    assert d2 == {"lion": {"yes": {"africa"}}, "polar bear": {"no": {"europe"}}}

    inc, d2 = _dict_set(d2, "europe", "lion", "no")
    assert inc
    assert d2 == {"lion": {"yes": {"africa"}, "no": {"europe"}}, "polar bear": {"no": {"europe"}}}

    inc, d2 = _dict_set(d2, "north america", "lion", "no")
    assert inc
    assert d2 == {"lion": {"yes": {"africa"}, "no": {"europe", "north america"}},
                  "polar bear": {"no": {"europe"}}}


def test_default_label(nodes):
    last_letters = {'Node': '', 'SensorNode': '', 'FusionNode': '', 'SensorFusionNode': '',
                    'RepeaterNode': 'Z'}
    node = nodes['a']
    label, last_letters = _default_label(node, last_letters)
    assert last_letters['Node'] == 'A'
    assert label == 'Node A'

    repeater = RepeaterNode()
    assert last_letters['RepeaterNode'] == 'Z'
    label, last_letters = _default_label(repeater, last_letters)
    assert last_letters['RepeaterNode'] == 'AA'
    assert label == 'RepeaterNode AA'


def test_default_letters():
    assert _default_letters('') == 'A'
    assert _default_letters('A') == 'B'
    assert _default_letters('Z') == 'AA'
    assert _default_letters('AA') == 'AB'
    assert _default_letters('AZ') == 'BA'
    assert _default_letters('ZZ') == 'AAA'
