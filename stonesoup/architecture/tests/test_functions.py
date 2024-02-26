
from .._functions import _dict_set, _default_label_gen
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
    node = nodes['a']
    label = next(_default_label_gen(type(node)))
    assert label == 'Node\nA'

    repeater = RepeaterNode()
    gen = _default_label_gen(type(repeater))
    label = [next(gen) for i in range(26)][-1]  # A-Z 26 chars
    assert label.split("\n")[-1] == 'Z'
    label = next(gen)
    assert label == 'RepeaterNode\nAA'
