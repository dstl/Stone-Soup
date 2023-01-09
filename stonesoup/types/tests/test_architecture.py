from ..architecture import Architecture, NetworkArchitecture, InformationArchitecture, \
    CombinedArchitecture, ProcessingNode, RepeaterNode

import pytest


def test_information_architecture():
    with pytest.raises(TypeError):
        InformationArchitecture(node_set={RepeaterNode()})
