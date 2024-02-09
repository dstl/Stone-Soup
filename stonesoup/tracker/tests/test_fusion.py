from ..fusion import SimpleFusionTracker
from ..simple import MultiTargetTracker


def test_fusion_tracker(initiator, deleter, detector, data_associator, updater):
    base_tracker = MultiTargetTracker(
        initiator, deleter, detector, data_associator, updater)
    _ = SimpleFusionTracker(base_tracker, 30)
    assert True
