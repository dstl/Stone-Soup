from ..augmentor.base import Augmentor


class IdentityAugmentor(Augmentor):

    def augment(self, states):
        return states
