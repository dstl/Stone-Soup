# -*- coding: utf-8 -*-

from .orbitalelements import OrbitalElements


class TwoLineElement(OrbitalElements):
    """
    TwoLineElement type

    A TwoLineElement type which contains an orbital state vector and meta data
    associated with the object. Is an extension of the OrbitalElements class.
    """

    def __init__(self, state_vector, metadata, *args, **kwargs):
        super().__init__(state_vector, *args, **kwargs)
        self.metadata = metadata

    @property
    def metadata(self):
        return self.metadata
