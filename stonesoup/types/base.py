import weakref

from ..base import Base


class Type(Base):
    """Base type"""

    def _weaken_history(self):
        """Weaken references held to historical objects.

        This is called on an object when a new object is created from it (e.g. a prediction
        from a prior), such that the new object strongly references this object, but this
        object only weakly references the objects it was in turn created from. This avoids a
        chain of strong references retaining the entire history in memory.

        Subclasses should extend this to weaken their own references, calling
        :code:`super()._weaken_history()` so behaviour combines across multiple inheritance.
        """

    def _weaken_property(self, name):
        """Replace the value of property `name` with a weak reference to it.

        The property should have a getter which resolves weak references."""
        property_name = f'_property_{name}'
        value = getattr(self, property_name)
        if value is not None and not isinstance(value, weakref.ReferenceType):
            setattr(self, property_name, weakref.ref(value))
