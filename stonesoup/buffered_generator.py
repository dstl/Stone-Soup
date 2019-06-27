class BufferedGenerator:
    """
    Converts a class with a generator method into a buffered generator.
    The generator method to be used is decorated using
    BufferedGenerator.generator_method. This can then be used as expected

    .. code-block:: python

        class Foo(BufferedGenerator):
            '''
            Example Foo generator
            '''

            @BufferedGenerator.generator_method
            def count_to_ten(self):
                "Counts to ten"
                for i in range(10):
                    yield i + 1

        foo = Foo()
        for i in foo:
            print(i)

    The current state of the generator is available using the 'current'
    attribute.

    .. code-block:: python
        foo = Foo()
        for i in foo:
            print(i)
            print(foo.current)
    """
    @staticmethod
    def generator_method(method):
        method.is_generator = True
        return method

    def _get_methods(self):
        for item in map(lambda name: getattr(self, name), dir(self)):
            if callable(item):
                yield item

    def __iter__(self):
        for method in self._get_methods():
            if hasattr(method, 'is_generator'):
                self._gen = method()
                return self

    def __next__(self):
        self.current = next(self._gen)
        return self.current
