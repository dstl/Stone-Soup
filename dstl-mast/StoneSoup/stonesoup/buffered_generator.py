import inspect


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

    def __iter__(self):
        for _, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if getattr(method, 'is_generator', False):
                for data in method():
                    self.current = data
                    yield self.current
                return
        raise AttributeError('Generator method undefined!')
