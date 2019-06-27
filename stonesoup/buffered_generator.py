class BufferedGenerator:
    @staticmethod
    def generator_method(method):
        method.is_generator = True
        return method

    def _get_methods(self):
        for item in map(lambda name: getattr(self, name), dir(self)):
            if callable(item):
                yield item

    def __iter__(self, *args, **kwargs):
        for method in self._get_methods():
            if hasattr(method, 'is_generator'):
                self._gen = method(*args, **kwargs)
                return self

    def __next__(self):
        self.current = next(self._gen)
        return self.current
