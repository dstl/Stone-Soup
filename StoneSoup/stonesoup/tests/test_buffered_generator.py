from stonesoup.buffered_generator import BufferedGenerator


class TestBuffer(BufferedGenerator):
    @BufferedGenerator.generator_method
    def create_numbers(self):
        for number in range(10):
            yield number


def test_generation():
    test = TestBuffer()
    for expected, actual in zip(range(10), test):
        assert expected == actual


def test_buffering():
    test = TestBuffer()
    for expected, actual in zip(range(10), (test.current for _ in test)):
        assert expected == actual
