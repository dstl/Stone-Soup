import pytest

pytest.importorskip('confluent_kafka')
from ..kafka import KafkaWriter  # noqa: E402


def test_writer_default():
    # Verify that the writer can be instantiated
    kafka_config = {
        "bootstrap.servers": "localhost:9092",
        "delivery.timeout.ms": 1,   # This is required, since actual sending of data is not tested
    }
    writer = KafkaWriter(
        kafka_config=kafka_config,
    )
    assert writer.kafka_config == kafka_config

    # This is a test to see if the write method can be called without errors
    # No data is actually written to the topic, since no Kafka server is running
    writer.write("test_topic", {"x": 1, "y": 2}, flush=False)
    writer.write("test_topic", {"x": 3, "y": 4})
    writer.write("test_topic", {"x": 5, "y": 6})
