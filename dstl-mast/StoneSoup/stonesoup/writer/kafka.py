import json
from typing import Dict

try:
    from confluent_kafka import Producer
except ImportError as error:  # pragma: no cover
    raise ImportError(
        "Kafka Writers require the dependency 'confluent-kafka' to be installed."
    ) from error

from .base import Writer
from ..base import Property


class KafkaWriter(Writer):
    """A simple Kafka writer that writes data to a Kafka topic.

    Parameters
    ----------
    """
    kafka_config: Dict[str, str] = Property(
        doc="Configuration properties for the underlying kafka consumer. See the "
            "`confluent-kafka documentation <https://docs.confluent.io/platform/current/clients/confluent-kafka-python/html/index.html#kafka-client-configuration>`_ " # noqa
            "for more details.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._producer = Producer(self.kafka_config)

    def write(self, topic, data, flush=True):
        as_json = json.dumps(data)
        self._producer.produce(topic, as_json)
        if flush:
            self._producer.flush()
