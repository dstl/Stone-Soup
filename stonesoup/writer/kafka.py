import json
from typing import Dict

from confluent_kafka import Producer
from stonesoup.base import Property
from stonesoup.writer import Writer


class KafkaWriter(Writer):
    """Write data to a Kafka topic"""
    kafka_config: Dict[str, str] = Property(
        default={}, doc="Keyword arguments for the underlying kafka producer"
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._producer = Producer(self.kafka_config)

    def write(self, topic, data, flush=True):
        as_json = json.dumps(data)
        self._producer.produce(topic, as_json)
        if flush:
            self._producer.flush()
