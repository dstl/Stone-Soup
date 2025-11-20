import datetime

import pytest
from dateutil.parser import parse

pytest.importorskip('confluent_kafka')
from ..kafka import KafkaDetectionReader, KafkaGroundTruthReader  # noqa: E402


@pytest.fixture(params=[KafkaDetectionReader, KafkaGroundTruthReader])
def reader_cls(request):
    return request.param


@pytest.mark.parametrize(
    "metadata_fields",
    [
        ["sensor_id"],
        None
    ]
)
def test_reader_default(reader_cls, metadata_fields):

    kwargs = {
        "topic": "test_topic",
        "state_vector_fields": ["x", "y"],
        "time_field": "timestamp",
        "metadata_fields": metadata_fields,
        "kafka_config": {"bootstrap.servers": "localhost:9092"},
        "buffer_size": 10,
    }
    if reader_cls == KafkaGroundTruthReader:
        kwargs["path_id_field"] = "path_id"

    reader = reader_cls(**kwargs)

    assert reader.topic == "test_topic"
    assert reader.state_vector_fields == ["x", "y"]
    assert reader.time_field == "timestamp"
    assert reader.metadata_fields == metadata_fields
    assert reader.kafka_config == {"bootstrap.servers": "localhost:9092"}
    assert reader._buffer.maxsize == 10
    if reader_cls == KafkaGroundTruthReader:
        assert reader.path_id_field == "path_id"
        assert reader._non_metadata_fields == ["x", "y", "timestamp", "path_id"]
    else:
        assert reader._non_metadata_fields == ["x", "y", "timestamp"]

    all_data = [
        {"x": 1, "y": 2, "timestamp": "2020-01-01T00:00:00Z", "sensor_id": "sensor1"},
        {"x": 3, "y": 4, "timestamp": "2020-01-01T00:00:01Z", "sensor_id": "sensor1"},
        {"x": 5, "y": 6, "timestamp": "2020-01-01T00:00:02Z", "sensor_id": "sensor1"},
    ]
    if reader_cls == KafkaGroundTruthReader:
        all_data[0]["path_id"] = 1
        all_data[1]["path_id"] = 2
        all_data[2]["path_id"] = 3

    for data in all_data:
        detection = reader._parse_data(data)
        assert detection.state_vector[0] == data["x"]
        assert detection.state_vector[1] == data["y"]
        assert detection.timestamp == parse(data["timestamp"], ignoretz=True)
        assert len(detection.metadata) == 1
        assert detection.metadata["sensor_id"] == data["sensor_id"]


@pytest.mark.parametrize(
    "timestamps, timestamp, time_field_format",
    [
        (
            ["1514815200", "1514815200", "1514815200"],
            True,
            None
        ),
        (
            ["2018-01-01T14:00:00Z", "2018-01-01T14:00:00Z", "2018-01-01T14:00:00Z"],
            False,
            "%Y-%m-%dT%H:%M:%SZ"
        )
    ]
)
def test_reader_timestamp(reader_cls, timestamps, timestamp, time_field_format):
    kwargs = {
        "topic": "test_topic",
        "state_vector_fields": ["x", "y"],
        "time_field": "timestamp",
        "timestamp": timestamp,
        "time_field_format": time_field_format,
        "kafka_config": {"bootstrap.servers": "localhost:9092"},
        "buffer_size": 10,
    }
    if reader_cls == KafkaGroundTruthReader:
        kwargs["path_id_field"] = "path_id"

    reader = reader_cls(**kwargs)

    assert reader.time_field == "timestamp"
    assert reader.timestamp == timestamp
    assert reader.time_field_format == time_field_format

    all_data = [
        {"x": 1, "y": 2, "sensor_id": "sensor1"},
        {"x": 3, "y": 4, "sensor_id": "sensor1"},
        {"x": 5, "y": 6, "sensor_id": "sensor1"},
    ]

    for i, data in enumerate(all_data):
        data["timestamp"] = timestamps[i]
        if reader_cls == KafkaGroundTruthReader:
            data["path_id"] = i + 1

    for data in all_data:
        detection = reader._parse_data(data)
        assert detection.timestamp.hour == 14
        assert detection.timestamp.minute == 0
        assert detection.timestamp.date() == datetime.date(2018, 1, 1)
