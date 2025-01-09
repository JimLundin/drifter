"""Tests for the schema functionality."""


import pytest

from drifter.core import (
    BinaryField,
    BooleanField,
    CategoricalField,
    DateField,
    DatetimeField,
    DurationField,
    EnumField,
    FloatField,
    IntegerField,
    NullField,
    Schema,
    SequenceField,
    StringField,
    StructField,
    TimeField,
)


def test_numeric_fields() -> None:
    """Test numeric field types."""
    schema = Schema(
        fields={
            "int8": IntegerField(nullable=False, bits=8, signed=True),
            "int16": IntegerField(nullable=True, bits=16, signed=True),
            "int32": IntegerField(nullable=False, bits=32, signed=True),
            "int64": IntegerField(nullable=True, bits=64, signed=True),
            "uint8": IntegerField(nullable=False, bits=8, signed=False),
            "uint16": IntegerField(nullable=True, bits=16, signed=False),
            "uint32": IntegerField(nullable=False, bits=32, signed=False),
            "uint64": IntegerField(nullable=True, bits=64, signed=False),
            "float32": FloatField(nullable=False, bits=32),
            "float64": FloatField(nullable=True, bits=64),
        },
    )

    data = schema.to_dict()
    restored = Schema.from_dict(data)
    assert restored == schema


def test_temporal_fields() -> None:
    """Test temporal field types."""
    schema = Schema(
        fields={
            "date": DateField(nullable=False),
            "time": TimeField(nullable=True),
            "datetime": DatetimeField(nullable=False),
            "duration": DurationField(nullable=True),
        },
    )

    data = schema.to_dict()
    restored = Schema.from_dict(data)
    assert restored == schema


def test_string_fields() -> None:
    """Test string field types."""
    schema = Schema(
        fields={
            "string": StringField(nullable=False),
            "category": CategoricalField(nullable=True, ordered=True),
            "enum": EnumField(nullable=False, variants=["a", "b", "c"]),
        },
    )

    data = schema.to_dict()
    restored = Schema.from_dict(data)
    assert restored == schema


def test_other_fields() -> None:
    """Test other field types."""
    schema = Schema(
        fields={
            "binary": BinaryField(nullable=False),
            "boolean": BooleanField(nullable=True),
            "null": NullField(nullable=True),
        },
    )

    data = schema.to_dict()
    restored = Schema.from_dict(data)
    assert restored == schema


def test_sequence_field() -> None:
    """Test sequence fields with and without size."""
    # Variable-length sequence
    variable = SequenceField(
        nullable=False,
        inner=IntegerField(nullable=True, bits=32, signed=True),
        size=None,
    )

    # Fixed-length sequence
    fixed = SequenceField(
        nullable=True,
        inner=StringField(nullable=False),
        size=6,  # Fixed length of 6 elements
    )

    # Nested sequences
    nested = SequenceField(
        nullable=True,
        inner=SequenceField(
            nullable=False,
            inner=FloatField(nullable=True, bits=64),
            size=4,  # Fixed inner length
        ),
        size=None,  # Variable outer length
    )

    schema = Schema(fields={"variable": variable, "fixed": fixed, "nested": nested})
    data = schema.to_dict()
    restored = Schema.from_dict(data)
    assert restored == schema


def test_struct_field() -> None:
    """Test struct fields."""
    # Simple struct
    simple = StructField(
        nullable=False,
        fields={
            "id": IntegerField(nullable=False, bits=32, signed=True),
            "name": StringField(nullable=False),
            "email": StringField(nullable=True),
        },
    )

    # Nested struct with sequences
    nested = StructField(
        nullable=True,
        fields={
            "id": IntegerField(nullable=False, bits=32, signed=True),
            "values": SequenceField(
                nullable=True,
                inner=FloatField(nullable=False, bits=64),
                size=9,  # Fixed size sequence
            ),
            "tags": SequenceField(
                nullable=True,
                inner=StringField(nullable=False),
                size=None,  # Variable length
            ),
        },
    )

    schema = Schema(fields={"simple": simple, "nested": nested})
    data = schema.to_dict()
    restored = Schema.from_dict(data)
    assert restored == schema


def test_invalid_field_type() -> None:
    """Test handling of invalid field types."""
    with pytest.raises(ValueError, match="Unknown field type: invalidtype"):
        Schema.from_dict(
            {
                "fields": {
                    "invalid": {
                        "type": "invalidtype",
                        "nullable": True,
                    },
                },
            },
        )


def test_complex_nested_schema() -> None:
    """Test a complex schema with multiple levels of nesting."""
    schema = Schema(
        fields={
            "metadata": StructField(
                nullable=False,
                fields={
                    "version": IntegerField(nullable=False, bits=32, signed=True),
                    "categories": SequenceField(
                        nullable=True,
                        inner=CategoricalField(nullable=False, ordered=True),
                        size=None,  # Variable length
                    ),
                },
            ),
            "data": SequenceField(
                nullable=False,
                inner=StructField(
                    nullable=False,
                    fields={
                        "id": IntegerField(nullable=False, bits=64, signed=True),
                        "timestamp": DatetimeField(nullable=False),
                        "measurements": SequenceField(
                            nullable=True,
                            inner=StructField(
                                nullable=False,
                                fields={
                                    "sensor_id": StringField(nullable=False),
                                    "value": FloatField(nullable=False, bits=64),
                                    "status": EnumField(
                                        nullable=True,
                                        variants=["ok", "error", "unknown"],
                                    ),
                                },
                            ),
                            size=10,  # Fixed-size measurement sequence
                        ),
                    },
                ),
                size=None,  # Variable length data sequence
            ),
        },
    )

    data = schema.to_dict()
    restored = Schema.from_dict(data)
    assert restored == schema
