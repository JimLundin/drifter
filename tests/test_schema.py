"""Tests for the schema functionality."""


import pytest

from drifter.core import (
    ArrayField,
    BinaryField,
    BooleanField,
    CategoricalField,
    DateField,
    DatetimeField,
    DurationField,
    EnumField,
    FloatField,
    IntegerField,
    ListField,
    NullField,
    Schema,
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


def test_array_field() -> None:
    """Test array fields with shape and width."""
    # Simple array
    simple = ArrayField(
        nullable=False,
        inner=IntegerField(nullable=True, bits=32, signed=True),
        shape=[2, 3],
        width=6,
    )

    # Nested array
    nested = ArrayField(
        nullable=True,
        inner=ArrayField(
            nullable=True,
            inner=StringField(nullable=False),
            shape=[4],
            width=4,
        ),
        shape=[2],
        width=2,
    )

    schema = Schema(fields={"simple": simple, "nested": nested})
    data = schema.to_dict()
    restored = Schema.from_dict(data)
    assert restored == schema


def test_list_field() -> None:
    """Test list fields."""
    # Simple list
    simple = ListField(
        nullable=False,
        inner=IntegerField(nullable=True, bits=32, signed=True),
    )

    # Nested list
    nested = ListField(
        nullable=True,
        inner=ListField(
            nullable=True,
            inner=StringField(nullable=False),
        ),
    )

    schema = Schema(fields={"simple": simple, "nested": nested})
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

    # Nested struct with array and list
    nested = StructField(
        nullable=True,
        fields={
            "id": IntegerField(nullable=False, bits=32, signed=True),
            "tags": ArrayField(
                nullable=True,
                inner=StringField(nullable=False),
                shape=[3],
                width=3,
            ),
            "metrics": ListField(
                nullable=True,
                inner=StructField(
                    nullable=False,
                    fields={
                        "name": StringField(nullable=False),
                        "value": FloatField(nullable=False, bits=64),
                    },
                ),
            ),
        },
    )

    schema = Schema(fields={"simple": simple, "nested": nested})
    data = schema.to_dict()
    restored = Schema.from_dict(data)
    assert restored == schema


def test_invalid_field_type() -> None:
    """Test handling of invalid field types."""
    with pytest.raises(ValueError, match="Unknown field type: InvalidType"):
        Schema.from_dict(
            {
                "fields": {
                    "invalid": {
                        "type": "InvalidType",
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
                    "categories": ListField(
                        nullable=True,
                        inner=CategoricalField(nullable=False, ordered=True),
                    ),
                },
            ),
            "data": ListField(
                nullable=False,
                inner=StructField(
                    nullable=False,
                    fields={
                        "id": IntegerField(nullable=False, bits=64, signed=True),
                        "timestamp": DatetimeField(nullable=False),
                        "measurements": ArrayField(
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
                            shape=[10],
                            width=10,
                        ),
                    },
                ),
            ),
        },
    )

    data = schema.to_dict()
    restored = Schema.from_dict(data)
    assert restored == schema
