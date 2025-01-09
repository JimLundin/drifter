"""Tests for the schema functionality."""


import pytest

from drifter.core import (
    BinaryField,
    BooleanField,
    DateField,
    DatetimeField,
    FloatField,
    IntegerField,
    ListField,
    Schema,
    StringField,
    StructField,
    TimeField,
    TimestampField,
)


def test_simple_fields() -> None:
    """Test creation and serialization of simple field types."""
    fields = {
        "int_field": IntegerField(nullable=False, bits=32, signed=True),
        "float_field": FloatField(nullable=True, bits=64),
        "string_field": StringField(nullable=True),
        "bool_field": BooleanField(nullable=False),
        "binary_field": BinaryField(nullable=True),
        "date_field": DateField(nullable=True),
        "time_field": TimeField(nullable=True),
        "timestamp_field": TimestampField(nullable=True),
        "datetime_field": DatetimeField(nullable=True),
    }

    schema = Schema(fields=fields)
    data = schema.to_dict()
    restored = Schema.from_dict(data)
    assert restored == schema


def test_list_field() -> None:
    """Test creation and serialization of list fields."""
    # Simple list
    int_list = ListField(
        nullable=False,
        inner=IntegerField(nullable=True, bits=32, signed=True),
    )

    # Nested list
    nested_list = ListField(
        nullable=True,
        inner=ListField(
            nullable=True,
            inner=StringField(nullable=False),
        ),
    )

    schema = Schema(fields={"ints": int_list, "strings": nested_list})
    data = schema.to_dict()
    restored = Schema.from_dict(data)
    assert restored == schema


def test_struct_field() -> None:
    """Test creation and serialization of struct fields."""
    # Simple struct
    person = StructField(
        nullable=False,
        fields={
            "id": IntegerField(nullable=False, bits=32, signed=True),
            "name": StringField(nullable=False),
            "email": StringField(nullable=True),
        },
    )

    # Nested struct with list
    company = StructField(
        nullable=True,
        fields={
            "id": IntegerField(nullable=False, bits=32, signed=True),
            "name": StringField(nullable=False),
            "employees": ListField(
                nullable=True,
                inner=StructField(
                    nullable=False,
                    fields={
                        "id": IntegerField(nullable=False, bits=32, signed=True),
                        "name": StringField(nullable=False),
                    },
                ),
            ),
        },
    )

    schema = Schema(fields={"person": person, "company": company})
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
                    "tags": ListField(
                        nullable=True,
                        inner=StringField(nullable=False),
                    ),
                },
            ),
            "data": ListField(
                nullable=False,
                inner=StructField(
                    nullable=False,
                    fields={
                        "id": IntegerField(nullable=False, bits=64, signed=True),
                        "timestamp": TimestampField(nullable=False),
                        "measurements": ListField(
                            nullable=True,
                            inner=StructField(
                                nullable=False,
                                fields={
                                    "sensor_id": StringField(nullable=False),
                                    "value": FloatField(nullable=False, bits=64),
                                    "quality": BooleanField(nullable=True),
                                },
                            ),
                        ),
                    },
                ),
            ),
        },
    )

    data = schema.to_dict()
    restored = Schema.from_dict(data)
    assert restored == schema
