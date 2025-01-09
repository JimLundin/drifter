"""Tests for Polars schema conversion."""

import datetime

import polars as pl

from drifter.core import (
    BinaryField,
    BooleanField,
    CategoricalField,
    DateField,
    DatetimeField,
    FloatField,
    IntegerField,
    SequenceField,
    StringField,
    StructField,
)
from drifter.polars import from_polars_schema


def test_numeric_types() -> None:
    """Test conversion of numeric types."""
    df1 = pl.DataFrame(
        {
            "float32": [1.0],
            "float64": [1.0],
            "int8": [1],
            "int16": [1],
            "int32": [1],
            "int64": [1],
            "uint8": [1],
            "uint16": [1],
            "uint32": [1],
            "uint64": [1],
        },
        schema={
            "float32": pl.Float32,
            "float64": pl.Float64,
            "int8": pl.Int8,
            "int16": pl.Int16,
            "int32": pl.Int32,
            "int64": pl.Int64,
            "uint8": pl.UInt8,
            "uint16": pl.UInt16,
            "uint32": pl.UInt32,
            "uint64": pl.UInt64,
        },
    )

    fields = from_polars_schema(df1)

    assert isinstance(fields["float32"], FloatField)
    assert fields["float32"].bits == 32
    assert isinstance(fields["float64"], FloatField)
    assert fields["float64"].bits == 64

    assert isinstance(fields["int8"], IntegerField)
    assert fields["int8"].bits == 8
    assert fields["int8"].signed is True
    assert isinstance(fields["int16"], IntegerField)
    assert fields["int16"].bits == 16
    assert fields["int16"].signed is True
    assert isinstance(fields["int32"], IntegerField)
    assert fields["int32"].bits == 32
    assert fields["int32"].signed is True
    assert isinstance(fields["int64"], IntegerField)
    assert fields["int64"].bits == 64
    assert fields["int64"].signed is True

    assert isinstance(fields["uint8"], IntegerField)
    assert fields["uint8"].bits == 8
    assert fields["uint8"].signed is False
    assert isinstance(fields["uint16"], IntegerField)
    assert fields["uint16"].bits == 16
    assert fields["uint16"].signed is False
    assert isinstance(fields["uint32"], IntegerField)
    assert fields["uint32"].bits == 32
    assert fields["uint32"].signed is False
    assert isinstance(fields["uint64"], IntegerField)
    assert fields["uint64"].bits == 64
    assert fields["uint64"].signed is False


def test_string_types() -> None:
    """Test conversion of string types."""
    df1 = pl.DataFrame(
        {
            "string": ["hello"],
            "binary": [b"world"],
        },
        schema={
            "string": pl.String,
            "binary": pl.Binary,
        },
    )

    fields = from_polars_schema(df1)

    assert isinstance(fields["string"], StringField)
    assert isinstance(fields["binary"], BinaryField)


def test_temporal_types() -> None:
    """Test conversion of temporal types."""
    df1 = pl.DataFrame(
        {
            "date": [datetime.date(2021, 1, 1)],
            "datetime": [datetime.datetime(2021, 1, 1, 12, 0, tzinfo=datetime.UTC)],
            "duration": [datetime.timedelta(days=1)],
            "time": [datetime.time(12, 0)],
        },
        schema={
            "date": pl.Date,
            "datetime": pl.Datetime,
            "duration": pl.Duration,
            "time": pl.Time,
        },
    )

    schema = from_polars_schema(df1)

    assert isinstance(schema["date"], DateField)
    assert isinstance(schema["datetime"], DatetimeField)
    assert isinstance(schema["duration"], DatetimeField)
    assert isinstance(schema["time"], DatetimeField)


def test_boolean_type() -> None:
    """Test conversion of boolean type."""
    df1 = pl.DataFrame({"bool": [True]})
    fields = from_polars_schema(df1)
    assert isinstance(fields["bool"], BooleanField)


def test_struct_type() -> None:
    """Test conversion of struct type."""
    df1 = pl.DataFrame(
        {
            "nested": [
                {
                    "id": 1,
                    "name": "test",
                    "inner": {"value": 1.0},
                },
            ],
        },
        schema={
            "nested": pl.Struct(
                {
                    "id": pl.Int32,
                    "name": pl.String,
                    "inner": pl.Struct({"value": pl.Float64}),
                },
            ),
        },
    )
    fields = from_polars_schema(df1)

    assert isinstance(fields["nested"], StructField)
    nested = fields["nested"]
    assert isinstance(nested.fields["id"], IntegerField)
    assert isinstance(nested.fields["name"], StringField)
    assert isinstance(nested.fields["inner"], StructField)
    assert isinstance(nested.fields["inner"].fields["value"], FloatField)


def test_nullability() -> None:
    """Test handling of nullable fields."""
    df1 = pl.DataFrame(
        {
            "int_with_null": [None],
            "int_without_null": [1],
        },
        schema={
            "int_with_null": pl.Int32,
            "int_without_null": pl.Int32,
        },
    )

    fields = from_polars_schema(df1)

    # All fields in Polars are nullable
    assert fields["int_with_null"].nullable is True
    assert fields["int_without_null"].nullable is True


def test_list_type() -> None:
    """Test conversion of list type."""
    df1 = pl.DataFrame(
        {
            "list_int": [[1, 2, 3]],
            "list_str": [["a", "b", "c"]],
            "list_float": [[1.0, 2.0, 3.0]],
        },
    )
    schema = from_polars_schema(df1)

    assert isinstance(schema["list_int"], SequenceField)
    assert isinstance(schema["list_int"].inner, IntegerField)
    assert schema["list_int"].size is None
    assert schema["list_int"].nullable is True

    assert isinstance(schema["list_str"], SequenceField)
    assert isinstance(schema["list_str"].inner, StringField)
    assert schema["list_str"].size is None
    assert schema["list_str"].nullable is True

    assert isinstance(schema["list_float"], SequenceField)
    assert isinstance(schema["list_float"].inner, FloatField)
    assert schema["list_float"].size is None
    assert schema["list_float"].nullable is True


def test_categorical_type() -> None:
    """Test conversion of categorical type."""
    df1 = pl.DataFrame(
        {
            "cat": pl.Series(["a", "b", "c"], dtype=pl.Categorical),
        },
    )
    schema = from_polars_schema(df1)

    assert isinstance(schema["cat"], CategoricalField)
    assert schema["cat"].ordered is True
    assert schema["cat"].nullable is True
