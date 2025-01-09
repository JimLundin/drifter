"""Polars schema conversion utilities."""

from collections.abc import Mapping

import polars as pl
from polars._typing import PolarsDataType

from drifter.core import (
    BinaryField,
    BooleanField,
    CategoricalField,
    DateField,
    DatetimeField,
    EnumField,
    Field,
    FloatField,
    IntegerField,
    SequenceField,
    StringField,
    StructField,
)


def _map_unsigned_dtype(
    dtype: PolarsDataType,
    *,
    nullable: bool = True,
) -> Field | None:
    """Map unsigned Polars dtypes to drifter Fields."""
    match dtype:
        case pl.UInt8():
            return IntegerField(nullable=nullable, bits=8, signed=False)
        case pl.UInt16():
            return IntegerField(nullable=nullable, bits=16, signed=False)
        case pl.UInt32():
            return IntegerField(nullable=nullable, bits=32, signed=False)
        case pl.UInt64():
            return IntegerField(nullable=nullable, bits=64, signed=False)
        case _:
            return None


def _map_signed_dtype(dtype: PolarsDataType, *, nullable: bool = True) -> Field | None:
    """Map signed Polars dtypes to drifter Fields."""
    match dtype:
        case pl.Int8():
            return IntegerField(nullable=nullable, bits=8, signed=True)
        case pl.Int16():
            return IntegerField(nullable=nullable, bits=16, signed=True)
        case pl.Int32():
            return IntegerField(nullable=nullable, bits=32, signed=True)
        case pl.Int64():
            return IntegerField(nullable=nullable, bits=64, signed=True)
        case _:
            return None


def _map_float_dtype(dtype: PolarsDataType, *, nullable: bool = True) -> Field | None:
    """Map float Polars dtypes to drifter Fields."""
    match dtype:
        case pl.Float32():
            return FloatField(nullable=nullable, bits=32)
        case pl.Float64():
            return FloatField(nullable=nullable, bits=64)
        case _:
            return None


def _map_temporal_dtype(
    dtype: PolarsDataType,
    *,
    nullable: bool = True,
) -> Field | None:
    """Map temporal Polars dtypes to drifter Fields."""
    match dtype:
        case pl.Date() | pl.Date:
            return DateField(nullable=nullable)
        case pl.Datetime() | pl.Datetime:
            return DatetimeField(nullable=nullable)
        case pl.Time() | pl.Time:
            return DatetimeField(nullable=nullable)
        case pl.Duration() | pl.Duration:
            return DatetimeField(nullable=nullable)
        case _:
            return None


def _map_string_dtype(dtype: PolarsDataType, *, nullable: bool = True) -> Field | None:
    """Map string-like Polars dtypes to drifter Fields."""
    match dtype:
        case pl.String() | pl.Utf8():
            return StringField(nullable=nullable)
        case pl.Binary():
            return BinaryField(nullable=nullable)
        case pl.Categorical():
            return CategoricalField(nullable=nullable, ordered=True)
        case pl.Enum():
            return EnumField(nullable=nullable, variants=list(dtype.categories))
        case _:
            return None


def _map_nested_dtype(
    dtype: PolarsDataType,
    *,
    nullable: bool = True,
) -> Field | None:
    """Map nested Polars dtypes to drifter Fields."""
    match dtype:
        case pl.List() | pl.Array():
            inner = _map_polars_dtype(dtype.inner, nullable=nullable)
            return SequenceField(
                nullable=nullable,
                inner=inner,
                size=None,
            )
        case pl.Array():
            inner = _map_polars_dtype(dtype.inner, nullable=nullable)
            return SequenceField(
                nullable=nullable,
                inner=inner,
                size=dtype.width,
            )
        case pl.Struct():
            fields = {
                field.name: _map_polars_dtype(field.dtype, nullable=nullable)
                for field in dtype.fields
            }
            return StructField(nullable=nullable, fields=fields)
        case _:
            return None


def _map_polars_dtype(dtype: PolarsDataType, *, nullable: bool = True) -> Field:
    """Map a Polars dtype to a drifter Field."""
    # Try each category of types
    match dtype:
        case pl.Boolean():
            return BooleanField(nullable=nullable)
        case pl.Null():
            return StringField(nullable=True)  # Always nullable
        case pl.Object() | pl.Object | pl.Unknown() | pl.Unknown:
            msg = f"Unsupported Polars dtype: {dtype}"
            raise TypeError(msg)
        case _:
            pass

    for mapper in [
        _map_signed_dtype,
        _map_unsigned_dtype,
        _map_float_dtype,
        _map_temporal_dtype,
        _map_string_dtype,
        _map_nested_dtype,
    ]:
        if result := mapper(dtype, nullable=nullable):
            return result

    msg = f"Unsupported Polars dtype: {dtype}"
    raise TypeError(msg)


def from_polars_schema(df: pl.DataFrame) -> Mapping[str, Field]:
    """Convert a Polars schema to a drifter schema.

    Args:
        df: A Polars DataFrame.

    Returns:
        A mapping from column names to drifter Fields.

    Raises:
        ValueError: If a Polars dtype is not supported.

    """
    return {
        name: _map_polars_dtype(dtype, nullable=True)  # All Polars fields are nullable
        for name, dtype in df.schema.items()
    }
