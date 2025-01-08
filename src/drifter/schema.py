"""Schema definition and change detection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict

import polars as pl

if TYPE_CHECKING:
    from polars._typing import PolarsDataType


class DType(StrEnum):
    """Base data types supported by the schema."""

    # Numeric types
    INT = auto()  # Int8, Int16, Int32, Int64, Int128
    UINT = auto()  # UInt8, UInt16, UInt32, UInt64
    FLOAT = auto()  # Float32, Float64
    DECIMAL = auto()  # Decimal128

    # String types
    STRING = auto()  # String/Utf8
    CATEGORICAL = auto()  # Categorical
    ENUM = auto()  # Enum

    # Temporal types
    DATE = auto()  # Date
    TIME = auto()  # Time
    DATETIME = auto()  # Datetime
    DURATION = auto()  # Duration

    # Other primitive types
    BOOL = auto()  # Boolean
    BINARY = auto()  # Binary
    NULL = auto()  # Null
    OBJECT = auto()  # Object
    UNKNOWN = auto()  # Unknown

    # Nested types
    LIST = auto()  # List
    ARRAY = auto()  # Array
    STRUCT = auto()  # Struct


type TimeUnit = Literal["ns", "us", "ms"]

class BaseField(TypedDict):
    """Base schema field with common attributes."""

    name: str
    type_hash: int


class PrimitiveField(BaseField):
    """Field for primitive types."""

    dtype: Literal[
        DType.STRING,
        DType.BOOL,
        DType.BINARY,
        DType.NULL,
        DType.OBJECT,
        DType.UNKNOWN,
    ]


class CategoricalField(BaseField):
    """Field for categorical types."""

    dtype: Literal[DType.CATEGORICAL]
    ordering: Literal["lexical", "physical"] | None


class EnumField(BaseField):
    """Field for enum types."""

    dtype: Literal[DType.ENUM]
    categories: list[str]


class NumericField(BaseField):
    """Field for numeric types with bit width."""

    dtype: Literal[DType.INT, DType.UINT, DType.FLOAT]
    bit_width: int


class DateField(BaseField):
    """Field for date types."""

    dtype: Literal[DType.DATE]


class TimeField(BaseField):
    """Field for time types."""

    dtype: Literal[DType.TIME]


class DatetimeField(BaseField):
    """Field for datetime types."""

    dtype: Literal[DType.DATETIME]
    time_unit: TimeUnit
    timezone: str | None


class DurationField(BaseField):
    """Field for duration types."""

    dtype: Literal[DType.DURATION]
    time_unit: TimeUnit


class ListField(BaseField):
    """Field for list types with inner type."""

    dtype: Literal[DType.LIST]
    inner: Field


class ArrayField(BaseField):
    """Field for array types with inner type."""

    dtype: Literal[DType.ARRAY]
    inner: Field
    size: int


class StructField(BaseField):
    """Field for struct types with field definitions."""

    dtype: Literal[DType.STRUCT]
    fields: dict[str, Field]


# Union type for all possible field types
Field = (
    PrimitiveField
    | CategoricalField
    | EnumField
    | NumericField
    | DateField
    | TimeField
    | DatetimeField
    | DurationField
    | ListField
    | ArrayField
    | StructField
)


class Schema(TypedDict):
    """The complete schema definition."""

    version: str
    fields: dict[str, Field]


@dataclass
class ColumnChange:
    """Represents a change in a column's schema."""

    field: str
    old: Field
    new: Field


@dataclass
class SchemaChanges:
    """Contains all detected schema changes between versions."""

    added: dict[str, Field]  # Added field names
    removed: dict[str, Field]  # Removed field names
    changed: dict[str, ColumnChange]  # Changed field types

    def has_changes(self) -> bool:
        """Check if there are any schema changes."""
        return bool(self.added or self.removed or self.changed)

    def __bool__(self) -> bool:
        """Boolean context - Check if there are any schema changes."""
        return self.has_changes()


def _get_base_dtype(dtype: PolarsDataType) -> tuple[DType, int | None]:
    """Get the base dtype and bit width from a Polars type."""
    # Map base types to our DType enum
    base_type_map = {
        pl.Utf8: (DType.STRING, None),
        pl.String: (DType.STRING, None),
        pl.Boolean: (DType.BOOL, None),
        pl.Date: (DType.DATE, None),
        pl.Time: (DType.TIME, None),
        pl.Datetime: (DType.DATETIME, None),
        pl.Duration: (DType.DURATION, None),
        pl.Binary: (DType.BINARY, None),
        pl.Null: (DType.NULL, None),
        pl.Object: (DType.OBJECT, None),
        pl.Unknown: (DType.UNKNOWN, None),
        pl.Categorical: (DType.CATEGORICAL, None),
        pl.Enum: (DType.ENUM, None),
        # Integer types
        pl.Int8: (DType.INT, 8),
        pl.Int16: (DType.INT, 16),
        pl.Int32: (DType.INT, 32),
        pl.Int64: (DType.INT, 64),
        pl.Int128: (DType.INT, 128),
        # Unsigned integer types
        pl.UInt8: (DType.UINT, 8),
        pl.UInt16: (DType.UINT, 16),
        pl.UInt32: (DType.UINT, 32),
        pl.UInt64: (DType.UINT, 64),
        # Float types
        pl.Float32: (DType.FLOAT, 32),
        pl.Float64: (DType.FLOAT, 64),
    }
    return base_type_map.get(dtype.base_type(), (DType.STRING, None))


def _polars_type_to_field(name: str, dtype: PolarsDataType) -> Field:
    """Convert a Polars type to our schema type system."""
    field: Field | None = None

    match dtype:
        case pl.List():
            field = ListField(
                name=name,
                dtype=DType.LIST,
                inner=_polars_type_to_field("inner", dtype.inner),
                type_hash=hash(dtype),
            )
        case pl.Array():
            field = ArrayField(
                name=name,
                dtype=DType.ARRAY,
                inner=_polars_type_to_field("inner", dtype.inner),
                size=dtype.size,
                type_hash=hash(dtype),
            )
        case pl.Struct():
            field = StructField(
                name=name,
                dtype=DType.STRUCT,
                fields={
                    field.name: _polars_type_to_field(field.name, field.dtype)
                    for field in dtype.fields
                },
                type_hash=hash(dtype),
            )
        case pl.Categorical():
            field = CategoricalField(
                name=name,
                dtype=DType.CATEGORICAL,
                ordering=dtype.ordering,
                type_hash=hash(dtype),
            )
        case pl.Enum():
            field = EnumField(
                name=name,
                dtype=DType.ENUM,
                categories=list(dtype.categories),
                type_hash=hash(dtype),
            )
        case pl.Date():
            field = DateField(
                name=name,
                dtype=DType.DATE,
                type_hash=hash(dtype),
            )
        case pl.Time():
            field = TimeField(
                name=name,
                dtype=DType.TIME,
                type_hash=hash(dtype),
            )
        case pl.Datetime():
            field = DatetimeField(
                name=name,
                dtype=DType.DATETIME,
                time_unit=dtype.time_unit,
                timezone=dtype.time_zone,
                type_hash=hash(dtype),
            )
        case pl.Duration():
            field = DurationField(
                name=name,
                dtype=DType.DURATION,
                time_unit=dtype.time_unit,
                type_hash=hash(dtype),
            )
        case _:
            # Get base type and bit width
            base_type, bit_width = _get_base_dtype(dtype)

            # Handle numeric types
            if base_type in (DType.INT, DType.UINT, DType.FLOAT):
                if bit_width is None:
                    msg = f"Unexpected bit width for numeric type: {base_type}"
                    raise ValueError(msg)

                field = NumericField(
                    name=name,
                    dtype=base_type,
                    bit_width=bit_width,
                    type_hash=hash(dtype),
                )
            # Handle primitive types
            elif base_type in (
                DType.STRING,
                DType.BOOL,
                DType.BINARY,
                DType.NULL,
                DType.OBJECT,
                DType.UNKNOWN,
            ):
                field = PrimitiveField(
                    name=name,
                    dtype=base_type,
                    type_hash=hash(dtype),
                )

    if field is None:
        msg = f"Unsupported type: {dtype}"
        raise ValueError(msg)
    return field


def _are_fields_equal(field1: Field, field2: Field) -> bool:
    """Compare two fields for equality using type hashes."""
    return field1["type_hash"] == field2["type_hash"]


def _create_schema(df: pl.DataFrame) -> Schema:
    """Create a schema from a DataFrame."""
    return {
        "version": "1.0",
        "fields": {
            name: _polars_type_to_field(name, dtype)
            for name, dtype in zip(df.columns, df.dtypes, strict=True)
        },
    }


def register(
    df: pl.DataFrame,
    name: str,
    schema_dir: str | Path = ".drifter",
) -> SchemaChanges:
    """Register a DataFrame's schema and detect any changes.

    Args:
        df: DataFrame to register schema for
        name: Unique identifier for this DataFrame's schema
        schema_dir: Directory to store schema history (default: .drifter)

    Returns:
        SchemaChanges object containing any detected changes

    """
    schema_dir = Path(schema_dir)
    schema_dir.mkdir(exist_ok=True)
    schema_file = schema_dir / f"{name}_schema_history.json"

    # Create current schema
    current_schema = _create_schema(df)

    # Initialize history if file doesn't exist
    if not schema_file.exists():
        with schema_file.open("w") as f:
            json.dump([current_schema], f, indent=2)
        return SchemaChanges(
            added=current_schema["fields"],
            removed={},
            changed={},
        )

    # Load history
    with schema_file.open() as f:
        history: list[Schema] = json.load(f)

    # Detect changes from previous schema
    prev_schema = history[-1]

    # Find added and removed fields
    added = {
        name: current_schema["fields"][name]
        for name in set(current_schema["fields"]) - set(prev_schema["fields"])
    }
    removed = {
        name: prev_schema["fields"][name]
        for name in set(prev_schema["fields"]) - set(current_schema["fields"])
    }

    # Find changed fields
    changed = {
        name: ColumnChange(
            field=name,
            old=prev_schema["fields"][name],
            new=current_schema["fields"][name],
        )
        for name in set(current_schema["fields"]) & set(prev_schema["fields"])
        if not _are_fields_equal(
            current_schema["fields"][name],
            prev_schema["fields"][name],
        )
    }

    # Create changes object
    changes = SchemaChanges(added=added, removed=removed, changed=changed)

    # Update history if there are changes
    if changes:
        history.append(current_schema)
        with schema_file.open("w") as f:
            json.dump(history, f, indent=2)

    return changes
